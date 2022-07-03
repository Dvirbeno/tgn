import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import dgl

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from modules.custom import EdgeUpdatingHeteroGraphConv, GNNLayer, TwoLayerHeteroGraphConv


class TGN(torch.nn.Module):
    def __init__(self, device,
                 n_edge_features,
                 n_node_features,
                 n_nodes,
                 n_layers=2,
                 n_heads=2,
                 dropout=0.1,
                 embedding_dimension=100,
                 message_dimension=100,
                 memory_dimension=500,
                 temporal_dim=100,
                 embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift=0, std_time_shift=1,
                 n_neighbors=None,
                 aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.n_edge_features = n_edge_features
        self.n_node_features = n_node_features
        self.temporal_dim = temporal_dim
        self.n_nodes = n_nodes

        self.embedding_dimension = embedding_dimension
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message

        self.time_encoder = TimeEncode(dimension=self.temporal_dim).to(self.device)

        self.mean_time_shift = mean_time_shift
        self.std_time_shift = std_time_shift

        self.memory_dimension = memory_dimension
        raw_message_dimension = self.memory_dimension + (
                self.n_edge_features + 1) + self.time_encoder.dimension  # (+1) for the result values attachted to feature edges
        message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
        self.memory = Memory(n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,
                             input_dimension=message_dimension,
                             message_dimension=message_dimension,
                             device=device)
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                         device=device)

        self.raw_message_processor = TwoLayerHeteroGraphConv(player_dim=self.memory_dimension,
                                                             match_dim=0,
                                                             edge_dim=(
                                                                              self.n_edge_features + 1) + self.time_encoder.dimension,
                                                             activation=F.elu,
                                                             hidden_dim=message_dimension,
                                                             out_dim=message_dimension).to(self.device)

        self.message_function = get_message_function(module_type=message_function,
                                                     raw_message_dimension=message_dimension,
                                                     message_dimension=message_dimension).to(self.device)
        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device).to(self.device)

        # self.pre_attn_lin_q = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)
        # self.pre_attn_lin_k = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)
        # self.pre_attn_lin_v = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)

        self.output_attn = torch.nn.MultiheadAttention(embed_dim=memory_dimension, num_heads=1,
                                                       batch_first=True, add_bias_kv=True).to(self.device)

        # TODO: lose it
        self.dummy_lin = torch.nn.Linear(self.memory_dimension, 1, device=self.device)
        #
        # self.embedding_module = get_embedding_module(module_type=embedding_module_type,
        #                                              node_features=self.node_raw_features,
        #                                              edge_features=self.edge_raw_features,
        #                                              memory=self.memory,
        #                                              neighbor_finder=self.neighbor_finder,
        #                                              time_encoder=self.time_encoder,
        #                                              n_layers=self.n_layers,
        #                                              n_node_features=self.n_node_features,
        #                                              n_edge_features=self.n_edge_features,
        #                                              n_time_features=self.temporal_dim,
        #                                              embedding_dimension=self.embedding_dimension,
        #                                              device=self.device,
        #                                              n_heads=n_heads, dropout=dropout,
        #                                              n_neighbors=self.n_neighbors)
        #
        # # MLP to compute probability on an edge given two node embeddings
        # self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
        #                                  self.n_node_features,
        #                                  1)

    def compute_temporal_embeddings(self, input_nodes, pair_graph, blocks, complete_graph):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        # which edge type? (assuming all edges in the batch are of the same type)
        for etype in pair_graph.canonical_etypes:
            if len(pair_graph.edata[dgl.EID][etype]) > 0:
                break
        # make sure only a single type of edges is available
        assert sum([len(pair_graph.edata[dgl.EID][etype]) > 0 for etype in pair_graph.canonical_etypes]) == 1

        is_player_dst = etype[-1] == 'player'
        node_ids_p = pair_graph.nodes['player'].data[dgl.NID]  # only the player related nodes
        node_ids_m = pair_graph.nodes['match'].data[dgl.NID]  # only the match related nodes

        edge_src, edge_dst = pair_graph.edges(etype=etype)
        batch_player_node_ids = (node_ids_p[edge_dst] if is_player_dst else node_ids_p[edge_src]).cpu().numpy()
        batch_match_node_ids = (node_ids_m[edge_src] if is_player_dst else node_ids_m[edge_dst]).cpu().numpy()
        edge_times = pair_graph.edges[etype].data['timestamp']

        all_relevant_nodes = np.concatenate([input_nodes['player'].numpy(),
                                             batch_player_node_ids])

        # Update memory for all nodes with messages stored in previous batches
        memory, last_update, updated_nodes = self.update_and_get_memory(all_relevant_nodes,
                                                                        batch_player_node_ids,
                                                                        complete_graph=complete_graph)

        ### Compute differences between the time the memory of a node was last updated,
        ### and the time for which we want to compute the embedding of a node
        time_diffs = edge_times - last_update
        norm_time_diffs = (time_diffs - self.mean_time_shift) / self.std_time_shift

        # Compute the embeddings using the embedding module
        if False:
            node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                     source_nodes=nodes,
                                                                     timestamps=timestamps,
                                                                     n_layers=self.n_layers,
                                                                     n_neighbors=n_neighbors,
                                                                     time_diffs=time_diffs)

            source_node_embedding = node_embedding[:n_samples]
            destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
            negative_node_embedding = node_embedding[2 * n_samples:]
        else:
            out_node_embeddings = memory
            out_node_embeddings[last_update == 0] = self.memory.default_memory

        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them) - This is now performed before
        # self.update_memory(positives, self.memory.raw_messages_storage)

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(updated_nodes)

        raw_messages = self.get_raw_messages(pair_graph,
                                             etype,
                                             batch_player_node_ids,
                                             batch_match_node_ids,
                                             edge_times
                                             )
        self.memory.store_raw_messages(np.unique(batch_player_node_ids), raw_messages)

        return out_node_embeddings

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def update_and_get_memory(self, all_relevant_node_ids,
                              out_node_ids,
                              complete_graph):
        # Aggregate messages for the same nodes
        nodes_to_update, raw_message_feats, message_timestamps, connector_node_id = \
            self.message_aggregator.aggregate(
                out_node_ids,
                self.memory.raw_messages_storage)

        if len(nodes_to_update) > 0:

            # TODO: verify this
            # ==================
            # sub_g = dgl.node_subgraph(complete_graph, {'match': connector_node_id})
            affected_nodes, propagated_messages, messages_ts = self.recurse_message_computation(nodes_to_update,
                                                                                                message_timestamps,
                                                                                                connector_node_id,
                                                                                                complete_graph)

            # ==================

            processed_messages = self.message_function.compute_message(propagated_messages)
        else:
            processed_messages = None
            affected_nodes = torch.empty(0)
            messages_ts = []

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(affected_nodes, processed_messages,
                                          timestamps=messages_ts)

        return self.memory.get_memory(out_node_ids), self.memory.last_update[out_node_ids], affected_nodes.numpy()

    def recurse_message_computation(self, induced_nodes, message_timestamps, connector_node_id, complete_graph):
        # sort the incoming messages
        message_timestamps, indices = torch.sort(message_timestamps)
        indices = indices.cpu().numpy()
        induced_nodes = induced_nodes[indices]
        connector_node_id = connector_node_id[indices]

        # we need to iterate for each connector_node_id once
        connector_to_players = defaultdict(list)
        connector_to_ts = defaultdict(list)
        for node_id, message_ts, conn_id in zip(induced_nodes, message_timestamps, connector_node_id):
            connector_to_players[conn_id].append(node_id)
            connector_to_ts[conn_id].append(message_ts.unsqueeze(0))

        for conn_id, conn_ts in connector_to_ts.items():
            connector_to_ts[conn_id] = torch.unique(torch.cat(conn_ts, 0))
        # make sure each connector is associated with a single unique ts
        assert all([len(conn_ts) == 1 for _, conn_ts in connector_to_ts.items()])

        all_affected_nodes = list()
        all_propagated_messages = list()
        propagated_messages_ts = list()

        for conn_id in connector_to_players:
            connector_subgraph = dgl.in_subgraph(complete_graph, {'match': [conn_id]})
            # remove all the unconnected (isolated) nodes - for getting the "pure" sub-graph:
            connector_subgraph = dgl.remove_nodes(connector_subgraph, (
                    connector_subgraph.in_degrees(etype='plays') == 0).nonzero().squeeze(), ntype='match')
            connector_subgraph = dgl.remove_nodes(connector_subgraph, (
                    connector_subgraph.out_degrees(etype='plays') == 0).nonzero().squeeze(), ntype='player')

            rel = connector_subgraph.edge_type_subgraph([('player', 'plays', 'match')])
            reverse_rel = dgl.reverse(rel, copy_edata=True)
            reversed_edges = reverse_rel.all_edges()
            connector_subgraph.add_edges(*reversed_edges, data=reverse_rel.edata,
                                         etype=('match', 'played_by', 'player'))

            # verify player nodes are ordered similarly on both types of edges
            assert torch.equal(connector_subgraph.edges(etype='plays')[0],
                               connector_subgraph.edges(etype='played_by')[-1])

            for type_of_edge in connector_subgraph.canonical_etypes:
                if connector_subgraph.num_nodes('player') != connector_subgraph.num_edges(type_of_edge):
                    # means that a player has two edges for the same match
                    seen_pid = []
                    delete_eid = []
                    u, v, e = connector_subgraph.edges('all', etype=type_of_edge)

                    is_player_dst = type_of_edge[-1] == 'player'

                    for idx_u, idx_v, idx_edge in zip(u, v, e):
                        idx_player = idx_v if is_player_dst else idx_u

                        if idx_player in seen_pid:
                            delete_eid.append(idx_edge)
                        else:
                            seen_pid.append(idx_player)
                    delete_eid = torch.stack(delete_eid)

                    if len(delete_eid) >= 1:
                        connector_subgraph.remove_edges(delete_eid, type_of_edge)

            affecting_nodes = connector_subgraph.nodes['player'].data[dgl.NID][
                connector_subgraph.edges(etype='plays')[0]]

            # make sure all to nodes we're interested in are vertices in the sliced subgraph
            assert all([node_id in affecting_nodes for node_id in connector_to_players[conn_id]])

            # Aggregate messages for the same nodes
            nodes_to_update, raw_message_feats, timestamps, connectors = \
                self.message_aggregator.aggregate(
                    affecting_nodes.numpy(),
                    self.memory.raw_messages_storage)
            assert all(
                connectors == conn_id), 'All the cached raw messages should be associated with the relevant node id'
            assert all(timestamps == connector_to_ts[conn_id]), 'Same for timestamps'
            assert all([nid in affecting_nodes for nid in nodes_to_update]) and len(affecting_nodes) == len(
                nodes_to_update), 'They should point to the same nodes, otherwise this update should have been performed earlier'

            connector_subgraph = connector_subgraph.to(self.device)

            prev_memory = self.memory.get_memory(affecting_nodes)
            last_t = self.memory.last_update[affecting_nodes]
            prev_memory[last_t == 0] = self.memory.default_memory
            time_diffs = timestamps - last_t
            encoded_time = self.time_encoder(time_diffs.unsqueeze(1)).squeeze(1)
            edge_feats = torch.cat([connector_subgraph.edges['plays'].data['feats'],
                                    connector_subgraph.edges['plays'].data['result'].unsqueeze(1),
                                    encoded_time], 1)

            # the two edge-related subgraphs (two mods/relations) are not aligned in terms of edges
            # that was previously the case - we might be able to remove this, when we're sure they are aligned
            # ======================================================================================
            # verify that edges are sorted on the first etype
            src_nid = connector_subgraph.edges(etype='plays')[0].detach().cpu().numpy()
            assert np.all(src_nid[:-1] < src_nid[1:])
            dst_nid = connector_subgraph.edges(etype='played_by')[-1]
            aligned_feats = edge_feats[dst_nid]

            nfeats, efeats = self.raw_message_processor(connector_subgraph,
                                                        node_inputs={
                                                            'player': prev_memory,
                                                            'match': torch.empty(connector_subgraph.num_nodes('match'),
                                                                                 device=self.device)
                                                        },
                                                        edge_feats={
                                                            'plays': edge_feats,
                                                            'played_by': aligned_feats
                                                        })

            all_affected_nodes.append(affecting_nodes)
            all_propagated_messages.append(nfeats['player'])
            propagated_messages_ts.append(timestamps)

        all_affected_nodes = torch.cat(all_affected_nodes, 0)
        all_propagated_messages = torch.cat(all_propagated_messages, 0)
        propagated_messages_ts = torch.cat(propagated_messages_ts, 0)

        assert len(torch.unique(all_affected_nodes)) == len(all_affected_nodes)

        return all_affected_nodes, all_propagated_messages, propagated_messages_ts

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self,
                         pair_graph,
                         etype,
                         player_node_ids,
                         match_node_ids,
                         edge_times):
        edge_features = torch.cat(
            [pair_graph.edges[etype].data['feats'], pair_graph.edges[etype].data['result'].unsqueeze(dim=1)],
            dim=1)

        cur_memory = self.memory.get_memory(player_node_ids)
        time_delta = edge_times - self.memory.last_update[player_node_ids]
        time_delta_encoding = self.time_encoder(time_delta.unsqueeze(dim=1)).view(len(
            player_node_ids), -1)

        raw_messages = torch.cat([cur_memory, edge_features, time_delta_encoding],
                                 dim=1)
        raw_messages_dict = defaultdict(list)

        for i in range(len(player_node_ids)):
            # this hack is meant to make sure we only store a single new raw message at a time per node
            # TODO: is it still required?
            raw_messages_dict[player_node_ids[i]] = [{'message_feats': raw_messages[i],
                                                      'ts': edge_times[i],
                                                      'match_nid': match_node_ids[i]
                                                      }]

            # raw_messages_dict[player_node_ids[i]].append({'message_feats': raw_messages[i],
            #                                               'ts': edge_times[i],
            #                                               'match_nid': match_node_ids[i]
            #                                               })

        return raw_messages_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
