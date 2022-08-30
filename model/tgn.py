import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import dgl

from utils.misc import flip_edges, validate_subgraph
from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from modules.custom import EdgeUpdatingHeteroGraphConv, GNNLayer, CompleteCycleGraphConv


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
                 use_source_embedding_in_message=False,
                 debug_mode=False):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.debug = debug_mode
        self.logger = logging.getLogger(__name__)

        self.membership_etypes = {'forward': ('player', 'member', 'team'),
                                  'backwards': ('team', 'contains', 'player')}
        self.opponents_etypes = {'forward': ('team', 'competes', 'match'),
                                 'backwards': ('match', 'played_by', 'team')}

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
                             device=device,
                             debug_mode=debug_mode)
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                         device=device)

        self.raw_message_processor = CompleteCycleGraphConv(player_dim=self.memory_dimension,
                                                            team_dim=0,
                                                            match_dim=0,

                                                            member_dim=(
                                                                               self.n_edge_features + 1) + self.time_encoder.dimension,
                                                            competes_dim=1,
                                                            activation=F.elu,
                                                            hidden_dim=message_dimension,
                                                            out_dim=message_dimension,
                                                            etype_mirrors={'member': 'contains',
                                                                           'competes': 'played_by'}).to(self.device)

        self.message_function = get_message_function(module_type=message_function,
                                                     raw_message_dimension=message_dimension,
                                                     message_dimension=message_dimension).to(self.device)
        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device).to(self.device)

        self.team_aggregator = dgl.nn.GATConv(in_feats=(memory_dimension, 1), out_feats=memory_dimension,
                                              num_heads=1).to(self.device)
        self.dummy_team_input = torch.zeros(1, 1, device=self.device)
        self.team_matchup = torch.nn.MultiheadAttention(embed_dim=memory_dimension, num_heads=1,
                                                        batch_first=True, add_bias_kv=True).to(self.device)

        self.score_projector = torch.nn.Linear(self.memory_dimension, 1, device=self.device)

        # self.pre_attn_lin_q = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)
        # self.pre_attn_lin_k = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)
        # self.pre_attn_lin_v = torch.nn.Linear(memory_dimension, memory_dimension).to(self.device)

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

        # make sure only two edge types of edges is available
        assert sum([len(pair_graph.edata[dgl.EID][etype]) > 0 for etype in pair_graph.canonical_etypes]) == 2

        node_ids_p = pair_graph.nodes['player'].data[dgl.NID]  # only the player related nodes
        node_ids_m = pair_graph.nodes['match'].data[dgl.NID]  # only the match related nodes

        m_edge_src, m_edge_dst = pair_graph.edges(etype='member')
        c_edge_src, c_edge_dst = pair_graph.edges(etype='competes')
        batch_player_node_ids = (node_ids_p[m_edge_src]).cpu().numpy()
        batch_match_node_ids = (node_ids_m[c_edge_dst]).cpu().numpy()
        edge_times = pair_graph.edges['member'].data['timestamp']

        assert torch.equal(pair_graph.nodes['player'].data[dgl.NID], node_ids_p[m_edge_src])
        assert len(np.unique(batch_match_node_ids)) == 1 and len(torch.unique(edge_times)) == 1

        # Update memory for all nodes with messages stored in previous batches
        memory, last_update, updated_nodes = self.update_and_get_memory(batch_player_node_ids,
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

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(updated_nodes)
        # add to raw message storage the information about the nodes involved in the current batch
        self.memory.store_uniform_raw_info(np.unique(batch_player_node_ids),
                                           {
                                               'ts': edge_times[0],
                                               'match_nid': batch_match_node_ids[0]
                                           })

        return out_node_embeddings, batch_player_node_ids

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
        self.memory_updater.update_and_store_memory(unique_nodes, unique_messages,
                                                    timestamps=unique_timestamps)

    def update_and_get_memory(self,
                              out_node_ids,
                              complete_graph):
        # Figure out which nodes (out of the complete set of nodes of interest) require memory update computation
        # if there are such - we acquire (using the raw memory storage) the corresponding matches ids (together with the
        # relevant timestamps)
        nodes_to_update, message_timestamps, connector_node_id = \
            self.message_aggregator.aggregate(
                out_node_ids,
                self.memory.raw_messages_storage)

        # in case there are any nodes which require memory computation
        if len(nodes_to_update) > 0:
            # in such case we need to compute the message according to which memory will be updated
            affected_nodes, propagated_messages, messages_ts = self.acquire_all_messages(nodes_to_update,
                                                                                         message_timestamps,
                                                                                         connector_node_id,
                                                                                         complete_graph)
            # a subsequent processing (not graph dependent) of the acquired message representation
            processed_messages = self.message_function.compute_message(propagated_messages)

            # Update the memory with the aggregated messages
            self.memory_updater.update_and_store_memory(affected_nodes, processed_messages,
                                                        timestamps=messages_ts)
        else:
            processed_messages = None
            affected_nodes = torch.empty(0)
            messages_ts = []

        return self.memory.get_memory(out_node_ids), self.memory.last_update[out_node_ids], affected_nodes.numpy()

    def acquire_all_messages(self, induced_nodes, message_timestamps, connector_node_id, complete_graph):
        # sort the incoming messages according to timestamps
        message_timestamps, indices = torch.sort(message_timestamps)
        indices = indices.cpu().numpy()
        induced_nodes = induced_nodes[indices]
        connector_node_id = connector_node_id[indices]

        # we need to iterate for each connector_node_id once
        connector_to_players = defaultdict(list)
        connector_to_ts = defaultdict(list)
        # map each match to the corresponding relevant node ids and relevant timestamps
        for node_id, message_ts, conn_id in zip(induced_nodes, message_timestamps, connector_node_id):
            connector_to_players[conn_id].append(node_id)
            connector_to_ts[conn_id].append(message_ts.unsqueeze(0))
        # make sure each connector is associated with a single unique ts
        for conn_id, conn_ts in connector_to_ts.items():
            connector_to_ts[conn_id] = torch.unique(torch.cat(conn_ts, 0))
        assert all([len(conn_ts) == 1 for _, conn_ts in connector_to_ts.items()])

        # for each of the listed matches, aggregate all the affected nodes together with the computed messages and the
        # corresponding timestamps
        all_affected_nodes = list()
        all_propagated_messages = list()
        propagated_messages_ts = list()

        for conn_id in connector_to_players:
            affected_nodes, nodes_outputs, timestamps = self.get_messages_from_subgraph(complete_graph, conn_id,
                                                                                        connector_to_players,
                                                                                        connector_to_ts)

            all_affected_nodes.append(affected_nodes)
            all_propagated_messages.append(nodes_outputs['player'])
            propagated_messages_ts.append(timestamps)

        all_affected_nodes = torch.cat(all_affected_nodes, 0)
        all_propagated_messages = torch.cat(all_propagated_messages, 0)
        propagated_messages_ts = torch.cat(propagated_messages_ts, 0)

        assert len(torch.unique(all_affected_nodes)) == len(all_affected_nodes), \
            "Every affected node should appear only once"

        return all_affected_nodes, all_propagated_messages, propagated_messages_ts

    def get_messages_from_subgraph(self, complete_graph, conn_id, connector_to_players, connector_to_ts):
        connector_subgraph = self.get_match_subgraph(complete_graph, conn_id)
        connector_subgraph = self.validate_subgraph(connector_subgraph)
        connector_subgraph = self.flip_subgraph_edges(connector_subgraph)

        # which player nodes are relevant?
        affecting_nodes = connector_subgraph.nodes['player'].data[dgl.NID]
        # make sure all to nodes we're interested in (w.r.t to the specific match) are vertices in the sliced subgraph
        assert all([node_id in affecting_nodes for node_id in connector_to_players[conn_id]])

        self.verify_edge_order_alignment(connector_subgraph)
        self.verify_raw_info_storage_alignment(affecting_nodes, conn_id, connector_to_ts[conn_id])
        if self.debug:
            # verify that edges are sorted on the first etype
            src_nid = connector_subgraph.edges(etype='member')[0].detach().cpu().numpy()
            assert np.all(src_nid[:-1] < src_nid[1:])

        connector_subgraph = connector_subgraph.to(self.device)
        prev_memory = self.memory.get_memory(affecting_nodes)

        # encode time difference
        cur_ts = connector_to_ts[conn_id].repeat(connector_subgraph.num_nodes('player'))  # same ts shared for all nodes
        last_t = self.memory.last_update[affecting_nodes]  # last ts recorded (and used for memory update) - each node
        time_diffs = connector_to_ts[conn_id] - last_t
        encoded_time = self.time_encoder(time_diffs.unsqueeze(1)).squeeze(1)

        # set the feature vector corresponding to each player-member-match edge and team-competes-match edge
        member_efeats = torch.cat([connector_subgraph.edges['member'].data['feats'],
                                   connector_subgraph.edges['member'].data['result'].unsqueeze(1),
                                   encoded_time], 1)
        competes_efeats = connector_subgraph.edges['competes'].data['result'].unsqueeze(1)

        nodes_outputs, edges_outputs = self.raw_message_processor(connector_subgraph,
                                                                  node_inputs={
                                                                      'player': prev_memory,
                                                                      'team': torch.empty(
                                                                          connector_subgraph.num_nodes('team'), 0,
                                                                          device=self.device),
                                                                      'match': torch.empty(
                                                                          connector_subgraph.num_nodes('match'),
                                                                          0, device=self.device),
                                                                  },
                                                                  edge_feats={
                                                                      'member': member_efeats,
                                                                      'contains': member_efeats,
                                                                      'competes': competes_efeats,
                                                                      'played_by': competes_efeats,
                                                                  })

        return affecting_nodes, nodes_outputs, cur_ts

    def verify_edge_order_alignment(self, connector_subgraph):
        # verify that edges of opposite types are ordered identically
        if self.debug:
            assert torch.equal(connector_subgraph.edges(etype='member')[0],
                               connector_subgraph.edges(etype='contains')[-1])
            assert torch.equal(connector_subgraph.edges(etype='contains')[0],
                               connector_subgraph.edges(etype='member')[-1])
            assert torch.equal(connector_subgraph.edges(etype='competes')[0],
                               connector_subgraph.edges(etype='played_by')[-1])

    def verify_raw_info_storage_alignment(self, subgraph_nodes, conn_id, match_ts):
        if not self.debug:
            return

        # fetch raw info from raw memory storage for all the subgraph player nodes
        nodes_to_update, timestamps, connectors = \
            self.message_aggregator.aggregate(
                subgraph_nodes.numpy(),
                self.memory.raw_messages_storage)
        assert all(
            connectors == conn_id), 'All the cached raw messages should be associated with the relevant match id'
        assert all(
            timestamps == match_ts), 'All the cached raw messages should be associated with the relevant timestamp'
        assert all([nid in subgraph_nodes for nid in nodes_to_update]) and len(subgraph_nodes) == len(
            nodes_to_update), 'They should point to the same nodes, otherwise this update should have been performed earlier'

    def validate_subgraph(self, connector_subgraph):
        if not self.debug:
            return connector_subgraph
        else:
            return validate_subgraph(connector_subgraph)

    @staticmethod
    def get_match_subgraph(complete_graph, match_id):
        # get the subgraph (until player level) associated with the relevant match
        connector_subgraph, _ = dgl.khop_in_subgraph(complete_graph,
                                                     nodes={'match': match_id},
                                                     k=2,
                                                     store_ids=True)

        return connector_subgraph

    def flip_subgraph_edges(self, connector_subgraph):
        # add opposite edges to get the complete undirected graph
        connector_subgraph = flip_edges(connector_subgraph,
                                        orig_etype=self.membership_etypes['forward'],
                                        rev_etype=self.membership_etypes['backwards'])
        connector_subgraph = flip_edges(connector_subgraph,
                                        orig_etype=self.opponents_etypes['forward'],
                                        rev_etype=self.opponents_etypes['backwards'])
        return connector_subgraph

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
                         player_node_ids,
                         match_node_id,
                         edge_time):

        raw_messages_dict = defaultdict(list)

        for i in range(len(player_node_ids)):
            # this hack is meant to make sure we only store a single new raw message at a time per node
            # TODO: is it still required?
            raw_messages_dict[player_node_ids[i]] = [{
                'ts': edge_time,
                'match_nid': match_node_id
            }]

        return raw_messages_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
