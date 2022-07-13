import argparse
import traceback
import time
import copy
import pickle
import os
import json

import numpy as np
import dgl
import torch
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from allrank.models import losses

from model.tgn import TGN
from utils.dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                               SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                               TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)
from utils.data_processing import compute_time_statistics
from evaluation import metrics

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)


def train(model, dataloader, optimizer, args, criterion, epoch_idx, output_dir):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()

    # will be used to aggregate batch results and metrics
    records = []
    cumulative_loss = 0

    output_file = os.path.join(output_dir, f"{epoch_idx}.csv")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if os.path.exists(output_file):  os.remove(output_file)

    for input_nodes, pair_g, blocks in dataloader:
        optimizer.zero_grad()

        pair_g = pair_g.to(model.device)
        blocks = [b.to(model.device) for b in blocks]

        # protection - in case there's only a single edge in the subgraph
        if pair_g.num_edges('played_by') <= 1:
            continue

        loss, metrics_all, known_nodes, known_loss, metrics_known = process_batch(input_nodes,
                                                                                  pair_g,
                                                                                  blocks,
                                                                                  dataloader,
                                                                                  model,
                                                                                  criterion)

        # accumulate the loss according to the configuration
        if not args.opt_known:
            cumulative_loss += loss
        elif known_loss is not None:
            cumulative_loss += known_loss

        # loss = criterion(dummy_pred, pair_g.edata['result'][('match', 'played_by', 'player')])
        retain_graph = True if batch_cnt == 0 and not args.fast_mode else False

        d = {'Batch': batch_cnt,
             'Time': time.time() - last_t,
             'OverallLoss': loss.item(),
             'numKnown': known_nodes.item(),
             }
        d.update(metrics_all)
        # print_str = ' , '.join(f"{k} : {v:0.3f}" for k, v in d.items())
        # print(print_str)

        if known_nodes > 1:
            dict_addition = {'knownLoss': known_loss.item()}
            dict_addition.update({f"known_{k}": v for k, v in metrics_known.items()})
            # print(' , '.join(f"{k} : {v:0.3f}" for k, v in dict_addition.items()))
            d.update(dict_addition)
        else:
            d.update({'knownLoss': np.nan})

        records.append(d)

        last_t = time.time()
        batch_cnt += 1

        if batch_cnt % args.backprop_every == 0:
            cumulative_loss /= args.backprop_every
            cumulative_loss.backward()
            optimizer.step()

            cumulative_loss = 0
            optimizer.zero_grad()
            model.memory.detach_memory()

        if batch_cnt % 100 == 0:
            print(f"Batch: {batch_cnt}")
            pd.DataFrame(records).to_csv(output_file,
                                         mode='a', header=not os.path.exists(output_file),
                                         index=False)
            records = []


def process_batch(input_nodes,
                  pair_g,
                  blocks,
                  dataloader,
                  model,
                  criterion):
    # protection against cases in which a single player has more than a single match to the same edge
    if pair_g.num_nodes('player') != pair_g.num_edges('played_by'):
        # means that a player has two edges for the same match
        seen_pid = []
        delete_eid = []
        u, v, e = pair_g.edges('all', etype='played_by')
        for idx_match, idx_player, idx_edge in zip(u, v, e):
            if idx_player in seen_pid:
                delete_eid.append(idx_edge)
            else:
                seen_pid.append(idx_player)
        delete_eid = torch.stack(delete_eid)

        if len(delete_eid) >= 1:
            pair_g.remove_edges(delete_eid, 'played_by')

    # get embedding for the source nodes in the subgraph of interest
    source_node_embeddings = model.compute_temporal_embeddings(input_nodes, pair_g, blocks,
                                                               complete_graph=dataloader.collator.g_sampling)

    # how many nodes in the subgraph were "seen" before and are known to the model
    known_nodes = (model.memory.last_update[pair_g.nodes['player'].data[dgl.NID]] > 0).sum()
    # which of the nodes are the ones that are known
    _, dstnodes = pair_g.edges(etype='played_by')
    p_nid = pair_g.nodes['player'].data[dgl.NID][dstnodes]
    known_edges = model.memory.last_update[p_nid] > 0

    # project the acquired embedding for getting the output score
    dummy_pred = torch.squeeze(model.dummy_lin(source_node_embeddings))
    # pred_pos, pred_neg = model.embed(input_nodes, pair_g, blocks)

    predicted_scores = dummy_pred.unsqueeze(0)
    # acquire the true target values for the nodes involved in the subgraph of interest
    true_relevance = pair_g.edata['result'][('match', 'played_by', 'player')].unsqueeze(0)
    # loss = +losses.listMLE(predicted_scores, true_relevance)
    # compute loss and metrics
    loss = criterion(predicted_scores, true_relevance)
    metrics_all = metrics.calc_metrics(preds=predicted_scores.squeeze(0).detach().cpu().numpy(),
                                       targets=true_relevance.squeeze(0).detach().cpu().numpy())

    # we also compute the loss and the metrics for the known nodes separately
    if known_nodes > 1:
        targets = true_relevance[:, known_edges]

        pred_known = torch.squeeze(model.dummy_lin(source_node_embeddings[known_edges])).unsqueeze(0)

        metrics_known = metrics.calc_metrics(preds=pred_known.squeeze(0).detach().cpu().numpy(),
                                             targets=targets.squeeze(0).detach().cpu().numpy())

        # known_loss = +losses.listMLE(predicted_scores[:, known_edges], true_relevance[:, known_edges])
        known_loss = criterion(pred_known, targets)
        # known_loss = +losses.marginRankNet(pred_known, targets)

        # cumulative_loss += known_loss
    else:
        known_loss, metrics_known = None, None

    return loss, metrics_all, known_nodes, known_loss, metrics_known


def test_val(model, dataloader, criterion, args, epoch_idx, output_dir):
    model.eval()

    output_file = os.path.join(output_dir, f"{epoch_idx}.csv")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if os.path.exists(output_file):  os.remove(output_file)

    # will be used to aggregate batch results and metrics
    records = []
    batch_cnt = 0
    last_t = time.time()

    with torch.no_grad():

        for input_nodes, pair_g, blocks in dataloader:
            pair_g = pair_g.to(model.device)

            # protection - in case there's only a single edge in the subgraph
            if pair_g.num_edges('played_by') <= 1:
                continue

            loss, metrics_all, known_nodes, known_loss, metrics_known = process_batch(input_nodes,
                                                                                      pair_g,
                                                                                      blocks,
                                                                                      dataloader,
                                                                                      model,
                                                                                      criterion)

            d = {'Batch': batch_cnt,
                 'Time': time.time() - last_t,
                 'OverallLoss': loss.item(),
                 'numKnown': known_nodes.item(),
                 }
            d.update(metrics_all)
            # print_str = ' , '.join(f"{k} : {v:0.3f}" for k, v in d.items())
            # print(print_str)

            if known_nodes > 1:
                dict_addition = {'knownLoss': known_loss.item()}
                dict_addition.update({f"known_{k}": v for k, v in metrics_known.items()})
                # print(' , '.join(f"{k} : {v:0.3f}" for k, v in dict_addition.items()))
                d.update(dict_addition)
            else:
                d.update({'knownLoss': np.nan})

            records.append(d)

            last_t = time.time()
            batch_cnt += 1

            if batch_cnt % 100 == 0:
                print(f"Validation Batch: {batch_cnt}")
                pd.DataFrame(records).to_csv(output_file,
                                             mode='a', header=not os.path.exists(output_file),
                                             index=False)
                records = []


def decompose_batches(group_indicator):
    # assume ordered
    cur_group = None
    cur_batch = None
    batches_list = []

    for idx, grp in enumerate(group_indicator):
        if grp != cur_group:

            if cur_batch is not None:
                batches_list.append(cur_batch)
            cur_batch = []
            cur_group = grp

        cur_batch.append(idx)

    batches_list.append(cur_batch)
    return batches_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='name of experiment', required=True)
    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs for training on entire dataset')
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode", action="store_true", default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--simple_mode", action="store_true", default=False,
                        help="Simple Mode directly delete the temporal edges from the original static graph")
    parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        help="dataset selection wikipedia/reddit")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--opt_known', action="store_true", default=False,
                        help="Optimizes the model according to known nodes only")

    args = parser.parse_args()
    print(args)

    output_dir = os.path.join('/mnt/DS_SHARED/users/dvirb/experiments/research/skill/graphs/pubg', args.name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'cli_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    assert not (
            args.fast_mode and args.simple_mode), "you can only choose one sampling mode"

    # if args.k_hop != 1:
    #     assert args.simple_mode, "this k-hop parameter only support simple mode"

    # Set device
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    data_path = os.path.normpath('/mnt/DS_SHARED/users/dvirb/data/research/graphs/games/pubg')

    print('Loading metadata file (if exists)')
    t_start = time.time()
    meta_file = os.path.join(data_path, 'complete_meta.pickle')
    if os.path.exists(meta_file):
        with open(meta_file, 'rb') as handle:
            loaded_meta = pickle.load(handle)
    else:
        loaded_meta = None
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    print('Loading graph object from disk')
    t_start = time.time()
    gs, _ = dgl.load_graphs(os.path.join(data_path, 'complete_ffa.bin'))
    data = gs[0]
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # reverse the edges
    print('Reversing edges to form the opposite relation (including edge data)')
    t_start = time.time()
    rel = data.edge_type_subgraph([('player', 'plays', 'match')])
    reverse_rel = dgl.reverse(rel, copy_edata=True)
    reversed_edges = reverse_rel.all_edges()
    data.add_edges(*reversed_edges, data=reverse_rel.edata, etype=('match', 'played_by', 'player'))
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # Pre-process data, mask new node in test set from original graph
    print('Counting nodes and verifying order')
    t_start = time.time()
    num_nodes = data.num_nodes()
    num_edges = data.num_edges(etype='plays')

    num_lasting_nodes = data.num_nodes('player')
    num_dispensable_nodes = data.num_nodes('match')

    # make sure the dataset is sorted
    assert torch.all(torch.diff(data.edges['plays'].data['timestamp']) >= 0)
    src_id, dst_id, edge_id = data.edges('all', etype='plays')
    assert torch.all(torch.diff(dst_id) >= 0)
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # set split according to preset fraction
    # =======================================
    print('Determining subset splits')
    t_start = time.time()

    # train
    train_div = int(TRAIN_SPLIT * num_edges)
    train_last_ts = data.edges['plays'].data['timestamp'][train_div]

    # validation
    trainval_div = int(VALID_SPLIT * num_edges)
    # get split time
    div_time = data.edges['plays'].data['timestamp'][trainval_div]
    # update index according to time considerations
    trainval_div = (1 + (data.edges['plays'].data['timestamp'] <= div_time).nonzero()[
        -1]).item()  # get last index where time is prior (or equal) to div_time, and assign the next index (+1)

    # Select new node from test set and remove them from entire graph
    test_split_ts = data.edges['plays'].data['timestamp'][trainval_div]
    test_nodes = {
        'player': data.edges(etype='plays')[0][trainval_div:].unique().numpy(),
        'match': data.edges(etype='plays')[1][trainval_div:].unique().numpy()
    }

    test_new_nodes = {
        'player': np.random.choice(test_nodes['player'], int(0.1 * len(test_nodes['player'])), replace=False)}
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    print(f"Creating a subgraph excluding the new-unseen nodes")
    t_start = time.time()
    in_subg = dgl.in_subgraph(data, test_new_nodes)
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = {
        etype: in_subg.edges[etype].data[dgl.EID][in_subg.edges[etype].data['timestamp'] < test_split_ts] for etype in
        data.canonical_etypes}  # for each edge type get the edge ids that precede `test_split_ts`
    new_node_out_eid_delete = {
        etype: out_subg.edges[etype].data[dgl.EID][out_subg.edges[etype].data['timestamp'] < test_split_ts] for etype in
        data.canonical_etypes}  # for each edge type get the edge ids that precede `test_split_ts`

    new_node_eid_delete = {etype: torch.cat(
        [new_node_in_eid_delete[etype], new_node_out_eid_delete[etype]]).unique() for etype in
                           data.canonical_etypes}

    graph_new_node = copy.deepcopy(data)
    # relative order preseved
    for etype, eid in new_node_eid_delete.items():
        graph_new_node.remove_edges(eid, etype)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = {
        etype: in_subg.edges[etype].data[dgl.EID] for etype in data.canonical_etypes
    }
    out_eid_delete = {
        etype: out_subg.edges[etype].data[dgl.EID] for etype in data.canonical_etypes
    }
    eid_delete = {
        etype: torch.cat([in_eid_delete[etype], out_eid_delete[etype]]).unique() for etype in data.canonical_etypes
    }

    graph_no_new_node = copy.deepcopy(data)
    for etype, eid in eid_delete.items():
        graph_no_new_node.remove_edges(eid, etype)
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # graph_no_new_node and graph_new_node should have same set of nid

    # Sampler Initialization
    print(f"Initializing Sampler")
    t_start = time.time()
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(data, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors, hops=args.k_hop)
        edge_collator = TemporalEdgeCollator
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # Set Train, validation, test and new node test id
    print(f"Creating boolean arrays indicating the subset-association of edges")
    t_start = time.time()
    train_bool = {etype: graph_no_new_node.edges[etype].data['timestamp'] <= train_last_ts for etype in
                  data.canonical_etypes}
    valid_bool = {etype: torch.logical_and(graph_no_new_node.edges[etype].data['timestamp'] > train_last_ts,
                                           graph_no_new_node.edges[etype].data['timestamp'] <= div_time) for etype in
                  data.canonical_etypes}
    test_bool = {etype: graph_no_new_node.edges[etype].data['timestamp'] > div_time for etype in data.canonical_etypes}
    test_new_node_bool = {etype: graph_new_node.edges[etype].data['timestamp'] > div_time for etype in
                          data.canonical_etypes}
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    print(f"Creating subset-specific seeds and setting batch samplers")
    t_start = time.time()
    train_seed, valid_seed, test_seed, test_new_node_seed = dict(), dict(), dict(), dict()
    for etype in data.canonical_etypes:  # get eids
        # instead of iterating over etypes, only a single etype is required
        is_match_source = etype.index('match') == 0

        srcs, dsts, eids = graph_no_new_node.edges('all', etype=etype)
        match_indicator = srcs if is_match_source else dsts

        train_seed[etype] = eids[train_bool[etype]]
        valid_seed[etype] = eids[valid_bool[etype]]
        test_seed[etype] = eids[test_bool[etype]]

        if loaded_meta is None:
            train_batch_sampler = decompose_batches(match_indicator[train_bool[etype]])
            valid_batch_sampler = decompose_batches(match_indicator[valid_bool[etype]])
            test_batch_sampler = decompose_batches(match_indicator[test_bool[etype]])
        else:
            train_batch_sampler = loaded_meta['train_batch_sampler']
            valid_batch_sampler = loaded_meta['valid_batch_sampler']
            test_batch_sampler = loaded_meta['test_batch_sampler']

        srcs, dsts, eids = graph_new_node.edges('all', etype=etype)
        match_indicator = srcs if is_match_source else dsts
        test_new_node_seed[etype] = eids[test_new_node_bool[etype]]

        if loaded_meta is None:
            test_new_node_batch_sampler = decompose_batches(match_indicator[test_new_node_bool[etype]])
        else:
            test_new_node_batch_sampler = loaded_meta['test_new_node_batch_sampler']

        # currently - because there are just two types of edges, and they are also reversed with the same data,
        # we don't need really need to consider the reverse edges (for the mirror etype) as seeds.
        # Hence - break # TODO: consider what to do with it when teams are involved
        break
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # Compute time statistics
    print(f"Computing (or loading from pre-computed meta) the time-statistics")
    t_start = time.time()
    if loaded_meta is None:
        mean_time_shift, std_time_shift = compute_time_statistics(data, edge_type=('player', 'plays', 'match'))
    else:
        mean_time_shift, std_time_shift = loaded_meta['mean_time_shift'], loaded_meta['std_time_shift']
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    print(f"Assinging node ids to graphs")
    t_start = time.time()
    g_sampling = None if args.fast_mode else graph_no_new_node
    new_node_g_sampling = None if args.fast_mode else graph_new_node
    if not args.fast_mode:
        for ntype in data.ntypes:
            new_node_g_sampling.nodes[ntype].data[dgl.NID] = new_node_g_sampling.nodes(ntype)
            g_sampling.nodes[ntype].data[dgl.NID] = new_node_g_sampling.nodes(ntype)
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    reverse_etypes = {
        ('player', 'plays', 'match'): ('match', 'played_by', 'player'),
        ('match', 'played_by', 'player'): ('player', 'plays', 'match')
    }

    print(f"Initializing data loaders")
    t_start = time.time()
    train_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                              eids=train_seed,
                                              graph_sampler=sampler,
                                              batch_sampler=train_batch_sampler,
                                              # batch_size=args.batch_size,  # * 2500,  # TODO: change back
                                              negative_sampler=None,
                                              # shuffle=False,
                                              # drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              # g_sampling=g_sampling,
                                              exclude='reverse_types',
                                              reverse_etypes=reverse_etypes)

    valid_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                              eids=valid_seed,
                                              graph_sampler=sampler,
                                              batch_sampler=valid_batch_sampler,
                                              # batch_size=args.batch_size,
                                              negative_sampler=None,
                                              # shuffle=False,
                                              # drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              # g_sampling=g_sampling,
                                              exclude='reverse_types',
                                              reverse_etypes=reverse_etypes)

    test_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                             eids=test_seed,
                                             graph_sampler=sampler,
                                             batch_sampler=test_batch_sampler,
                                             # batch_size=args.batch_size,
                                             negative_sampler=None,
                                             # shuffle=False,
                                             # drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             # g_sampling=g_sampling,
                                             exclude='reverse_types',
                                             reverse_etypes=reverse_etypes)

    test_new_node_dataloader = TemporalEdgeDataLoader(g=graph_new_node,
                                                      eids=test_new_node_seed,
                                                      graph_sampler=new_node_sampler if args.fast_mode else sampler,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=None,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      # g_sampling=new_node_g_sampling,
                                                      exclude='reverse_types',
                                                      reverse_etypes=reverse_etypes)
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    etypes = data.canonical_etypes
    assert data.edata['feats'][etypes[0]].shape[1] == data.edata['feats'][etypes[1]].shape[1]
    edge_dim = data.edata['feats'][etypes[0]].shape[1]

    if loaded_meta is None:
        with open(meta_file, 'wb') as handle:
            pickle.dump({
                'mean_time_shift': mean_time_shift,
                'std_time_shift': std_time_shift,
                'train_batch_sampler': train_batch_sampler,
                'valid_batch_sampler': valid_batch_sampler,
                'test_batch_sampler': test_batch_sampler,
                'test_new_node_batch_sampler': test_new_node_batch_sampler
            },
                handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Initialize Model
    print(f"Initializing model and optimizer")
    t_start = time.time()
    model = TGN(n_edge_features=edge_dim,
                n_node_features=0,
                n_nodes=num_lasting_nodes,
                device=device,
                n_layers=args.n_layer,
                n_heads=args.num_heads,
                dropout=args.drop_out,
                embedding_dimension=args.embedding_dim,
                message_dimension=args.message_dim,
                memory_dimension=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_module_type=args.embedding_module,
                message_function=args.message_function,
                aggregator_type=args.aggregator,
                memory_updater_type=args.memory_updater,
                mean_time_shift=mean_time_shift, std_time_shift=std_time_shift,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message,
                )

    criterion = losses.rankNet
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print(f"Endured {(time.time() - t_start):.3f} Seconds")

    # Implement Logging mechanism
    f = open("logging.txt", 'w')
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            print('=' * 50)
            print(f"Starting epoch no. {i}")
            print('=' * 50)

            print(f"Initializing memory")
            t_start = time.time()
            model.memory.__init_memory__()
            print(f"Endured {(time.time() - t_start):.3f} Seconds")

            print('=' * 50)
            print(f"Starting training for epoch no. {i}")
            print('=' * 50)
            train(model, train_dataloader, optimizer, args, criterion, epoch_idx=i, output_dir=output_dir)

            print('=' * 50)
            print(f"Starting validation for epoch no. {i}")
            print('=' * 50)
            test_val(model, valid_dataloader, criterion, args, epoch_idx=i,
                     output_dir=os.path.join(output_dir, 'validation'))

            continue  # TODO: right now running only for validation

            memory_checkpoint = model.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            test_ap, test_auc = test_val(
                model, test_dataloader, sampler, criterion, args)
            model.restore_memory(memory_checkpoint)
            if args.fast_mode:
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap, nn_test_auc = test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                i, train_loss, val_ap, val_auc))
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                i, nn_test_ap, nn_test_auc))

            f.writelines(log_content)
            model.reset_memory()
            if i < args.epochs - 1 and args.fast_mode:
                sampler.reset()
            print(log_content[0], log_content[1], log_content[2])
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
