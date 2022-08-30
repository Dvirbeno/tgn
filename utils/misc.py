from typing import Tuple
import torch
import dgl


def flip_edges(orig_graph: dgl.heterograph, orig_etype: Tuple[str, str, str], rev_etype: Tuple[str, str, str]):
    rel_sg = orig_graph.edge_type_subgraph([orig_etype])
    reverse_rel = dgl.reverse(rel_sg, copy_edata=True)

    reversed_edges = reverse_rel.all_edges()
    orig_graph.add_edges(*reversed_edges, data=reverse_rel.edata,
                         etype=rev_etype)
    return orig_graph


def validate_subgraph(subgraph):
    for type_of_edge in subgraph.canonical_etypes:
        if 'member' not in type_of_edge:
            continue

        if subgraph.num_nodes('player') != subgraph.num_edges(type_of_edge):
            # means that a player has two edges for the same match
            seen_pid = []
            delete_eid = []
            u, v, e = subgraph.edges('all', etype=type_of_edge)

            is_player_dst = type_of_edge[-1] == 'player'

            for idx_u, idx_v, idx_edge in zip(u, v, e):
                idx_player = idx_v if is_player_dst else idx_u

                if idx_player in seen_pid:
                    delete_eid.append(idx_edge)
                else:
                    seen_pid.append(idx_player)
            delete_eid = torch.stack(delete_eid)

            if len(delete_eid) >= 1:
                subgraph.remove_edges(delete_eid, type_of_edge)

    return subgraph
