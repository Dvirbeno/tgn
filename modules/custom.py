import math
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, source_dim, dest_dim, edge_dim, activation, hidden_dim, ndim_out, edim_out,
                 ignore_srcdata=False, ignore_dstdata=False):
        super(GNNLayer, self).__init__()
        self.W_msg = nn.Linear(source_dim + edge_dim, hidden_dim)
        self.W_apply = nn.Linear(dest_dim + ndim_out, ndim_out)
        self.W_edge = nn.Linear(source_dim + edge_dim + hidden_dim, edim_out)
        self.activation = activation
        self.ignore_srcdata = ignore_srcdata
        self.ignore_dstdata = ignore_dstdata

        if ignore_srcdata:
            self.message_func = self.message_func_ignoring_src
            self.f_edge = self.f_edge_ignoring_src
        else:
            self.message_func = self.default_message_func
            self.f_edge = self.default_f_edge

    def default_message_func(self, edges):
        return {'m': self.activation(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 1)))}

    def message_func_ignoring_src(self, edges):
        return {'m': self.activation(self.W_msg(edges.data['h']))}

    def default_f_edge(self, edges):
        return {'eh': self.W_edge(torch.cat([edges.src['h'], edges.data['h'], edges.dst['h_neigh']], 1))}

    def f_edge_ignoring_src(self, edges):
        return {'eh': self.W_edge(torch.cat([edges.data['h'], edges.dst['h_neigh']], 1))}

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.srcdata['h'], g.dstdata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g.apply_edges(self.f_edge)

            apply_inputs = g.dstdata['h_neigh'] if self.ignore_dstdata else torch.cat(
                [g.dstdata['h'], g.dstdata['h_neigh']], 1)
            g.dstdata['h'] = self.W_apply(apply_inputs)

            return g.dstdata['h'], g.edata['eh']


class EdgeUpdatingHeteroGraphConv(dglnn.HeteroGraphConv):
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        edge_outputs = {ety: [] for _, ety, _ in g.canonical_etypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata, edge_data = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
                edge_outputs[etype] = edge_data
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts, edge_outputs


class TwoLayerHeteroGraphConv(nn.Module):
    def __init__(self, player_dim, match_dim, edge_dim, activation, hidden_dim, out_dim):
        super(TwoLayerHeteroGraphConv, self).__init__()
        self.conv1 = EdgeUpdatingHeteroGraphConv({
            'plays': GNNLayer(source_dim=player_dim,
                              dest_dim=match_dim,
                              edge_dim=edge_dim,
                              activation=activation,
                              hidden_dim=hidden_dim,
                              ndim_out=hidden_dim,
                              edim_out=hidden_dim,
                              ignore_dstdata=match_dim == 0,
                              ignore_srcdata=player_dim == 0),
            'played_by': GNNLayer(source_dim=match_dim,
                                  dest_dim=player_dim,
                                  edge_dim=edge_dim,
                                  activation=activation,
                                  hidden_dim=hidden_dim,
                                  ndim_out=hidden_dim,
                                  edim_out=hidden_dim,
                                  ignore_srcdata=match_dim == 0,
                                  ignore_dstdata=player_dim == 0)},
            aggregate='sum')

        self.conv2 = EdgeUpdatingHeteroGraphConv({
            'plays': GNNLayer(source_dim=hidden_dim,
                              dest_dim=hidden_dim,
                              edge_dim=hidden_dim,
                              activation=activation,
                              hidden_dim=hidden_dim,
                              ndim_out=out_dim,
                              edim_out=out_dim),
            'played_by': GNNLayer(source_dim=hidden_dim,
                                  dest_dim=hidden_dim,
                                  edge_dim=hidden_dim,
                                  activation=activation,
                                  hidden_dim=hidden_dim,
                                  ndim_out=out_dim,
                                  edim_out=out_dim)},
            aggregate='sum')

    def forward(self, g, node_inputs, edge_feats):
        nfeats, efeats = self.conv1(g, inputs=node_inputs,
                                    mod_kwargs={k: {'efeats': v} for k, v in edge_feats.items()})
        nouts, eouts = self.conv2(g, inputs=nfeats, mod_kwargs={k: {'efeats': v} for k, v in efeats.items()})

        return nouts, eouts


class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))

        h_t = torch.mul(1 - z_t, h) + torch.mul(z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view(h_t.size(0), -1)
        return h_t
