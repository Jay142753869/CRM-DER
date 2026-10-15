from dataclasses import dataclass
from typing import Any

import dgl.function as fn
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MessageCapture:
    owner: torch.Tensor
    relation: torch.Tensor
    messages: torch.Tensor
    time: Any
    position: int
    layer: int
    partner: Any = None


class UnionRGCNLayer(nn.Module):
    """Author's additive relation message, linear projection, indegree mean, RReLU."""
    def __init__(self, dim, rel_emb, self_loop=True, dropout=0.2, skip_connect=False,
                 historical=False):
        super().__init__()
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.historical = historical
        if historical:
            # Retain original inactive parameters and RNG consumption for initialization parity.
            self.weight_t = nn.Parameter(torch.randn(1, dim))
            self.bias_t = nn.Parameter(torch.randn(1, dim))
            self.weight_neighbor = nn.Parameter(torch.Tensor(dim, dim))
            nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        else:
            self.neighbor_linear = nn.Linear(dim, dim)
        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(dim, dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(dim, dim))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        if skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(dim, dim))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout) if dropout else None
        # Stable aliases from construction, unlike author's forward-time registration.
        self.rel_emb = rel_emb

    def forward(self, graph, embedding, relation, capture=None, metadata=None):
        with graph.local_scope():
            graph.ndata['h'] = embedding
            if self.self_loop:
                active = graph.in_degrees() > 0
                loop = embedding @ self.evolve_loop_weight
                loop[active] = (embedding @ self.loop_weight)[active]

            def message(edges):
                rel = relation.index_select(0, edges.data['type']).view(-1, embedding.shape[-1])
                node = edges.src['h'].view(-1, embedding.shape[-1])
                if self.historical:
                    fre = torch.softmax(edges.data['fre'].float().unsqueeze(1), dim=0)
                    msg = (node + rel * fre) @ self.weight_neighbor
                else:
                    msg = self.neighbor_linear(node + rel)
                if capture is not None and edges.data['type'].numel() > 0:
                    # Empty snapshots have no dynamic messages; do not emit a 0-row capture.
                    capture(MessageCapture(edges.dst['id'].reshape(-1), edges.data['type'],
                                           msg, partner=edges.src['id'].reshape(-1) if getattr(self,'capture_partners',False) else None,
                                           **metadata))
                return {'msg': msg}

            graph.update_all(message, fn.sum(msg='msg', out='h'),
                             lambda nodes: {'h': nodes.data['h'] * nodes.data['norm']})
            result = graph.ndata['h']
            if self.self_loop:
                result = result + loop
            # Author's RGCNCell always passes [] as prev_h: skip branch remains inactive.
            result = F.rrelu(result, training=False)
            return self.dropout(result) if self.dropout is not None else result


class RGCNCell(nn.Module):
    def __init__(self, cfg, relation, layer_relation=None, historical=False):
        super().__init__()
        self.rel_emb = relation
        actual_relation = relation if layer_relation is None else layer_relation
        self.layers = nn.ModuleList([
            UnionRGCNLayer(cfg.hidden_dim, actual_relation, cfg.self_loop, cfg.dropout,
                           cfg.skip_connect and idx > 0, historical)
            for idx in range(cfg.num_layers)
        ])

    def forward(self, graph, embedding, relation, capture=None, time=None, position=0):
        node_ids = graph.ndata['id'].reshape(-1)
        result = embedding[node_ids]
        for idx, layer in enumerate(self.layers):
            result = layer(graph, result, relation,
                           capture=capture if idx == len(self.layers)-1 else None,
                           metadata={'time': time, 'position': position, 'layer': idx})
        return result
