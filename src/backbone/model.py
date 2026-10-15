from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config import BaselineConfig
from .decoder import ConvTransE
from .encoder import Transformer_encoder
from .layers import RGCNCell, MessageCapture
from .query import Qencoder


@dataclass
class QueryContext:
    subjects: torch.Tensor
    spans: torch.Tensor
    indices: torch.Tensor


@dataclass
class HistoryOutput:
    E: torch.Tensor
    R: torch.Tensor
    H_raw: torch.Tensor
    padding: torch.Tensor
    messages: List[MessageCapture]
    static_E: Optional[torch.Tensor]
    static_R: Optional[torch.Tensor]
    query_token: torch.Tensor
    query_mask: torch.Tensor
    requests: torch.Tensor


class BaselineModel(nn.Module):
    """Explicit-config author backbone with live hooks for C1/C2/C3.

    Initialization order includes author inactive layers to preserve default-256
    random draws before MA/Qencoder. No imports parse arguments or select CUDA.
    """
    def __init__(self, config: BaselineConfig, query_encoder_factory=None,
                 query_encoder_seed=1043):
        super().__init__()
        self.config = config
        c = config
        d = c.hidden_dim
        self.emb_rel = nn.Parameter(torch.Tensor(c.num_relations * 2, d))
        nn.init.xavier_normal_(self.emb_rel)
        self.dynamic_emb = nn.Parameter(torch.Tensor(c.num_entities, d))
        nn.init.normal_(self.dynamic_emb)
        if c.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(c.num_words, d))
            nn.init.kaiming_normal_(self.words_emb, nonlinearity='relu')
            self.static_remb = nn.Parameter(torch.Tensor(c.num_static_relations * 2, d))
            nn.init.kaiming_normal_(self.static_remb, nonlinearity='relu')
            # Cell alias was author's dynamic relation even when static forward uses static_remb.
            self.static_rgcn_layer = RGCNCell(c, self.emb_rel, layer_relation=self.static_remb)
            self.static_decoder = ConvTransE(d, c.q_type, static=True)
        self.rgcn = RGCNCell(c, self.emb_rel)
        self.his_rgcn_layer = RGCNCell(c, self.emb_rel, historical=True)
        self.time_gate_weight = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.zeros(d))
        self.pre_gate_weight = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.entity_cell = nn.GRUCell(d, d)
        self.decoder_ob = ConvTransE(d, c.q_type)
        self.MA = Transformer_encoder(d_model=d, dim_feedforward=512, nhead=4,
                                      num_encoder_layers=2, dropout=0.2,
                                      use_query=c.add_query, profile=c.profile)
        condition_axis = c.num_entities if c.task == 'relation' else c.num_relations * 2
        self.query_encoder = nn.Sequential(nn.Linear(condition_axis, d), nn.ReLU())
        self.Qencoder = Qencoder(input_dim=d, hidden_dim=d, out_dim=d)
        if query_encoder_factory is not None:
            from .initialization import isolated_module
            self.Qencoder = isolated_module(lambda: query_encoder_factory(d), query_encoder_seed)

    @property
    def device(self):
        return self.dynamic_emb.device

    def _requests(self, requests):
        requests = torch.as_tensor(requests, device=self.device)
        if requests.ndim != 2 or requests.shape[1] != 2:
            raise ValueError('known requests must be [B,2]: (s,r) for entity or (s,o) for relation; no gold column')
        if requests.dtype not in (torch.int64, torch.int32):
            raise ValueError('known request IDs must be integer')
        requests = requests.long()
        axis = self.config.num_entities if self.config.task == 'relation' else self.config.num_relations*2
        if ((requests[:, 0] < 0) | (requests[:, 0] >= self.config.num_entities)).any():
            raise ValueError('subject outside entity axis')
        if ((requests[:, 1] < 0) | (requests[:, 1] >= axis)).any():
            raise ValueError('known condition outside task axis')
        return requests

    def query_context(self, requests):
        requests = self._requests(requests)
        subjects = requests[:, 0].unique(sorted=True)
        indices, spans, start = [], [], 0
        for subject in subjects:
            values = requests[requests[:, 0] == subject, 1]
            indices.append(values)
            spans.append([start, start + len(values)])
            start += len(values)
        return QueryContext(subjects,
                            torch.tensor(spans, dtype=torch.long, device=self.device).reshape(-1, 2),
                            torch.cat(indices) if indices else requests.new_empty(0))

    def _query_token(self, requests, initial):
        c = self.config
        context = self.query_context(requests)
        active = torch.zeros(c.num_entities, dtype=torch.bool, device=self.device)
        active[context.subjects] = True
        if c.use_onehot or not c.add_query:
            axis = c.num_entities if c.task == 'relation' else c.num_relations*2
            counts = initial.new_zeros(c.num_entities, axis)
            for subject, (start, end) in zip(context.subjects, context.spans):
                idx = context.indices[start:end]
                counts[subject].index_add_(0, idx, counts.new_ones(len(idx)))
                if len(idx):
                    counts[subject] /= len(idx)
            return self.query_encoder(counts), active
        token = initial.new_zeros(c.num_entities, c.hidden_dim)
        if not len(context.subjects):
            return token, active
        max_items = int((context.spans[:, 1] - context.spans[:, 0]).max())
        # Zero padding to complete-direction maximum is an intentional CNN property.
        items = initial.new_zeros(c.num_entities, max_items, c.hidden_dim)
        conditions = initial if c.task == 'relation' else self.emb_rel
        for subject, (start, end) in zip(context.subjects, context.spans):
            idx = context.indices[start:end]
            items[subject, :len(idx)] = conditions[idx]
        if getattr(self.Qencoder, 'requires_lengths', False):
            lengths = context.spans[:, 1] - context.spans[:, 0]
            token[context.subjects] = self.Qencoder(items[context.subjects],
                                                   self.dynamic_emb[context.subjects],
                                                   lengths=lengths)
        else:
            # Keep the author's two-argument path unchanged for default encoders.
            token[context.subjects] = self.Qencoder(items[context.subjects],
                                                   self.dynamic_emb[context.subjects])
        return token, active

    def _check_graph(self, graph, expected_nodes, relation_axis):
        graph = graph.to(self.device)
        if graph.num_nodes() != expected_nodes:
            raise ValueError('graph must contain the complete configured node axis')
        if 'id' not in graph.ndata or 'norm' not in graph.ndata or 'type' not in graph.edata:
            raise ValueError('graph needs ndata id/norm and edata type')
        if not torch.equal(graph.ndata['id'].reshape(-1), torch.arange(expected_nodes, device=self.device)):
            raise ValueError('graph must use ordered global node IDs')
        if (graph.edata['type'] < 0).any() or (graph.edata['type'] >= relation_axis).any():
            raise ValueError('edge relation outside configured directed axis')
        return graph

    def encode_history(self, requests, graphs, static_graph=None, times=None, capture=False):
        """graphs are strictly past, chronological, at most history_len; caller enforces T.

        H_raw is pre-Transformer graph output, left-padded [N,L,D]. In GRU mode
        it contains raw graph outputs before recurrence. Padding means missing
        frame, never node inactivity. Captures are final dynamic layer messages.
        """
        c = self.config
        requests = self._requests(requests)
        if len(graphs) > c.history_len:
            raise ValueError('pass only the selected last history_len snapshots')
        if not graphs and c.profile == 'baseline_original':
            raise ValueError('author forward is undefined for empty history; use baseline_fixed fallback')
        if times is not None and len(times) != len(graphs):
            raise ValueError('times must align one-to-one with graphs')
        if times is not None and any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('history times must be strictly increasing')
        static_e, static_r = None, None
        if c.use_static:
            if static_graph is None:
                raise ValueError('static mode needs a real background graph')
            graph = self._check_graph(static_graph, c.num_entities+c.num_words,
                                      c.num_static_relations*2)
            static_e = self.static_rgcn_layer(graph, torch.cat([self.dynamic_emb, self.words_emb]),
                                               self.static_remb)
            static_e = F.normalize(static_e) if c.layer_norm else static_e
            static_r = F.normalize(self.static_remb) if c.layer_norm else self.static_remb
            initial = static_e[:c.num_entities]
        else:
            initial = F.normalize(self.dynamic_emb) if c.layer_norm else self.dynamic_emb
        raw, messages = [], []
        state = initial
        for position, graph in enumerate(graphs):
            graph = self._check_graph(graph, c.num_entities, c.num_relations*2)
            current = self.rgcn(graph, initial if c.temporal == 'TF' else state, self.emb_rel,
                                capture=messages.append if capture else None,
                                time=times[position] if times is not None else None,
                                position=c.history_len-len(graphs)+position)
            current = F.normalize(current) if c.layer_norm else current
            raw.append(current)
            if c.temporal == 'GRU':
                state = self.entity_cell(current, state)
                state = F.normalize(state) if c.layer_norm else state
        if c.temporal == 'TF' or c.use_onehot or not c.add_query:
            # Author computes CNN after dynamic graph dropout, before MA dropout.
            token, query_mask = self._query_token(requests, initial)
        else:
            # Author GRU never calls CNN/MA; avoid consuming their dropout RNG.
            token = initial.new_zeros(c.num_entities, c.hidden_dim)
            query_mask = torch.zeros(c.num_entities, dtype=torch.bool, device=self.device)
            query_mask[requests[:, 0]] = True
        left = c.history_len-len(raw)
        zeros = initial.new_zeros(left, c.num_entities, c.hidden_dim)
        sequence = torch.stack(raw, dim=0) if raw else zeros
        if raw and left:
            sequence = torch.cat([zeros, sequence], dim=0)
        H_raw = sequence.permute(1, 0, 2)
        padding = torch.zeros(c.num_entities, c.history_len, dtype=torch.bool, device=self.device)
        padding[:, :left] = True
        if c.temporal == 'TF':
            E = self.MA(H_raw, token.unsqueeze(1), padding, mask=None)[:, -1]
            E = F.normalize(E) if c.layer_norm else E
        else:
            E = state
        relation = F.normalize(self.emb_rel) if c.layer_norm else self.emb_rel
        return HistoryOutput(E, relation, H_raw, padding, messages, static_e, static_r,
                             token, query_mask, requests.clone())

    def decode_queries(self, history, requests, subject_rows=None):
        requests = self._requests(requests)
        if not torch.equal(requests, history.requests):
            raise ValueError('decode must use the complete direction context encoded by encode_history')
        subject = history.E[requests[:, 0]] if subject_rows is None else subject_rows
        if subject.shape != (len(requests), self.config.hidden_dim):
            raise ValueError('C1 subject_rows must be [B,D] in request row order')
        condition = (history.E if self.config.task == 'relation' else history.R)[requests[:, 1]]
        return self.decoder_ob.query(subject, condition)

    @staticmethod
    def score_queries(query_x, candidates):
        return query_x @ candidates.T

    def score_requests(self, history, requests, subject_rows=None):
        query = self.decode_queries(history, requests, subject_rows)
        candidates = history.R if self.config.task == 'relation' else history.E
        return self.score_queries(query, candidates), query

    @staticmethod
    def task_loss(scores, targets):
        # Match QDRL-main-new's stable NLL formulation across candidate scales.
        return F.nll_loss(F.log_softmax(scores, dim=1), targets)

    @staticmethod
    def predict_logprob(scores):
        return torch.log(F.softmax(scores, dim=1))

    def static_loss(self, history, triples=None):
        if triples is None or len(triples) == 0:
            return history.E.new_zeros(())
        if history.static_E is None:
            raise ValueError('static loss requires static mode')
        triples = torch.as_tensor(triples, device=self.device, dtype=torch.long)
        scores = self.static_decoder(history.static_E, history.static_R, triples)
        return self.task_loss(scores, triples[:, 2])

    @staticmethod
    def original_total_loss(task_loss, static_loss):
        return 0.9*task_loss + 0.1*static_loss if static_loss.item() > 3 else task_loss

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        from .initialization import validate_alias_state
        validate_alias_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
