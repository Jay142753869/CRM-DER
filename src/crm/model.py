# -*- coding: utf-8 -*-
"""CRM-DER 正式模型：真实主干 + 统一事件 bank + 因果状态槽 + 双端读出。

计算图（每次预测一次前向）：
    真实事实帧 -> DGL -> RGCN(最后一层捕获) -> H_raw[N,L,D] -> MA -> E[N,D]
                                    │
                                    ├── group_messages -> FrameGroups（物理重复保留，count 保留）
                                    │
                     ┌──────────────┴───────────────┐
                     ▼                              ▼
              FactAdapter.bank_from_groups    CausalRelationState.forward_frames
              （原始 t<T 校验、单位换算）      （逐帧 read-before-write，输入依赖门控）
                     │                              │
                     │                       adapter.bank_from_slots(source=STATE)
                     └──────────────┬───────────────┘
                                    ▼
                     readout.prepare()  ← 唯一一次 K/V 构建（不 detach）
                                    │
              ┌─────────────────────┴──────────────────────┐
              ▼                                            ▼
   read_subject -> subject_rows -> ConvTransE query   read_candidates -> delta[B,N]
                                    │
                     raw = ConvTransE(query) @ E.T  (+ C2 显式恢复对照，默认关闭)
                                    │
                          logits = raw + gamma_cand * delta
"""
from dataclasses import dataclass, field

import torch
from torch import nn

from ..backbone.config import BaselineConfig
from ..backbone.model import BaselineModel
from ..backbone.initialization import isolated_module
from ..c2.history_selector import QueryHistoryScoreSelector
from .core import (CausalRelationState, DualEndReadout, EventBank, FactAdapter,
                   StateFrame, SOURCE_STATE, GATE_MODES)
from .frame_groups import FrameGroups, group_messages


@dataclass
class CRMOutput:
    raw_scores: torch.Tensor
    query_x: torch.Tensor
    history: object
    groups: list
    requests: torch.Tensor
    stats: dict = field(default_factory=dict)


class CRMModel(nn.Module):
    """保留真实 RGCN/MA/CNN/ConvTransE，用 CRM 替换 C1/C2/C3/C4 的独立投影。"""

    def __init__(self, meta, args):
        super().__init__()
        self.meta, self.args = meta, args
        d = args.n_hidden
        profile = 'baseline_fixed'
        config = BaselineConfig(meta.N, meta.R, hidden_dim=d,
                                history_len=args.train_history_len, num_layers=args.n_layers,
                                use_static=args.add_static_graph, num_words=meta.static.num_words,
                                num_static_relations=meta.static.relation_axis,
                                layer_norm=args.layer_norm, dropout=args.dropout,
                                add_query=args.add_query, temporal=args.pre_type,
                                task='entity', profile=profile)
        self.backbone = BaselineModel(config)
        if args.common_state:
            state = torch.load(args.common_state, map_location='cpu')
            self.backbone.load_state_dict(state, strict=True)
        # A-nostate 消融臂：state_mode='off' 时不构造状态模块，也不注入状态槽。
        self.state = isolated_module(lambda: CausalRelationState(
            d, args.state_rank_k, args.state_rank_v, gate_mode=args.state_gate), args.seed + 201) \
            if args.state_mode == 'on' else None
        self.readout = isolated_module(lambda: DualEndReadout(
            d, args.crm_hidden, gamma_subj_init=args.gamma_subj_init,
            gamma_cand_init=0.0), args.seed + 202)
        # 默认关闭独立 C2；提供显式恢复对照臂。
        self.c2 = isolated_module(lambda: QueryHistoryScoreSelector(
            d, args.train_history_len, hidden_dim=args.c2_hidden,
            use_time_bias=args.c2_time_bias, gamma_init=0.0), args.seed + 203) \
            if args.enable_c2 else None
        self.adapter = FactAdapter(meta.N, meta.R, d, unit=args.time_unit,
                                   device=None, dtype=torch.float32)
        self.static_decoder = self.backbone.static_decoder if args.add_static_graph else None
        self.last_stats = {}

    # ------------------------------------------------------------------ #
    @property
    def device(self):
        return self.backbone.dynamic_emb.device

    def _state_slots(self, adapter, groups, relations, prediction_time):
        """状态槽来自严格更早帧；第 0 帧读出恒为零（结构性常数），不进入 bank。

        state_mode='off'（A-nostate 消融臂）返回 None：既不计算也不注入任何状态槽。
        """
        if self.state is None:
            return None
        if len(groups) < 2:
            return None
        frames = [StateFrame(g.mean, g.relation, g.owner, g.timestamp) for g in groups]
        out = self.state(frames, self.meta.N, relations, prediction_time,
                         unit=self.args.time_unit)
        values = out['values'][1:]
        if not values:
            return None
        mean = torch.cat(values, 0)
        rel = torch.cat(out['rels'][1:], 0)
        owner = torch.cat(out['owners'][1:], 0)
        stamp = torch.tensor(out['timestamps'][1:], device=mean.device, dtype=mean.dtype)
        stamp = torch.cat([stamp[i].expand(values[i].shape[0]) for i in range(len(values))])
        bank = adapter.bank_from_slots(mean, rel, owner, stamp,
                                       source=torch.full_like(owner, SOURCE_STATE))
        bank.verify_before(prediction_time)
        return bank

    def forward(self, requests, graphs, times, prediction_time, static_graph=None):
        if len(times) != len(graphs) or any(t >= prediction_time for t in times):
            raise ValueError('history must be aligned and strictly before prediction time')
        # 新机制明确请求 capture；不依赖任何旧 C1/C4 flag。
        history = self.backbone.encode_history(requests, graphs, static_graph, times, capture=True)
        relations = history.R
        groups = [group_messages(cap.owner, cap.relation, cap.messages, timestamp=cap.time)
                  for cap in history.messages]
        adapter = self.adapter.to(history.E.device, history.E.dtype)
        recent = adapter.bank_from_groups(groups, prediction_time)
        state_bank = self._state_slots(adapter, groups, relations, prediction_time)
        prepared = self.readout.prepare(recent, relations, prediction_time, extra=state_bank)
        subject_rows = self.readout.read_subject(prepared, history.E, relations, history.requests)
        query = self.backbone.decode_queries(history, history.requests, subject_rows)
        raw = self.backbone.score_queries(query, history.E)
        c2_delta = None
        if self.c2 is not None:
            slots = torch.tensor([cap.position for cap in history.messages],
                                 dtype=torch.long, device=history.E.device)
            c2_delta = self.c2(query, history.H_raw, history.requests, history.padding,
                               slot_ids=slots, task='entity', bank_kind='H_raw')
            raw = raw + c2_delta
        delta = self.readout.read_candidates(prepared, query, self.meta.N,
                                             chunk=self.args.crm_query_chunk,
                                             recompute=self.args.crm_checkpoint_chunks)
        raw = raw + delta
        self.last_stats = dict(slots=int(prepared.k.shape[0]),
                               recent_slots=int(recent.mean.shape[0]),
                               state_slots=int(0 if state_bank is None else len(state_bank)),
                               frames=len(groups),
                               prepare_calls=self.readout.prepare_calls,
                               capture=len(history.messages),
                               c2_enabled=self.c2 is not None,
                               c2_delta_norm=None if c2_delta is None else float(c2_delta.detach().abs().mean()),
                               delta_norm=float(delta.detach().abs().mean()),
                               gamma_subj=float(self.readout.gamma_subj.detach()),
                               gamma_cand=float(self.readout.gamma_cand.detach()),
                               state_gate_mode=None if self.state is None else self.state.gate_mode,
                               state_mode=self.args.state_mode,
                               state_gate_mean=None if state_bank is None else None)
        return CRMOutput(raw, query, history, groups, history.requests, dict(self.last_stats))

    # ------------------------------------------------------------------ #
    def static_loss(self, history, triples):
        if triples is None or len(triples) == 0:
            basis = history.static_E if history.static_E is not None else history.E
            return basis.new_zeros(())
        if history.static_E is None:
            raise ValueError('static loss requires static mode')
        triples = torch.as_tensor(triples, device=history.static_E.device, dtype=torch.long)
        # Full static direction query/BN/dropout exactly once; only score/loss
        # blocks are recomputed, so chunking does not alter query context.
        query = self.backbone.static_decoder.query(history.static_E[triples[:, 0]],
                                                   history.static_R[triples[:, 1]])
        from torch.utils.checkpoint import checkpoint
        def block_loss(q, candidates, labels):
            return self.backbone.task_loss(q @ candidates.T, labels)
        losses = []
        for start in range(0, len(triples), self.args.static_score_chunk):
            stop = min(start + self.args.static_score_chunk, len(triples))
            inputs = (query[start:stop], history.static_E, triples[start:stop, 2])
            local = checkpoint(block_loss, *inputs, use_reentrant=False) if torch.is_grad_enabled() else block_loss(*inputs)
            losses.append(local * ((stop - start) / len(triples)))
        return sum(losses)

    def task_loss(self, scores, targets):
        return self.backbone.task_loss(scores, targets)

    def constructors(self):
        return dict(
            arch_version='crm-der-v2',
            state=None if self.state is None else dict(
                dim=self.args.n_hidden, rank_k=self.args.state_rank_k,
                rank_v=self.args.state_rank_v, gate_mode=self.state.gate_mode,
                params=sum(p.numel() for p in self.state.parameters())),
            state_mode=self.args.state_mode,
            readout=dict(dim=self.args.n_hidden, hidden=self.args.crm_hidden,
                         gamma_subj_init=self.args.gamma_subj_init, gamma_cand_init=0.0,
                         checkpoint_chunks=self.args.crm_checkpoint_chunks,
                         params=sum(p.numel() for p in self.readout.parameters())),
            c2=None if self.c2 is None else dict(h_dim=self.args.n_hidden,
                                                 hidden=self.args.c2_hidden,
                                                 time_bias=self.args.c2_time_bias,
                                                 params=sum(p.numel() for p in self.c2.parameters())),
            sharing=dict(keys_values='single_prepare_shared_by_both_ends',
                         kv_projections_shared=True,
                         query_output_projections_shared=False),
            bank=dict(time_unit=self.args.time_unit,
                      source_tags=dict(recent=0, state=1, long_live=2),
                      long_slots='off' if not self.args.use_long_slots else 'enabled_separate_source_adapter'),
            capture='explicit_true_inside_crm_forward',
            backbone=dict(hidden_dim=self.args.n_hidden, history_len=self.args.train_history_len,
                          num_layers=self.args.n_layers, temporal=self.args.pre_type,
                          add_query=self.args.add_query, use_static=self.args.add_static_graph))
