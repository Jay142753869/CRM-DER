# -*- coding: utf-8 -*-
"""CRM-DER CPU 原型 v2：修复总控评审确认的四处缺陷。

与 v1 (`crm_model.py`, 保留为 before_v2 副本) 的差异：

R02-M1 每候选逻辑 null
  v1 把唯一 null 槽挂在 owner=0，零 value 仍改变该候选的 softmax 分母，
  因此只有实体 0 得到 null。v2 为**每个候选**各挂一个逻辑 null：
      denom[b,c] = sum_{g:owner=c} exp(logit_g - m_c) + exp(null_logit_b - m_c)
      m_c = max(max_{g:owner=c} logit_g, null_logit_b)
  null value 恒零（不进分子），但**进分母**。无槽候选分母恰为 1、delta 恰为 0。
  该公式对实体 ID 任意置换等变（独立 oracle 见 tests/test_crm_v2.py）。

R02-M2 门控
  v1 只有 lambda = sigmoid(logit_decay) 全局标量，与输入无关。
  v2 提供两种显式模式：
      gate_mode='input'  : 每 owner 门控，输入 = 当前帧该 owner 的组均值、
                           关系背景均值、帧时间特征；同帧组置换不变。
      gate_mode='global' : 保留 v1 的单一标量，作为同信息简化控制。
  两种模式的参数增量在 param_count 中单列。

R02-M3 一次共享 K/V
  v1 的 read_subject / read_candidates 各自调用 _keys_values，bank 被构造两次。
  v2 提供 prepare() -> PreparedBank，一次构建 k/v（不 detach），两端只消费同一资源；
  用 prepare_calls 计数证明"真实 full forward 只构建一次"。

R02-M5 原始时间验证边界
  v1 的 EventBank 只有 age，age 本身无法证明槽严格早于 T。
  v2 的 EventBank 保留原始真实时间戳 timestamp，并提供 verify_before(T)；
  FactAdapter 负责单位换算、设备/dtype、有限值/形状/ID 校验。

本文件只证明接口、因果边界、梯度连通、分块等价与置换等变，
不证明任何 MRR 收益。合成数据，CPU 或任意显式 device。
"""
import math
from dataclasses import dataclass, field
from numbers import Real

import torch
import torch.nn as nn

# 槽来源标签：只用于记账/诊断与 source adapter 区分，绝不替代时间语义。
SOURCE_RECENT = 0   # 近 L 帧 RGCN 组消息
SOURCE_STATE = 1    # 因果状态读出槽（逐帧产生）
SOURCE_LONG_LIVE = 2  # 长期 live E×R 重编码槽（默认关闭）

GATE_MODES = ('input', 'global')


def _as_scalar_time(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('{} must be a finite real scalar (raw dataset time unit)'.format(name))
    return float(value)


def _ids(name, value, size=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(name + ' must be a one-dimensional int64 tensor')
    if value.numel():
        if size is not None and (value.min().item() < 0 or value.max().item() >= size):
            raise ValueError(name + ' ID outside permitted range')
        elif size is None and value.min().item() < 0:
            raise ValueError(name + ' ID must be nonnegative')


def _floating(name, value, dim=None, rows=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(name + ' must be a floating point tensor')
    if not torch.isfinite(value).all():
        raise ValueError(name + ' contains non-finite values')
    if dim is not None and (value.ndim != 2 or value.shape[1] != dim):
        raise ValueError('{} must be [rows,{}]'.format(name, dim))
    if rows is not None and value.shape[0] != rows:
        raise ValueError('{} row count mismatch'.format(name))


def age_features(age):
    """age: [G] 非负浮点（原始时间差 / unit）。返回 [G,2] 有界特征。"""
    if not isinstance(age, torch.Tensor) or age.ndim != 1 or not age.is_floating_point():
        raise ValueError('age must be a floating [G] tensor')
    if not torch.isfinite(age).all():
        raise ValueError('age contains non-finite values')
    a = age.clamp(min=0)
    return torch.stack([torch.log1p(a) / math.log1p(64.0), 1.0 / (1.0 + a)], dim=-1)


class EventBank:
    """一次预测（时刻 T）的统一事件记忆。

    每个槽严格满足**原始** timestamp < T；age 只是派生特征，不承担因果证明。

    fields:
      mean      [G, D]  槽内容（近期=组消息均值；长期=live 重编码；状态=状态读出）
      rel       [G]     有向关系 id（0..2R-1）
      owner     [G]     槽归属实体 id
      timestamp [G]     原始真实时间戳（数据集时间单位）
      source    [G]     来源标签（SOURCE_*），仅记账
      unit      float   时间单位（原始 ticks / unit），来自数据集元数据
    """

    def __init__(self, mean, rel, owner, timestamp, source=None, unit=1.0):
        if callable(timestamp):
            raise ValueError(
                'EventBank API 在 v2 已变更：第 4 个位置参数现在是**原始时间戳** Tensor[G]，'
                '不再是派生年龄。age 现在是方法 EventBank.age(prediction_time)。'
                '总控 null 探针的等价重跑脚本：audit_v2/probe_v2.py。')
        _floating('bank.mean', mean)
        _ids('bank.rel', rel)
        _ids('bank.owner', owner)
        _floating('bank.timestamp', timestamp)
        if not (mean.shape[0] == rel.numel() == owner.numel() == timestamp.numel()):
            raise ValueError('bank fields must share the group axis')
        if rel.numel() and rel.min().item() < 0:
            raise ValueError('bank directed relation must be nonnegative')
        if rel.device != mean.device or owner.device != mean.device or timestamp.device != mean.device:
            raise ValueError('bank fields must share one device')
        if timestamp.dtype != mean.dtype:
            raise ValueError('bank timestamps must share the slot dtype')
        self.unit = float(unit)
        if not math.isfinite(self.unit) or self.unit <= 0:
            raise ValueError('time unit must be positive and finite')
        self.mean, self.rel, self.owner, self.timestamp = mean, rel, owner, timestamp
        self.source = (torch.full_like(owner, SOURCE_RECENT) if source is None else source)
        _ids('bank.source', self.source)

    # ------------------------------------------------------------------ #
    @property
    def dim(self):
        return self.mean.shape[1]

    def __len__(self):
        return self.mean.shape[0]

    def age(self, prediction_time):
        """派生年龄 = (T - t) / unit；不改变因果边界。"""
        T = _as_scalar_time(prediction_time, 'prediction_time')
        return (T - self.timestamp) / self.unit

    def verify_before(self, prediction_time):
        """原始时间边界：所有槽必须严格 t < T。这是因果性的**唯一**证据。"""
        T = _as_scalar_time(prediction_time, 'prediction_time')
        if self.timestamp.numel() and not bool((self.timestamp < T).all().item()):
            worst = float(self.timestamp.max().item())
            raise ValueError(
                'EventBank violates strict causality: max raw timestamp {} >= prediction time {}'
                .format(worst, T))
        return True

    def with_extra(self, extra):
        """把状态槽/长期槽并入同一逻辑 bank（同一次 prepare 消费）。"""
        if extra is None or len(extra) == 0:
            return self
        if isinstance(extra, EventBank):
            if extra.dim != self.dim:
                raise ValueError('extra bank dim mismatch')
            if extra.mean.device != self.mean.device or extra.mean.dtype != self.mean.dtype:
                raise ValueError('extra bank device/dtype mismatch')
            if extra.unit != self.unit:
                raise ValueError('extra bank time unit mismatch')
            return EventBank(torch.cat([self.mean, extra.mean], 0),
                             torch.cat([self.rel, extra.rel], 0),
                             torch.cat([self.owner, extra.owner], 0),
                             torch.cat([self.timestamp, extra.timestamp], 0),
                             torch.cat([self.source, extra.source], 0), unit=self.unit)
        raise ValueError('extra must be an EventBank')


@dataclass
class StateFrame:
    """因果状态扫描的一帧。timestamp 是**原始**真实时间，必须严格 < T。"""
    mean: torch.Tensor
    rel: torch.Tensor
    owner: torch.Tensor
    timestamp: float

    def __post_init__(self):
        _floating('frame.mean', self.mean)
        _ids('frame.rel', self.rel)
        _ids('frame.owner', self.owner)
        if self.mean.shape[0] != self.rel.numel() or self.mean.shape[0] != self.owner.numel():
            raise ValueError('frame fields must share the group axis')
        if self.rel.device != self.mean.device or self.owner.device != self.mean.device:
            raise ValueError('frame fields must share one device')
        self.timestamp = _as_scalar_time(self.timestamp, 'frame.timestamp')


class CausalRelationState(nn.Module):
    """每 owner 的关系键控因果状态 S[rank_k, rank_v]，逐帧 read-before-write。

    帧 t 的处理：
      read_g  = (S_{owner_g} * W_k(R[rel_g])).sum(dim=k)   # 只用 t' < t 的 S
      out_g   = W_out(read_g)                              # 状态槽 value
      write   : S_o <- lambda_o * S_o + (1 - lambda_o) * mean_{g:owner=o}(k_g (x) v_g)

    gate_mode='input'  : lambda_o 由当前帧该 owner 的组均值、关系背景均值与帧时间特征
                         决定（同帧组置换不变）。
    gate_mode='global' : lambda = sigmoid(logit_decay)，单一标量，与输入无关（同信息简化控制）。
    """

    def __init__(self, dim, rank_k=8, rank_v=32, gate_mode='input'):
        super().__init__()
        if gate_mode not in GATE_MODES:
            raise ValueError('gate_mode must be one of ' + repr(GATE_MODES))
        self.dim, self.rank_k, self.rank_v = dim, rank_k, rank_v
        self.gate_mode = gate_mode
        self.W_k = nn.Linear(dim, rank_k, bias=False)    # 关系 -> 键
        self.W_v = nn.Linear(dim, rank_v, bias=False)    # 消息 -> 值
        self.W_out = nn.Linear(rank_v, dim, bias=False)  # 状态读出 -> 槽内容
        if gate_mode == 'input':
            # 输入 = [owner 组均值(D), owner 关系背景均值(D), 帧时间特征(2)]
            self.gate_linear = nn.Linear(2 * dim + 2, 1)
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.zeros_(self.gate_linear.bias)
            self.logit_decay = None
        else:
            self.gate_linear = None
            self.logit_decay = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.W_out.weight)  # 零初始化：起步无贡献

    # ------------------------------------------------------------------ #
    def gate_parameters(self):
        """门控分支的参数增量（与 global 模式对比时单列）。"""
        if self.gate_mode == 'input':
            return list(self.gate_linear.parameters())
        return [self.logit_decay]

    def _owner_gate(self, frame, rel_emb, age_feat):
        """返回 (per-group lambda，按规范序排列；per-owner gate 向量)。

        规范序 = 按 (owner, rel) 稳定排序；同帧组置换下该序列逐位不变。
        """
        Gt = frame.mean.shape[0]
        device, dtype = frame.mean.device, frame.mean.dtype
        if Gt == 0:
            return (frame.mean.new_zeros(0), torch.empty(0, device=device, dtype=dtype),
                    torch.empty(0, device=device, dtype=torch.long))
        # 规范序：按 (owner, rel) 稳定排序，保证同帧组置换下求和逐位一致。
        order = torch.argsort(frame.owner * (rel_emb.shape[0] + 1) + frame.rel, stable=True)
        owner = frame.owner[order]
        uniq, inverse, counts = torch.unique(owner, return_inverse=True, return_counts=True)
        agg = frame.mean.new_zeros((uniq.numel(), self.dim))
        agg.index_add_(0, inverse, frame.mean[order])
        agg = agg / counts.to(dtype).unsqueeze(1)
        relagg = frame.mean.new_zeros((uniq.numel(), self.dim))
        relagg.index_add_(0, inverse, rel_emb[frame.rel[order]].to(dtype))
        relagg = relagg / counts.to(dtype).unsqueeze(1)
        timefeat = age_feat.reshape(1, -1).expand(uniq.numel(), -1)
        gate = torch.sigmoid(self.gate_linear(torch.cat([agg, relagg, timefeat], dim=-1))).squeeze(-1)
        return gate[inverse], gate, order

    def forward(self, frames, n_entities, rel_emb, prediction_time, unit=1.0):
        """nn.Module 入口：走 __call__，使 forward hook / autocast 等机制对状态扫描生效。

        实现与 forward_frames 完全一致；调用方（CRMModel）必须用 `self.state(...)`，
        不允许绕过 __call__ 直接调 forward_frames，否则 hook 统计会失真。
        """
        return self.forward_frames(frames, n_entities, rel_emb, prediction_time, unit=unit)

    def forward_frames(self, frames, n_entities, rel_emb, prediction_time, unit=1.0):
        """frames: list[StateFrame] 按原始时间严格升序。

        返回 dict：
          values  : list of [Gt, D]  每帧各组读到的"读出前值"（状态槽 value）
          gates   : list of [Gt]     每帧每组的门控值（global 模式为广播标量）
          gate_means: list of float  每帧平均门控（诊断）
          rels/owners/timestamps/ages : 与 values 对齐的槽元数据
        """
        if not isinstance(rel_emb, torch.Tensor) or not rel_emb.is_floating_point():
            raise ValueError('rel_emb must be a floating [2R,D] tensor')
        if rel_emb.shape[1] != self.dim:
            raise ValueError('rel_emb dim mismatch')
        T = _as_scalar_time(prediction_time, 'prediction_time')
        unit = float(unit)
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError('unit must be positive and finite')
        previous = None
        for frame in frames:
            if frame.timestamp >= T:
                raise ValueError('state frames must be strictly before prediction time '
                                 '(raw timestamp boundary)')
            if previous is not None and frame.timestamp <= previous:
                raise ValueError('state frames must use strictly increasing raw timestamps')
            previous = frame.timestamp
            if frame.mean.shape[1] != self.dim:
                raise ValueError('state frame dim mismatch')
            if frame.mean.dtype != rel_emb.dtype or frame.mean.device != rel_emb.device:
                raise ValueError('state frames must share rel_emb dtype/device')
            _ids('state frame owner', frame.owner, n_entities)
            _ids('state frame directed relation', frame.rel, rel_emb.shape[0])
        S = torch.zeros(n_entities, self.rank_k, self.rank_v,
                        device=rel_emb.device, dtype=rel_emb.dtype)
        values, gates, gate_means = [], [], []
        rels, owners, timestamps, ages = [], [], [], []
        for frame in frames:
            k = self.W_k(rel_emb[frame.rel])   # [Gt, rk]
            v = self.W_v(frame.mean)           # [Gt, rv]
            read = (S[frame.owner] * k.unsqueeze(-1)).sum(dim=1)  # [Gt, rv]
            values.append(self.W_out(read))
            age = (T - frame.timestamp) / unit
            Gt = frame.mean.shape[0]
            if self.gate_mode == 'input':
                age_feat = age_features(torch.as_tensor(age, device=rel_emb.device,
                                                        dtype=rel_emb.dtype).reshape(1))
                lam_sorted, _, order = self._owner_gate(frame, rel_emb, age_feat)
                lam = frame.mean.new_zeros(Gt)
                lam[order] = lam_sorted            # 回到原组顺序，供调用方/诊断使用
            else:
                order = torch.argsort(frame.owner * (rel_emb.shape[0] + 1) + frame.rel, stable=True)
                lam = torch.sigmoid(self.logit_decay).reshape(1).expand(Gt).to(rel_emb.dtype)
                lam_sorted = lam[order]
            gates.append(lam)
            gate_means.append(float(lam.mean().detach()) if lam.numel() else float('nan'))
            # 写入：帧内聚合同 owner 多组（规范序，置换不变）
            owner = frame.owner[order]
            uniq, inverse, counts = torch.unique(owner, return_inverse=True, return_counts=True)
            upd = frame.mean.new_zeros((uniq.numel(), self.rank_k, self.rank_v))
            outer = k[order].unsqueeze(-1) * v[order].unsqueeze(1)   # [Gt, rk, rv]
            upd.index_add_(0, inverse, outer)
            upd = upd / counts.to(frame.mean.dtype).view(-1, 1, 1)
            lam_o = frame.mean.new_zeros(uniq.numel())
            lam_o.index_add_(0, inverse, lam_sorted)
            lam_o = lam_o / counts.to(frame.mean.dtype)
            S = S.clone()
            S[uniq] = lam_o.view(-1, 1, 1) * S[uniq] + (1.0 - lam_o).view(-1, 1, 1) * upd
            rels.append(frame.rel)
            owners.append(frame.owner)
            timestamps.append(frame.timestamp)
            ages.append(age)
        return dict(values=values, gates=gates, gate_means=gate_means,
                    rels=rels, owners=owners, timestamps=timestamps, ages=ages)


@dataclass
class PreparedBank:
    """一次 prepare 产出的共享 K/V 资源。两端只消费，不重建。"""
    k: torch.Tensor
    v: torch.Tensor
    owner: torch.Tensor
    null_key: torch.Tensor
    source: torch.Tensor
    scale: float
    stats: dict = field(default_factory=dict)


class DualEndReadout(nn.Module):
    """统一 bank 的双端条件读出。共享 key/value 投影，方向特定 query/output 投影。"""

    def __init__(self, dim, hidden=64, gamma_subj_init=0.01, gamma_cand_init=0.0):
        super().__init__()
        D, H = dim, hidden
        self.dim, self.hidden = D, H
        self.W_m = nn.Linear(D, H, bias=False)
        self.W_a = nn.Linear(D, H, bias=False)
        self.W_age = nn.Linear(2, H, bias=False)
        self.W_v = nn.Linear(D, H, bias=False)
        self.null_key = nn.Parameter(torch.randn(H) * 0.02)
        # 主体端
        self.q_subj = nn.Sequential(nn.Linear(2 * D, H), nn.ReLU(), nn.Linear(H, H))
        self.W_os = nn.Linear(H, D, bias=False)
        self.gamma_subj = nn.Parameter(torch.tensor(float(gamma_subj_init)))
        # 候选端
        self.W_qc = nn.Linear(D, H, bias=False)
        self.gamma_cand = nn.Parameter(torch.tensor(float(gamma_cand_init)))
        self.prepare_calls = 0
        self.last_stats = {}

    # ------------------------------------------------------------------ #
    def prepare(self, bank, rel_emb, prediction_time, extra=None):
        """构建一次 K/V。返回 PreparedBank；不 detach 任何输入。"""
        if not isinstance(bank, EventBank):
            raise ValueError('prepare expects an EventBank')
        if rel_emb.ndim != 2 or rel_emb.shape[1] != self.dim or not rel_emb.is_floating_point():
            raise ValueError('rel_emb must be floating [2R,D]')
        if rel_emb.device != self.W_m.weight.device or rel_emb.dtype != self.W_m.weight.dtype:
            raise ValueError('rel_emb dtype/device must match the module')
        if bank.dim != self.dim:
            raise ValueError('bank dim mismatch')
        if bank.mean.device != rel_emb.device or bank.mean.dtype != rel_emb.dtype:
            raise ValueError('bank device/dtype must match rel_emb')
        if bank.rel.numel() and bank.rel.max().item() >= rel_emb.shape[0]:
            raise ValueError('bank directed relation outside the [2R] table')
        _ids('bank.owner', bank.owner)
        bank.verify_before(prediction_time)
        merged = bank.with_extra(extra)
        merged.verify_before(prediction_time)
        _ids('merged bank relation', merged.rel, rel_emb.shape[0])
        self.prepare_calls += 1
        k = self.W_m(merged.mean) + self.W_a(rel_emb[merged.rel]) + self.W_age(age_features(merged.age(prediction_time)))
        v = self.W_v(merged.mean)
        self.last_stats = dict(prepare_calls=self.prepare_calls, slots=int(merged.mean.shape[0]),
                               sources=[int(x) for x in torch.unique(merged.source).tolist()])
        return PreparedBank(k=k, v=v, owner=merged.owner, null_key=self.null_key,
                            source=merged.source, scale=1.0 / math.sqrt(self.hidden),
                            stats=dict(self.last_stats))

    # ------------------------------------------------------------------ #
    def read_subject(self, prepared, E, R, requests, *, rel_emb=None, prediction_time=None):
        """requests: [B,2] (s, r)。返回 [B, D] 注入解码器的主体行。

        每个请求主体在自身分段内共享一个逻辑 null（value 恒零）；无槽主体 E'=E[s]。
        正式路径要求传入 prepare() 得到的 PreparedBank；为兼容旧调用点，
        也接受原始 EventBank，但必须显式给出 rel_emb 与 prediction_time。
        """
        prepared = self._coerce(prepared, rel_emb, prediction_time)
        if E.ndim != 2 or E.shape[1] != self.dim or R.ndim != 2 or R.shape[1] != self.dim:
            raise ValueError('E/R must be [N,D]/[2R,D]')
        _ids('request subject', requests[:, 0], E.shape[0])
        _ids('request relation', requests[:, 1], R.shape[0])
        q = self.q_subj(torch.cat([E[requests[:, 0]], R[requests[:, 1]]], dim=-1))
        s_idx = requests[:, 0]
        out = prepared.v.new_zeros((s_idx.numel(), prepared.v.shape[1]))
        for subject in s_idx.unique():
            rows = (s_idx == subject).nonzero(as_tuple=True)[0]
            slots = (prepared.owner == subject).nonzero(as_tuple=True)[0]
            kb = torch.cat([prepared.k[slots], prepared.null_key[None]], 0)
            vb = torch.cat([prepared.v[slots], prepared.v.new_zeros(1, prepared.v.shape[1])], 0)
            alpha = torch.softmax(kb @ q[rows].t() * prepared.scale, dim=0)
            out = out.index_copy(0, rows, alpha.t() @ vb)
        return E[s_idx] + torch.tanh(self.gamma_subj) * self.W_os(out)

    def _candidate_delta(self, prepared, q_dec, n_entities):
        """每候选逻辑 null 的分段 softmax（非分块参考实现）。"""
        if prepared.owner.numel() and prepared.owner.max().item() >= n_entities:
            raise ValueError('bank owner outside candidate axis')
        rows = q_dec.shape[0]
        device, dtype = q_dec.device, q_dec.dtype
        u = self.W_qc(q_dec)                                   # [B,H]
        if prepared.k.numel() == 0:
            return q_dec.new_zeros((rows, n_entities))
        logits = (prepared.k @ u.t()) * prepared.scale        # [G,B]
        null_logit = (u @ prepared.null_key) * prepared.scale  # [B]
        owner = prepared.owner.unsqueeze(1).expand(-1, rows)   # [G,B]
        seg_max = logits.new_full((n_entities, rows), float('-inf'))
        seg_max = seg_max.scatter_reduce(0, owner, logits, reduce='amax', include_self=True)
        cand_max = torch.maximum(seg_max, null_logit.unsqueeze(0))   # [N,B]
        exps = torch.exp(logits - cand_max.gather(0, owner))
        null_exps = torch.exp(null_logit.unsqueeze(0) - cand_max)     # [N,B]
        denom = exps.new_zeros((n_entities, rows)).index_add(0, prepared.owner, exps) + null_exps
        alpha = exps / denom.gather(0, owner).clamp(min=1e-12)
        contrib = alpha * (prepared.v @ u.t())                 # [G,B]
        delta = q_dec.new_zeros((n_entities, rows)).index_add(0, prepared.owner, contrib)
        return delta.t()

    def _coerce(self, prepared, rel_emb, prediction_time):
        """旧调用点兼容：允许直接传 EventBank，但必须显式给出 rel_emb 与 prediction_time。

        正式路径始终是 prepare() -> PreparedBank；本兼容路径会真实调用 prepare()，
        因此 prepare_calls 仍如实计数。
        """
        if isinstance(prepared, PreparedBank):
            return prepared
        if isinstance(prepared, EventBank):
            if rel_emb is None or prediction_time is None:
                raise ValueError(
                    'passing a raw EventBank requires explicit rel_emb and prediction_time; '
                    'the formal path is prepare(bank, rel_emb, prediction_time) -> PreparedBank')
            return self.prepare(prepared, rel_emb, prediction_time)
        raise ValueError('expected a PreparedBank from prepare(), or an EventBank with explicit '
                         'rel_emb/prediction_time')

    def read_candidates(self, prepared, q_dec, n_entities, chunk=32, *, recompute=False,
                        rel_emb=None, prediction_time=None):
        """q_dec: [B, D] 解码器查询向量。返回 delta [B, N]（分块精确等价）。"""
        prepared = self._coerce(prepared, rel_emb, prediction_time)
        if not isinstance(n_entities, int) or n_entities <= 0:
            raise ValueError('n_entities must be a positive int')
        if q_dec.ndim != 2 or q_dec.shape[1] != self.dim or not q_dec.is_floating_point():
            raise ValueError('q_dec must be floating [B,D]')
        if q_dec.dtype != self.W_qc.weight.dtype or q_dec.device != self.W_qc.weight.device:
            raise ValueError('q_dec dtype/device must match the module')
        chunk = max(1, int(chunk))
        delta = q_dec.new_zeros((q_dec.shape[0], n_entities))
        for start in range(0, q_dec.shape[0], chunk):
            if recompute and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                def candidate_block(q, keys, values, null_key):
                    local = PreparedBank(keys, values, prepared.owner, null_key,
                                         prepared.source, prepared.scale)
                    return self._candidate_delta(local, q, n_entities)
                local_delta = checkpoint(candidate_block, q_dec[start:start + chunk],
                                         prepared.k, prepared.v, prepared.null_key,
                                         use_reentrant=False)
            else:
                local_delta = self._candidate_delta(prepared, q_dec[start:start + chunk], n_entities)
            delta[start:start + chunk] = local_delta
        return self.gamma_cand * delta

    # 兼容旧接口：单次调用也走 prepare，保证 bank 只构建一次。
    def read_subject_from_bank(self, bank, rel_emb, E, R, requests, prediction_time, extra=None):
        return self.read_subject(self.prepare(bank, rel_emb, prediction_time, extra), E, R, requests)

    def read_candidates_from_bank(self, bank, rel_emb, q_dec, n_entities, prediction_time,
                                  chunk=32, extra=None):
        return self.read_candidates(self.prepare(bank, rel_emb, prediction_time, extra),
                                    q_dec, n_entities, chunk)


class FactAdapter:
    """原始事实 -> EventBank 的正式适配器。

    责任边界：
      * 严格 t < T 用**原始**时间戳判定，绝不依赖 age；
      * 单位换算由数据集元数据显式给出；
      * 设备/dtype 从显式参数或目标模块取，**不默认 CPU**；
      * 形状/有限值/ID 范围校验贯穿。
    """

    def __init__(self, n_entities, num_relations, dim, unit=1.0, device=None, dtype=None):
        if not isinstance(n_entities, int) or n_entities <= 0:
            raise ValueError('n_entities must be a positive int')
        if not isinstance(num_relations, int) or num_relations <= 0:
            raise ValueError('num_relations (original R) must be a positive int')
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError('dim must be a positive int')
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError('unit must be positive and finite')
        self.n_entities, self.num_relations, self.dim = n_entities, num_relations, dim
        self.unit = float(unit)
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.dtype = torch.float32 if dtype is None else dtype

    def directed_axis(self):
        return 2 * self.num_relations

    def to(self, device, dtype=None):
        return FactAdapter(self.n_entities, self.num_relations, self.dim, self.unit,
                           device=device, dtype=self.dtype if dtype is None else dtype)

    def bank_from_slots(self, mean, rel, owner, timestamp, source=None):
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype)
        rel = torch.as_tensor(rel, device=self.device, dtype=torch.long)
        owner = torch.as_tensor(owner, device=self.device, dtype=torch.long)
        timestamp = torch.as_tensor(timestamp, device=self.device, dtype=self.dtype)
        _ids('slot relation', rel, self.directed_axis())
        _ids('slot owner', owner, self.n_entities)
        return EventBank(mean, rel, owner, timestamp, source=source, unit=self.unit)

    def bank_from_groups(self, groups, prediction_time, source=SOURCE_RECENT):
        """groups: list of 带 owner/relation/mean/timestamp 的帧组（如 M04 FrameGroups）。

        返回的 EventBank 已通过原始时间边界校验。
        """
        T = _as_scalar_time(prediction_time, 'prediction_time')
        means, rels, owners, stamps = [], [], [], []
        previous = None
        for group in groups:
            try:
                g_owner, g_rel, g_mean, g_ts = group.owner, group.relation, group.mean, group.timestamp
            except AttributeError:
                raise ValueError('groups must expose owner/relation/mean/timestamp')
            ts = _as_scalar_time(g_ts, 'group timestamp')
            if previous is not None and ts <= previous:
                raise ValueError('history groups must use strictly increasing raw timestamps')
            previous = ts
            if ts >= T:
                raise ValueError('history group is not strictly before prediction time')
            if not isinstance(g_mean, torch.Tensor) or not g_mean.is_floating_point():
                raise ValueError('group mean must be a floating tensor')
            if g_mean.device != self.device:
                raise ValueError('group mean device mismatch; adapter device is explicit, never assumed CPU')
            means.append(g_mean.to(self.dtype))
            rels.append(torch.as_tensor(g_rel, device=self.device, dtype=torch.long))
            owners.append(torch.as_tensor(g_owner, device=self.device, dtype=torch.long))
            stamps.append(torch.full((g_mean.shape[0],), ts, device=self.device, dtype=self.dtype))
        if not means:
            empty = torch.zeros(0, self.dim, device=self.device, dtype=self.dtype)
            return EventBank(empty, torch.zeros(0, dtype=torch.long, device=self.device),
                             torch.zeros(0, dtype=torch.long, device=self.device),
                             torch.zeros(0, device=self.device, dtype=self.dtype),
                             torch.zeros(0, dtype=torch.long, device=self.device), unit=self.unit)
        mean = torch.cat(means, 0)
        if mean.shape[1] != self.dim:
            raise ValueError('group mean dim disagrees with adapter dim')
        bank = EventBank(mean, torch.cat(rels, 0), torch.cat(owners, 0), torch.cat(stamps, 0),
                         torch.full((mean.shape[0],), int(source), dtype=torch.long, device=self.device),
                         unit=self.unit)
        bank.verify_before(T)
        return bank


def param_count(module):
    return sum(p.numel() for p in module.parameters())


def state_param_breakdown(dim=256, rank_k=8, rank_v=32):
    """门控模式参数单列，供承诺表与规格更新使用。"""
    shared = dim * rank_k + dim * rank_v + rank_v * dim
    return dict(shared=shared,
                global_gate=1,
                input_gate=2 * dim + 2 + 1,
                state_global=shared + 1,
                state_input=shared + 2 * dim + 3,
                delta_input_vs_global=(2 * dim + 3) - 1)
