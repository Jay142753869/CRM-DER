"""Forward-local dynamic message groups and subject-only target readout.

No current target objects, static events, learned activation caches, or CLI state
are inputs. Capture only the final layer of a strictly past dynamic frame.
"""
import math
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Sequence

import torch
from torch import nn


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('timestamp must be a finite real scalar in dataset time units')


def _ids(name, value, size=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(name + ' must be a one-dimensional int64 tensor')
    if value.numel() and (value.min().item() < 0 or (size is not None and value.max().item() >= size)):
        raise ValueError(name + ' ID outside permitted range')


@dataclass(frozen=True)
class FrameGroups:
    """Nonempty groups in one actual frame, including an empty frame container.

    owner is the global receiving entity; relation is the incoming directed ID.
    edge_messages, edge_group, partner are optional references from this forward.
    The four-item sequence view matches the legacy M05 composer API. Timestamp is
    explicit metadata and never replaces the externally supplied position vector.
    """
    owner: torch.Tensor
    relation: torch.Tensor
    mean: torch.Tensor
    count: torch.Tensor
    timestamp: Real
    edge_messages: Optional[torch.Tensor] = None
    edge_group: Optional[torch.Tensor] = None
    partner: Optional[torch.Tensor] = None

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (self.owner, self.relation, self.mean, self.count)[index]


def group_messages(owner, relation, messages, *, timestamp, partners=None, retain_edges=False):
    """Mean actual per-edge messages by (receiver, directed incoming relation).

    Call once per history frame. Repeated rows are repeated events. The input
    message tensor is neither transformed nor detached. No inverse is added here.
    """
    _timestamp(timestamp)
    _ids('owner', owner)
    _ids('relation', relation)
    if not isinstance(messages, torch.Tensor) or messages.ndim != 2 or not messages.is_floating_point():
        raise ValueError('messages must be floating point [edges, dim]')
    if messages.shape[1] <= 0 or owner.numel() != messages.shape[0] or relation.shape != owner.shape:
        raise ValueError('edge metadata and message shape mismatch')
    if owner.device != messages.device or relation.device != messages.device:
        raise ValueError('all group inputs must be on the same device')
    if retain_edges:
        _ids('partners', partners)
        if partners.shape != owner.shape or partners.device != messages.device:
            raise ValueError('partners must match edge count and device')
    keys, inverse, counts = torch.unique(torch.stack((owner, relation), 1), dim=0,
        sorted=True, return_inverse=True, return_counts=True)
    sums = messages.new_zeros((keys.shape[0], messages.shape[1])).index_add(0, inverse, messages)
    mean = sums / counts.to(messages.dtype).unsqueeze(1)
    return FrameGroups(keys[:, 0], keys[:, 1], mean, counts, timestamp,
        messages if retain_edges else None, inverse if retain_edges else None,
        partners if retain_edges else None)


def propagate_dynamic_last_layer(graph, message_fn, apply_fn, capture, *, timestamp,
                                 num_entities, num_relations, graph_kind,
                                 is_last_layer, retain_edges=False):
    """M01 hook: materialize once, then feed the SAME tensor to sum and groups.

    This covers neighbor propagation only; M01 retains its existing self-loop,
    skip, activation, dropout and static logic around the hook. graph.ndata['id']
    must map local nodes to global entities. num_relations is original R, so
    incoming edge types must already be in [0, 2R). DGL is imported only here.
    """
    if graph_kind != 'dynamic' or not is_last_layer:
        raise ValueError('C1 capture is allowed only on the final dynamic history layer')
    _timestamp(timestamp)
    if num_entities <= 0 or num_relations <= 0:
        raise ValueError('dataset N and original R must be positive')
    if 'msg' in graph.edata:
        raise ValueError("temporary edge field 'msg' is already occupied")
    import dgl.function as fn
    try:
        graph.apply_edges(message_fn)
        messages = graph.edata['msg']
        source, destination = graph.edges(order='eid')
        node_ids = graph.ndata['id'].reshape(-1)
        owner, partners = node_ids[destination], node_ids[source]
        relation = graph.edata['type'].reshape(-1)
        _ids('dynamic owner', owner, num_entities)
        _ids('dynamic partner', partners, num_entities)
        _ids('directed relation', relation, 2 * num_relations)
        groups = group_messages(owner, relation, messages, timestamp=timestamp,
                                partners=partners, retain_edges=retain_edges)
        graph.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'h'), apply_fn)
        capture.append(groups)
        return graph.ndata['h']
    finally:
        if 'msg' in graph.edata:
            graph.edata.pop('msg')


class PartnerMessageSketch(nn.Module):
    """Experimental identity/message binding residual for each group value.

    Signed deterministic tags are lossy identity features, not a partner lookup.
    The factorized control removes within-group tag/message covariance. The
    output starts at zero; its first gradient opens the path to inner projections.
    No additional scalar zero gate is used.
    """
    def __init__(self, dim, rank=16, seed=43):
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError('sketch dimensions must be positive')
        self.dim, self.rank = dim, rank
        # Signature seed must survive strict checkpoint loading as well as config.
        self.register_buffer('signature_seed', torch.tensor(int(seed) % 2147483647, dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.message = nn.Linear(dim, rank, bias=False)
            self.relation = nn.Linear(dim, rank, bias=False)
            self.output = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.output.weight)

    def signatures(self, partners, dtype):
        """Compute only observed IDs, avoiding an N x rank identity table.

        Products remain below signed int64 max after reduction modulo 2^31-1.
        Dataset IDs are nonnegative; reduction before multiplication is explicit.
        """
        _ids('partners', partners)
        prime = 2147483647
        axis = torch.arange(self.rank, device=partners.device, dtype=torch.long)
        value = (partners[:, None] % prime + 1 + (axis % prime) * 104729 + self.signature_seed) % prime
        value = (value * 48271 + 1) % prime
        value = (value * value) % prime
        return (2 * (value % 2) - 1).to(dtype) / math.sqrt(self.rank)

    def forward(self, groups: Sequence[FrameGroups], relations, mode='joint'):
        if mode not in ('joint', 'factorized'):
            raise ValueError('sketch mode must be joint or factorized')
        if relations.ndim != 2 or relations.shape[1] != self.dim or not relations.is_floating_point():
            raise ValueError('sketch relations must be floating [2R,D]')
        corrections = []
        for group in groups:
            if not isinstance(group, FrameGroups):
                raise ValueError('sketch groups must be FrameGroups')
            if any(x is None for x in (group.edge_messages, group.edge_group, group.partner)):
                raise ValueError('partner sketch requires retained real edge messages/group index/partner IDs')
            ng = group.owner.numel()
            _ids('edge_group', group.edge_group, ng)
            _ids('partner', group.partner)
            _ids('group directed relation', group.relation, relations.shape[0])
            _ids('group count', group.count)
            if (group.edge_messages.ndim != 2 or group.edge_messages.shape[1] != self.dim
                or group.edge_group.numel() != group.edge_messages.shape[0]
                or group.partner.shape != group.edge_group.shape
                or group.mean.shape != (ng, self.dim) or group.relation.numel() != ng
                or group.count.numel() != ng or (group.count <= 0).any().item()):
                raise ValueError('retained edge/group shapes or counts are inconsistent')
            tensors = (group.edge_messages, group.edge_group, group.partner, group.mean,
                       group.owner, group.relation, group.count)
            if (any(x.device != relations.device for x in tensors)
                or group.mean.dtype != relations.dtype or group.edge_messages.dtype != relations.dtype):
                raise ValueError('sketch edge/group tensors must share relation dtype/device')
            if not torch.equal(torch.bincount(group.edge_group, minlength=ng), group.count):
                raise ValueError('retained edge mapping does not reproduce physical event counts')
            tags = self.signatures(group.partner, relations.dtype)
            projected = self.message(group.edge_messages)
            denominator = group.count.to(projected.dtype)[:, None]
            if mode == 'joint':
                features = projected.new_zeros(ng, self.rank).index_add(0, group.edge_group, tags * projected)
                features = features / denominator
            else:
                tag_mean = tags.new_zeros(ng, self.rank).index_add(0, group.edge_group, tags) / denominator
                # Bias-free projection commutes with event-weighted mean.
                features = tag_mean * self.message(group.mean)
            gate = torch.sigmoid(self.relation(relations[group.relation]))
            corrections.append(self.output(features * gate))
        return torch.cat(corrections) if corrections else relations.new_zeros(0, self.dim)


class GroupedReadout(nn.Module):
    """Faithful C1; M05 supplies optional flattened group VALUE corrections.

    requests_sr is exactly [B,2], never a triple or an answer. The original
    formula and parameter names are retained for readout checkpoint compatibility.
    partner_sketch_rank=0 is the default and has no experimental parameters.
    """
    def __init__(self, dim, *, gamma_init=.01, seed=None, partner_sketch_rank=0,
                 partner_sketch_seed=43, partner_sketch_mode='joint'):
        super().__init__()
        if dim <= 0 or partner_sketch_rank < 0 or not math.isfinite(gamma_init):
            raise ValueError('invalid readout dimension, sketch rank or gamma')
        if partner_sketch_mode not in ('joint', 'factorized'):
            raise ValueError('sketch mode must be joint or factorized')
        context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        with context:
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            self.query = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.key = nn.Linear(3 * dim, dim, bias=False)
            self.value = nn.Linear(dim, dim, bias=False)
            self.output = nn.Linear(dim, dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.dim = dim
        self.partner_sketch_mode = partner_sketch_mode
        self.partner_sketch = (PartnerMessageSketch(dim, partner_sketch_rank, partner_sketch_seed)
                               if partner_sketch_rank else None)

    def _validate(self, embedding, relations, requests_sr, groups, positions, prediction_time, task):
        if task != 'entity':
            raise ValueError('C1 target-r readout supports entity prediction only; relation r is unknown otherwise')
        if embedding.ndim != 2 or relations.ndim != 2 or embedding.shape[1] != self.dim or relations.shape[1] != self.dim:
            raise ValueError('E/R must be [N,D]/[2R,D] with the configured D')
        if not embedding.is_floating_point() or relations.dtype != embedding.dtype or relations.device != embedding.device:
            raise ValueError('E/R floating dtype and device must match')
        if requests_sr.dtype != torch.long or requests_sr.ndim != 2 or requests_sr.shape[1] != 2:
            raise ValueError('requests_sr must contain exactly known (subject, relation), int64 [B,2]')
        if requests_sr.device != embedding.device:
            raise ValueError('requests must share embedding device')
        _ids('request subject', requests_sr[:, 0], embedding.shape[0])
        _ids('request relation', requests_sr[:, 1], relations.shape[0])
        if positions.shape != (len(groups), self.dim) or positions.device != embedding.device or positions.dtype != embedding.dtype:
            raise ValueError('positions must be [frames,D] on E dtype/device, including actual empty frames')
        if prediction_time is not None:
            _timestamp(prediction_time)
        previous = None
        for group in groups:
            if not isinstance(group, FrameGroups):
                raise ValueError('groups must be timestamped FrameGroups; adapt legacy tuples explicitly')
            _timestamp(group.timestamp)
            if previous is not None and group.timestamp <= previous:
                raise ValueError('history groups must use strictly increasing actual frame timestamps')
            if prediction_time is not None and group.timestamp >= prediction_time:
                raise ValueError('history must be strictly before prediction time')
            previous = group.timestamp
            _ids('group owner', group.owner, embedding.shape[0])
            _ids('group directed relation', group.relation, relations.shape[0])
            if group.mean.shape != (group.owner.numel(), self.dim) or group.relation.shape != group.owner.shape:
                raise ValueError('group shape mismatch')
            _ids('group count', group.count)
            if group.count.shape != group.owner.shape or (group.count <= 0).any().item():
                raise ValueError('materialized groups must have positive event counts')
            if any(x.device != embedding.device for x in (group.owner, group.relation, group.mean, group.count)) or group.mean.dtype != embedding.dtype:
                raise ValueError('group dtype/device mismatch')

    def forward(self, embedding, relations, requests_sr, groups, positions, *,
                value_correction=None, prediction_time=None, task='entity'):
        self._validate(embedding, relations, requests_sr, groups, positions, prediction_time, task)
        subjects, rel = requests_sr[:, 0], requests_sr[:, 1]
        base = embedding[subjects]
        slot_count = sum(group.owner.numel() for group in groups)
        if value_correction is not None and (value_correction.shape != (slot_count, self.dim)
            or value_correction.device != embedding.device or value_correction.dtype != embedding.dtype):
            raise ValueError('M05 value_correction must be [sum(frame groups),D] in flattened frame order')
        if not groups or not slot_count or not subjects.numel():
            return base
        query = self.query(torch.cat((base, relations[rel]), -1))
        owners = torch.cat([group.owner for group in groups])
        means = torch.cat([group.mean for group in groups])
        edge_rel = torch.cat([group.relation for group in groups])
        pos = torch.cat([positions[i].expand(group.owner.numel(), -1) for i, group in enumerate(groups)])
        keys = self.key(torch.cat((means, relations[edge_rel], pos), -1))
        values = self.value(means)
        if value_correction is not None:
            values = values + value_correction
        if self.partner_sketch is not None:
            values = values + self.partner_sketch(groups, relations, self.partner_sketch_mode)
        # No B x all-global-slots attention matrix. Each subject sees only its slots.
        result = torch.zeros_like(base)
        for subject in subjects.unique():
            rows = (subjects == subject).nonzero(as_tuple=True)[0]
            slots = (owners == subject).nonzero(as_tuple=True)[0]
            key = torch.cat((keys[slots], self.null_key.unsqueeze(0)), 0)
            value = torch.cat((values[slots], values.new_zeros(1, self.dim)), 0)
            weights = torch.softmax(query[rows] @ key.T / math.sqrt(self.dim), -1)
            result = result.index_copy(0, rows, weights @ value)
        return base + self.gamma.tanh() * self.output(result)
