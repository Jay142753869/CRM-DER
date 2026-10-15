"""C2: faithful decoder-query temporal selection and full candidate scoring."""

import math
from typing import Optional

import torch
from torch import Tensor, nn


class QueryHistoryScoreSelector(nn.Module):
    """Return gamma-scaled [B,N] logits, never modify the decoder or bank.

    H_raw is pre-MA [N,L,D], padding [N,L] is True for absent time slots.
    requests_sr [B,2] contains only known subject and directed relation IDs.
    Attention is query-level, shared over all candidates. The default exactly
    preserves the original QDHS formula on its common snapshot padding domain.
    """

    def __init__(self, h_dim: int, max_history_len: int, hidden_dim: int = 64,
                 use_time_bias: bool = False, gamma_init: float = 0.0):
        super().__init__()
        if min(h_dim, max_history_len, hidden_dim) <= 0:
            raise ValueError('dimensions and max_history_len must be positive')
        if not math.isfinite(gamma_init):
            raise ValueError('gamma_init must be finite')
        self.h_dim = h_dim
        self.max_history_len = max_history_len
        self.W_h = nn.Linear(h_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(h_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.use_time_bias = use_time_bias
        if use_time_bias:
            self.time_bias = nn.Parameter(torch.zeros(max_history_len))
        self.last_alpha = None
        self.last_selector_stats = {}

    def _validate(self, query_x, H_raw, requests_sr, padding, slot_ids,
                  task, bank_kind):
        if task != 'entity':
            raise ValueError('C2 supports entity prediction only; disable C2 for relation prediction')
        if bank_kind != 'H_raw':
            raise ValueError('C2 requires H_raw pre-MA; V2 post-MA history is a different mechanism')
        if query_x.ndim != 2 or H_raw.ndim != 3:
            raise ValueError('query_x must be [B,D] and H_raw [N,L,D]')
        batch, dim = query_x.shape
        n, length, bank_dim = H_raw.shape
        if dim != self.h_dim or bank_dim != dim:
            raise ValueError('query and H_raw hidden dimensions must match the selector')
        if requests_sr.shape != (batch, 2) or requests_sr.dtype != torch.long:
            raise ValueError('requests_sr must be LongTensor[B,2]; targets must stay separate')
        if padding.shape != (n, length) or padding.dtype != torch.bool:
            raise ValueError('padding must be BoolTensor[N,L], True means absent time slot')
        if not query_x.is_floating_point() or query_x.dtype != H_raw.dtype:
            raise ValueError('query_x and H_raw must share a floating dtype')
        if query_x.dtype != self.W_h.weight.dtype:
            raise ValueError('move selector and inputs to the same dtype')
        if any(t.device != query_x.device for t in
               (H_raw, requests_sr, padding, self.W_h.weight)):
            raise ValueError('selector and inputs must share a device')
        if batch and (requests_sr[:, 0].min().item() < 0 or
                      requests_sr[:, 0].max().item() >= n or
                      requests_sr[:, 1].min().item() < 0):
            raise ValueError('subject index outside H_raw or negative directed relation ID')
        if slot_ids is not None:
            if slot_ids.shape != (length,) or slot_ids.dtype != torch.long:
                raise ValueError('slot_ids must be LongTensor[L]')
            if slot_ids.device != query_x.device:
                raise ValueError('slot_ids must share input device')
            if length and (slot_ids.min().item() < 0 or
                           slot_ids.max().item() >= self.max_history_len):
                raise ValueError('slot_ids outside the configured MA slot table')
            if length > 1 and not (slot_ids[1:] > slot_ids[:-1]).all().item():
                raise ValueError('slot_ids must follow strictly increasing MA slot order')
        elif self.use_time_bias and length:
            raise ValueError('time_bias requires explicit slot_ids aligned with MA positions')

    def _weights(self, query_x, clean_bank, requests_sr, padding, slot_ids):
        batch, length = query_x.size(0), clean_bank.size(1)
        if batch == 0 or length == 0:
            return query_x.new_zeros((batch, length))
        heads = requests_sr[:, 0]
        logits = self.v(torch.tanh(self.W_h(clean_bank[heads]) +
                                  self.W_q(query_x).unsqueeze(1))).squeeze(-1)
        if self.use_time_bias:
            logits = logits + self.time_bias[slot_ids].unsqueeze(0)
        head_padding = padding[heads]
        # Avoid softmax(-inf, ..., -inf); empty rows are exactly zero after masking.
        masked = logits.masked_fill(head_padding, float('-inf'))
        masked = torch.where(head_padding.all(dim=1, keepdim=True),
                             torch.zeros_like(masked), masked)
        return torch.softmax(masked, dim=-1).masked_fill(head_padding, 0.0)

    def temporal_weights(self, query_x: Tensor, H_raw: Tensor, requests_sr: Tensor,
                         padding: Tensor, *, slot_ids: Optional[Tensor] = None,
                         task: str = 'entity', bank_kind: str = 'H_raw') -> Tensor:
        """Differentiable [B,L] alpha for diagnostics, without cached autograd graphs."""
        self._validate(query_x, H_raw, requests_sr, padding, slot_ids, task, bank_kind)
        clean_bank = H_raw.masked_fill(padding.unsqueeze(-1), 0.0)
        return self._weights(query_x, clean_bank, requests_sr, padding, slot_ids)

    def _record_stats(self, alpha, head_padding, enabled):
        zero = self.gamma.detach().new_zeros(())
        if alpha.size(0) and alpha.size(1):
            count = (~head_padding).sum(-1).to(alpha.dtype)
            entropy = -(alpha * alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(-1)
            normalized_entropy = (entropy / count.clamp_min(2).log()).mean()
            variance = alpha.var(dim=0, unbiased=False).mean()
            empty_fraction = (count == 0).to(alpha.dtype).mean()
        else:
            normalized_entropy = variance = zero
            empty_fraction = zero if not alpha.size(0) else zero + 1
        self.last_alpha = alpha.detach()
        self.last_selector_stats = {
            'gamma': self.gamma.detach().clone(),
            'alpha_entropy': normalized_entropy.detach(),
            'alpha_variance': variance.detach(),
            'empty_head_fraction': empty_fraction.detach(),
            'enabled': zero + float(enabled),
        }

    def forward(self, query_x: Tensor, H_raw: Tensor, requests_sr: Tensor,
                padding: Tensor, *, slot_ids: Optional[Tensor] = None,
                enabled: bool = True, task: str = 'entity',
                bank_kind: str = 'H_raw') -> Tensor:
        """Train and predict use exactly this same full-N, gold-free path.

        Padding values are removed before projections/scoring. For differing
        candidate masks, absent candidate slots contribute zero without
        candidate-specific renormalization of the shared query attention.
        """
        self._validate(query_x, H_raw, requests_sr, padding, slot_ids, task, bank_kind)
        batch, n, length = query_x.size(0), H_raw.size(0), H_raw.size(1)
        if not enabled:
            alpha = query_x.new_zeros((batch, length))
            self._record_stats(alpha, padding[requests_sr[:, 0]], enabled=False)
            return query_x.new_zeros((batch, n))
        clean_bank = H_raw.masked_fill(padding.unsqueeze(-1), 0.0)
        alpha = self._weights(query_x, clean_bank, requests_sr, padding, slot_ids)
        self._record_stats(alpha, padding[requests_sr[:, 0]], enabled=True)
        if batch == 0 or length == 0:
            return self.gamma * query_x.new_zeros((batch, n))
        # Reassociate sum_l,d alpha[b,l]*q[b,d]*H[c,l,d], avoiding [B,L,N].
        weighted_query = (alpha.unsqueeze(-1) * query_x.unsqueeze(1)).reshape(batch, -1)
        candidate_bank = clean_bank.reshape(n, length * self.h_dim)
        return self.gamma * torch.mm(weighted_query, candidate_bank.transpose(0, 1))
