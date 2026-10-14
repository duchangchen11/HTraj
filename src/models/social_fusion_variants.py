"""Fusion-only changes. Raw A1 states and graph selection stay untouched."""
import math
import torch
from torch import nn


class ControlledSocialFusion(nn.Module):
    def __init__(self, original, hidden_dim, max_order, residual=False, normalization=False, initial_gate=0.05):
        super().__init__()
        self.residual, self.normalization = residual, normalization
        self.scores = original.scores
        self.normalizers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max_order)]) if normalization else nn.ModuleList()
        if residual:
            self.delta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.gate = nn.Linear(2*hidden_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, math.log(initial_gate/(1-initial_gate)))
        else:
            self.mix = original.mix

    def forward(self, ego, raw, support, intervention=None):
        intervention = intervention or {}
        support = support.clone()
        cap = intervention.get('order_cap', raw.shape[2])
        support[:, :, cap:] = False
        used = raw
        if self.normalization and not intervention.get('disable_normalization', False):
            used = torch.stack([norm(raw[:, :, q]) for q, norm in enumerate(self.normalizers)], 2)
        # Mask AFTER affine LN. A learned beta must not create an invalid order.
        used = torch.where(support[..., None], used, torch.zeros_like(used))
        scores = torch.cat([scorer(torch.cat([ego, used[:, :, q]], -1))
                            for q, scorer in enumerate(self.scores)], -1)
        available = support.any(-1)
        scores = scores.masked_fill(~support, -torch.inf)
        scores = torch.where(available[..., None], scores, torch.zeros_like(scores))
        beta = torch.softmax(scores, -1)*support.to(scores.dtype)
        pooled = (used*beta[..., None]).sum(2)
        gate = ego.new_zeros(ego.shape[:-1])
        delta = torch.zeros_like(ego)
        if self.residual:
            delta = self.delta(pooled)
            gate = torch.sigmoid(self.gate(torch.cat([ego, pooled], -1))).squeeze(-1)
            if 'fixed_gate' in intervention:
                gate = torch.full_like(gate, float(intervention['fixed_gate']))
            gate = torch.where(available, gate, torch.zeros_like(gate))
            context = ego+gate[..., None]*delta
        else:
            context = self.mix(torch.cat([ego, pooled], -1))
            context = torch.where(available[..., None], context, ego)
        if intervention.get('disable_residual', False):
            context, gate = ego, torch.zeros_like(gate)
        return context, beta, dict(used_order_states=used, order_support_last=support,
            residual_gate=gate, social_delta=delta, fused_context=context,
            residual_enabled=self.residual, normalization_enabled=self.normalization)
