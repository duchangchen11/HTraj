"""Stage3.6 CV anchor and matched, first-order residual information controls.

Archived model implementations are never changed. All graph inputs are
observations. Uniform spatial/time pooling introduces no learned attention,
edge memory, higher-order state, or new prediction objective.
"""
import torch
from torch import nn
from .ped_base import PedBase
from .relation_routing import HistoricalCV
from .ego_encoder import motion_inputs
from .temporal_graph import select_candidates


class CVResidual(PedBase):
    """Same encoder/head/initialization/parameter count as PedBase; add CV."""
    def forward(self, observed, valid=None, timestamps=None, return_debug=False):
        encoded = self.encoder(observed, valid, timestamps)
        delta = self.decoder(encoded['hidden'][:, -1], torch.zeros_like(encoded['positions'][:, -1]), encoded['valid'][:, -1])
        cv = HistoricalCV()(observed, valid, timestamps)
        prediction = cv + delta
        return (prediction, dict(**encoded, cv_prediction=cv, ego_delta=delta)) if return_debug else prediction


def random_candidates(encoded, t, real, radius=4.):
    """Same scene/time, known velocities, no self, radius, incoming budget.

    A coordinate/time hash ranks eligible relations without agent-index ties.
    Ties are dropped as a whole; any rare budget discrepancy is recorded.
    With <=k eligible sources, edge sets cannot be randomized at all.
    """
    pos, known = encoded['positions'][:, t], encoded['continuity'][:, t]
    n = pos.shape[1]
    distance = (pos[:, :, None] - pos[:, None, :]).norm(dim=-1)
    eligible = known[:, :, None] & known[:, None, :] & (distance <= radius)
    eligible &= ~torch.eye(n, device=pos.device, dtype=torch.bool)[None]
    a, z = pos[:, :, None].double(), pos[:, None, :].double()
    score = torch.frac(torch.sin(a[..., 0]*12.9898 + a[..., 1]*78.233 + z[..., 0]*39.425 + z[..., 1]*11.135 + (t+1)*.618)*43758.5453).abs()
    budget = real.sum(-1)
    ranked = torch.where(eligible, score, torch.full_like(score, torch.inf)).sort(-1).values
    boundary = ranked.gather(-1, (budget-1).clamp_min(0)[..., None])
    inclusive = eligible & (score <= boundary) & (budget > 0)[..., None]
    strict = eligible & (score < boundary) & (budget > 0)[..., None]
    return torch.where((inclusive.sum(-1) > budget)[..., None], strict, inclusive)


class FrozenEgoResidual(nn.Module):
    MODES = ('Ego-Only', 'Ego-Residual', 'Social-Previous', 'Social-Current', 'Social-Random')

    def __init__(self, anchor, mode, cv_output=False, hidden_dim=64, prediction_steps=12, radius=4., max_neighbors=8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        self.anchor, self.mode, self.cv_output = anchor, mode, cv_output
        self.radius, self.max_neighbors = radius, max_neighbors
        self.steps, self.hidden_dim = prediction_steps, hidden_dim
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.eval()
        if mode != 'Ego-Only':
            self.payload = nn.Sequential(nn.Linear(hidden_dim+7, hidden_dim), nn.ReLU())
            # One identical direct 12-step correction MLP in every control.
            self.delta = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, prediction_steps*2))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            self.gate_logit = nn.Parameter(torch.tensor(-2.))

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    @torch.no_grad()
    def encode_anchor(self, observed, valid=None, timestamps=None):
        base, encoded = self.anchor(observed, valid, timestamps, return_debug=True)
        if self.cv_output:
            base = HistoricalCV()(observed, valid, timestamps)
        return base.detach(), {k: v.detach() if torch.is_tensor(v) else v for k, v in encoded.items()}

    def forward_encoded(self, encoded, base, return_debug=False, remove_messages=False):
        hidden = encoded['hidden'].detach()
        b, T, n, d = hidden.shape
        if self.mode == 'Ego-Only' and not return_debug:
            return base
        total = hidden.new_zeros(b*n, d)
        frames = hidden.new_zeros(b*n)
        real_edges = used_edges = changed = budget_difference = 0
        real_distance, used_distance = [], []
        per_frame_messages = []
        for t in range(1, T):
            if self.mode.startswith('Social-') or return_debug:
                real, geometry = select_candidates(encoded['positions'][:, t], encoded['velocity'][:, t], encoded['continuity'][:, t], self.radius, self.max_neighbors)
                selected = random_candidates(encoded, t, real, self.radius) if self.mode == 'Social-Random' else real
                if remove_messages:
                    selected = torch.zeros_like(selected)
                if return_debug:
                    real_edges += int(real.sum())
                    used_edges += int(selected.sum()) if self.mode.startswith('Social-') else 0
                    changed += int((selected & ~real).sum()) if self.mode.startswith('Social-') else 0
                    budget_difference += int((selected.sum(-1)-real.sum(-1)).abs().sum()) if self.mode.startswith('Social-') else 0
                    distance = (encoded['positions'][:, t, :, None]-encoded['positions'][:, t, None, :]).norm(dim=-1)
                    real_distance.append(distance[real].detach())
                    used_distance.append(distance[selected].detach() if self.mode.startswith('Social-') else hidden.new_zeros(0))
            if self.mode == 'Ego-Only':
                continue
            if self.mode == 'Ego-Residual':
                frame_support = encoded['continuity'][:, t] & ~torch.full_like(encoded['continuity'][:, t], remove_messages)
                own = torch.cat([hidden[:, t-1], encoded['motion_features'][:, t]], -1)
                message = self.payload(own).reshape(b*n, d)*frame_support.reshape(-1, 1)
            else:
                bi, target, source = selected.nonzero(as_tuple=True)
                source_t = t if self.mode == 'Social-Current' else t-1
                edge_features = torch.cat([geometry[bi, target, source, :6], torch.ones(len(target), 1, device=hidden.device)], -1)
                contents = torch.cat([hidden[bi, source_t, source], edge_features], -1)
                value = self.payload(contents)
                flat_target = bi*n+target
                degree = hidden.new_zeros(b*n).index_add(0, flat_target, torch.ones(len(target), device=hidden.device))
                message = hidden.new_zeros(b*n, d).index_add(0, flat_target, value)/degree[:, None].clamp_min(1)
                frame_support = (degree > 0).reshape(b, n)
            # Temporal gaps invalidate previously pooled target messages.
            continuity = encoded['continuity'][:, t].reshape(-1)
            total = total*continuity[:, None] + message
            frames = frames*continuity + frame_support.reshape(-1)
            per_frame_messages.append(message.reshape(b, n, d))
        has_message = (frames > 0).reshape(b, n) & encoded['valid'][:, -1]
        if self.mode == 'Ego-Only':
            correction = torch.zeros_like(base)
            raw_delta = torch.zeros_like(base)
            gate = hidden.new_zeros(b, n)
            pooled = hidden.new_zeros(b, n, d)
            prediction = base
        else:
            pooled = (total/frames[:, None].clamp_min(1)).reshape(b, n, d)
            context = torch.cat([hidden[:, -1], pooled], -1)
            raw_delta = self.delta(context).reshape(b, n, self.steps, 2).permute(0, 2, 1, 3)
            gate = torch.sigmoid(self.gate_logit).expand(b, n)*has_message
            correction = gate[:, None, :, None]*raw_delta
            # Exact equality, including zero corrections for unsupported nodes.
            prediction = torch.where(has_message[:, None, :, None], base+correction, base)
        if not return_debug:
            return prediction
        empty = hidden.new_zeros(0)
        debug = dict(base_prediction=base, has_message=has_message, correction=correction, raw_delta=raw_delta, gate=gate,
            pooled_message=pooled, hidden=hidden, valid=encoded['valid'], real_edges=real_edges, used_edges=used_edges,
            changed_edges=changed, incoming_budget_absolute_difference=budget_difference,
            real_edge_distance=torch.cat(real_distance) if real_distance else empty,
            used_edge_distance=torch.cat(used_distance) if used_distance else empty,
            frame_messages=torch.stack(per_frame_messages, 1) if per_frame_messages else hidden.new_zeros(b, 0, n, d))
        return prediction, debug

    def forward(self, observed, valid=None, timestamps=None, return_debug=False, remove_messages=False):
        base, encoded = self.encode_anchor(observed, valid, timestamps)
        return self.forward_encoded(encoded, base, return_debug, remove_messages)

    def forward_cached(self, batch, return_debug=False):
        encoded = motion_inputs(batch['observed'], batch['observed_valid'], batch['timestamps'])
        encoded['hidden'] = batch['anchor_hidden']
        return self.forward_encoded(encoded, batch['anchor_prediction'], return_debug)
