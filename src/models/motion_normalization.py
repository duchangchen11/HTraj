"""Stage3.7 coordinate controls; archived motion models remain unchanged.

M1 is an alias: PedBase already uses derivatives and anchored displacement.
Heading is estimated from the current contiguous observation segment only.
Subthreshold/unknown heading uses the identity frame (not rotation equivariant).
The frame uses the whole available history: intermediate GRU states must NOT
be reused as prefix-causal social payloads without recomputing prefix frames.
"""
import torch
from .ego_encoder import motion_inputs
from .ped_base import PedBase
from .relation_routing import HistoricalCV


TranslationNormalizedPedBase = PedBase


def heading_frame(observed, valid=None, timestamps=None, threshold=.2):
    if threshold <= 0:
        raise ValueError('Positive heading threshold required')
    e = motion_inputs(observed, valid, timestamps)
    b, T, n, _ = observed.shape
    if timestamps is None:
        timestamps = torch.arange(T, device=observed.device, dtype=torch.float64)[None].expand(b, -1)*.4
    displacement = observed.new_zeros(b, n, 2)
    duration = observed.new_zeros(b, n)
    suffix = e['valid'][:, -1].clone()
    for t in range(T-1, 0, -1):
        suffix = suffix & e['continuity'][:, t]
        dt = (timestamps[:, t]-timestamps[:, t-1]).to(observed.dtype)[:, None]
        displacement = displacement+e['velocity'][:, t]*(dt*suffix)[..., None]
        duration = duration+dt*suffix
    mean_velocity = displacement/duration.clamp_min(1e-12)[..., None]
    speed = mean_velocity.norm(dim=-1)
    reliable = (duration > 0) & (speed >= threshold)
    fallback = torch.zeros_like(mean_velocity); fallback[..., 0] = 1
    unit = torch.where(reliable[..., None], mean_velocity/speed.clamp_min(1e-12)[..., None], fallback)
    return dict(origin=e['positions'][:, -1], unit=unit, reliable=reliable,
                mean_velocity=mean_velocity, heading_speed=speed, duration=duration,
                valid=e['valid'])


def to_local(vectors, unit):
    """Row vectors times R: global -> heading-aligned axes, no scale change."""
    c, s = unit[:, None, :, 0], unit[:, None, :, 1]
    x, y = vectors[..., 0], vectors[..., 1]
    return torch.stack([c*x+s*y, -s*x+c*y], -1)


def to_global(vectors, unit):
    c, s = unit[:, None, :, 0], unit[:, None, :, 1]
    x, y = vectors[..., 0], vectors[..., 1]
    return torch.stack([c*x-s*y, s*x+c*y], -1)


class HeadingNormalized(PedBase):
    """M2/M3: identical 35,352 parameters and initialization to PedBase."""
    def __init__(self, hidden_dim=64, prediction_steps=12, cv_residual=False, threshold=.2):
        super().__init__(hidden_dim, prediction_steps)
        self.cv_residual, self.threshold = cv_residual, threshold

    def forward(self, observed, valid=None, timestamps=None, return_debug=False):
        frame = heading_frame(observed, valid, timestamps, self.threshold)
        safe = torch.where(frame['valid'][..., None], observed, torch.zeros_like(observed))
        local = to_local(safe-frame['origin'][:, None], frame['unit'])
        local = torch.where(frame['valid'][..., None], local, torch.zeros_like(local))
        encoded = self.encoder(local, frame['valid'], timestamps)
        delta_local = self.decoder(encoded['hidden'][:, -1], torch.zeros_like(frame['origin']), frame['valid'][:, -1])
        delta_global = to_global(delta_local, frame['unit'])
        base = HistoricalCV()(observed, valid, timestamps) if self.cv_residual else frame['origin'][:, None].expand_as(delta_global)
        prediction = torch.where(frame['valid'][:, -1, None, :, None], base+delta_global, torch.zeros_like(delta_global))
        if not return_debug:
            return prediction
        return prediction, dict(**encoded, frame=frame, delta_local=delta_local,
                                ego_delta=delta_global, base_prediction=base)
