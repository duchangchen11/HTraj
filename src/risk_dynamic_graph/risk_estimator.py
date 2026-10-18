"""Finite-horizon constant-relative-velocity geometry, not collision intent."""
import torch


def closest_approach(relative_position, relative_velocity, velocity_known,
                     horizon=4.8, distance_scale=1., speed_epsilon=1e-4):
    if horizon <= 0 or distance_scale <= 0 or speed_epsilon <= 0:
        raise ValueError('Positive scales required')
    speed2 = relative_velocity.square().sum(-1)
    moving = velocity_known & (speed2 > speed_epsilon ** 2)
    dot = (relative_position * relative_velocity).sum(-1)
    tau = torch.where(moving, -dot / speed2.clamp_min(speed_epsilon ** 2), 0.).clamp(0., horizon)
    closest = relative_position + tau[..., None] * relative_velocity
    dca = torch.linalg.vector_norm(closest, dim=-1)
    # Unknown relative velocity: tau=0, DCA=current distance, distance-only fallback.
    # Known equal velocities: same geometry but known flag remains distinct.
    risk = torch.exp(-0.5 * (dca / distance_scale).square() - tau / horizon)
    return dict(t_cpa=tau, dca=dca, risk=risk, velocity_known=velocity_known,
                approaching=moving & (dot < 0))
