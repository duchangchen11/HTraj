"""Prefix-only motion features and shared per-pedestrian GRU."""
import torch
from torch import nn


def motion_inputs(observed, valid=None, timestamps=None, nominal_dt=0.4):
    if observed.ndim != 4 or observed.shape[-1] != 2 or min(observed.shape[:3]) < 1:
        raise ValueError('Expected [B,T,N,2] with positive dimensions')
    b, t_steps, n, _ = observed.shape
    if valid is None:
        valid = torch.ones((b, t_steps, n), dtype=torch.bool, device=observed.device)
    if valid.shape != observed.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('valid must be Boolean [B,T,N]')
    if not torch.isfinite(observed[valid]).all():
        raise ValueError('Nonfinite valid coordinates')
    positions = torch.where(valid[..., None], observed, torch.zeros_like(observed))
    if timestamps is None:
        timestamps = torch.arange(t_steps, device=observed.device, dtype=torch.float64)[None].expand(b, -1)*nominal_dt
    if timestamps.shape != (b, t_steps) or not torch.isfinite(timestamps).all():
        raise ValueError('timestamps must be finite [B,T]')
    intervals = timestamps[:, 1:]-timestamps[:, :-1]
    if (intervals <= 0).any():
        raise ValueError('Timestamps must increase strictly')
    previous_velocity = positions.new_zeros(b, n, 2)
    previous_motion_valid = torch.zeros(b, n, dtype=torch.bool, device=positions.device)
    velocities, continuity, features = [], [], []
    for t in range(t_steps):
        motion_valid = torch.zeros_like(previous_motion_valid)
        velocity = torch.zeros_like(previous_velocity)
        acceleration = torch.zeros_like(velocity)
        if t:
            dt = intervals[:, t-1].to(positions.dtype)[:, None, None]
            # A long sampling gap starts a fresh track state; not a normal step.
            step_valid = (intervals[:, t-1] <= nominal_dt*1.5)[:, None]
            motion_valid = valid[:, t] & valid[:, t-1] & step_valid
            velocity = torch.where(motion_valid[..., None], (positions[:, t]-positions[:, t-1])/dt, velocity)
            acceleration_valid = motion_valid & previous_motion_valid
            acceleration = torch.where(acceleration_valid[..., None], (velocity-previous_velocity)/dt, acceleration)
        else:
            acceleration_valid = motion_valid
        feature = torch.cat([velocity, acceleration, valid[:, t, :, None].to(positions.dtype),
                             motion_valid[..., None].to(positions.dtype), acceleration_valid[..., None].to(positions.dtype)], -1)
        velocities.append(velocity)
        continuity.append(motion_valid)
        features.append(feature)
        previous_velocity, previous_motion_valid = velocity, motion_valid
    return dict(velocity=torch.stack(velocities, 1), continuity=torch.stack(continuity, 1),
                motion_features=torch.stack(features, 1), positions=positions, valid=valid)


class EgoEncoder(nn.Module):
    def __init__(self, hidden_dim=64, nominal_dt=0.4):
        super().__init__()
        self.hidden_dim, self.nominal_dt = hidden_dim, nominal_dt
        self.motion_embedding = nn.Sequential(nn.Linear(7, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, observed, valid=None, timestamps=None):
        encoded = motion_inputs(observed, valid, timestamps, self.nominal_dt)
        b, t_steps, n, _ = observed.shape
        state = observed.new_zeros(b, n, self.hidden_dim)
        states = []
        for t in range(t_steps):
            old = torch.where(encoded['continuity'][:, t, :, None], state, torch.zeros_like(state))
            feature = encoded['motion_features'][:, t]
            state = self.gru(self.motion_embedding(feature).reshape(b*n, -1), old.reshape(b*n, -1)).reshape(b, n, -1)
            state = torch.where(encoded['valid'][:, t, :, None], state, torch.zeros_like(state))
            states.append(state)
        return dict(**encoded, hidden=torch.stack(states, 1))
