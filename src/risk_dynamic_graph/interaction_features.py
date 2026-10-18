"""Only backward differences within the supplied history prefix, in m/s."""
import torch


def historical_motion(position, valid, dt=0.4):
    if position.ndim != 4 or position.shape[-1] != 2 or valid.shape != position.shape[:-1]:
        raise ValueError('Expected position[B,T,N,2], valid[B,T,N]')
    if valid.dtype != torch.bool or dt <= 0 or position.shape[1] < 1:
        raise ValueError('Boolean mask and positive dt required')
    if not torch.isfinite(position[valid]).all():
        raise ValueError('Nonfinite observable position')
    clean = torch.where(valid[..., None], position, torch.zeros_like(position))
    vv = torch.zeros_like(valid)
    vv[:, 1:] = valid[:, 1:] & valid[:, :-1]
    v = torch.zeros_like(clean)
    v[:, 1:] = torch.where(vv[:, 1:, :, None], (clean[:, 1:] - clean[:, :-1]) / dt, 0.)
    return clean, v, vv


def pair_features(position, velocity, velocity_valid):
    # A[...,i,j] carries j -> i. Relative orientation is source minus receiver.
    r = position[..., None, :, :] - position[..., :, None, :]
    w = velocity[..., None, :, :] - velocity[..., :, None, :]
    known = velocity_valid[..., :, None] & velocity_valid[..., None, :]
    w = torch.where(known[..., None], w, 0.)
    return r, w, known
