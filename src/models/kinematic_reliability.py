"""Stage3.8 parameter-free, history-only kinematics. No archive mutations.

All estimates extrapolate from the LAST observed position, including OLS.
Only the current contiguous suffix is usable after a gap. OLS uses up to K
observations, with <2 observations falling back to explicit zero velocity.
Component median averages the middle two values when the count is even.
"""
import torch
from torch import nn
from .ego_encoder import motion_inputs

NAMES = ('CV-Last2', 'CV-OLS4', 'CV-OLS8', 'CV-Median')


def history_state(observed, valid=None, timestamps=None):
    e = motion_inputs(observed, valid, timestamps)
    b, T, n, _ = observed.shape
    if timestamps is None:
        timestamps = torch.arange(T, device=observed.device, dtype=torch.float64)[None].expand(b, -1)*.4
    suffix = torch.zeros_like(e['valid']); suffix[:, -1] = e['valid'][:, -1]
    for t in range(T-2, -1, -1): suffix[:, t] = suffix[:, t+1] & e['continuity'][:, t+1]
    dt = (timestamps[:, 1:]-timestamps[:, :-1]).to(observed.dtype)[:, :, None]
    steps = suffix[:, 1:] & suffix[:, :-1]
    return e, timestamps, suffix, steps, dt


def fit_velocity(e, timestamps, suffix, points):
    T = suffix.shape[1]; keep = suffix.clone()
    if T > points: keep[:, :T-points] = False
    weights = keep.to(e['positions'].dtype); count = weights.sum(1)
    time = (timestamps-timestamps[:, -1:]).to(weights.dtype)[:, :, None]
    time_mean = (time*weights).sum(1)/count.clamp_min(1)
    # Center spatial coordinates at the cutoff before fitting for stability.
    x = e['positions']-e['positions'][:, -1:]
    mean = (x*weights[..., None]).sum(1)/count.clamp_min(1)[..., None]
    centered_t = time-time_mean[:, None]
    denominator = (centered_t.square()*weights).sum(1)
    numerator = (centered_t[..., None]*(x-mean[:, None])*weights[..., None]).sum(1)
    velocity = torch.where((count >= 2)[..., None], numerator/denominator.clamp_min(1e-12)[..., None], torch.zeros_like(numerator))
    fitted = mean[:, None]+centered_t[..., None]*velocity[:, None]
    squared_residual = ((x-fitted).square().sum(-1)*weights).sum(1)/count.clamp_min(1)
    return velocity, squared_residual.sqrt(), count


def estimated_velocity(observed, valid=None, timestamps=None, name='CV-Last2'):
    if name not in NAMES: raise ValueError(name)
    e, ts, suffix, steps, dt = history_state(observed, valid, timestamps)
    if name == 'CV-Last2': velocity = e['velocity'][:, -1]
    elif name in ('CV-OLS4', 'CV-OLS8'):
        velocity = fit_velocity(e, ts, suffix, 4 if name == 'CV-OLS4' else 8)[0]
    else:
        count = steps.sum(1)
        values = torch.where(steps[..., None], e['velocity'][:, 1:], torch.full_like(e['velocity'][:, 1:], torch.inf)).sort(dim=1).values
        if values.shape[1] == 0: return torch.zeros_like(e['positions'][:, -1])
        lower = ((count-1).clamp_min(0)//2)[:, None, :, None].expand(-1, 1, -1, 2)
        upper = (count//2)[:, None, :, None].expand_as(lower)
        med = (values.gather(1, lower).squeeze(1)+values.gather(1, upper).squeeze(1))*.5
        velocity = torch.where((count > 0)[..., None], med, torch.zeros_like(med))
    return torch.where(e['valid'][:, -1, :, None], velocity, torch.zeros_like(velocity))


class KinematicPredictor(nn.Module):
    def __init__(self, name='CV-Last2'):
        super().__init__()
        if name not in NAMES: raise ValueError(name)
        self.name = name

    def forward(self, observed, valid=None, timestamps=None):
        e = motion_inputs(observed, valid, timestamps)
        velocity = estimated_velocity(observed, valid, timestamps, self.name)
        horizon = torch.arange(1, 13, device=observed.device, dtype=observed.dtype)*.4
        y = e['positions'][:, -1, None]+velocity[:, None]*horizon[None, :, None, None]
        return torch.where(e['valid'][:, -1, None, :, None], y, torch.zeros_like(y))


def reliability_features(observed, valid=None, timestamps=None):
    e, ts, suffix, steps, dt = history_state(observed, valid, timestamps)
    velocity = e['velocity'][:, 1:]; count = steps.sum(1)
    duration = (dt*steps).sum(1)
    mean = (velocity*dt[..., None]*steps[..., None]).sum(1)/duration.clamp_min(1e-12)[..., None]
    speed = velocity.norm(dim=-1)
    mean_scalar_speed = (speed*dt*steps).sum(1)/duration.clamp_min(1e-12)
    variance = ((velocity-mean[:, None]).square().sum(-1)*steps).sum(1)/count.clamp_min(1)
    path = (speed*dt*steps).sum(1); net = (mean*duration[..., None]).norm(dim=-1)
    pair = steps[:, 1:] & steps[:, :-1]
    change = (velocity[:, 1:]-velocity[:, :-1]).norm(dim=-1)
    mean_change = (change*pair).sum(1)/pair.sum(1).clamp_min(1)
    last = e['velocity'][:, -1]; last_speed = last.norm(dim=-1); mean_speed = mean.norm(dim=-1)
    angle_known = (last_speed >= .2) & (mean_speed >= .2) & (count > 0)
    cosine = (last*mean).sum(-1)/(last_speed*mean_speed).clamp_min(1e-12)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    angle = torch.where(angle_known, angle, torch.zeros_like(angle))
    # Angular span over ALL moving historical velocity pairs also detects
    # reversal after a pause, which adjacent moving-pair angles would miss.
    pair_known = steps[:, :, None] & steps[:, None, :] & (speed[:, :, None] >= .2) & (speed[:, None, :] >= .2)
    upper = torch.ones(velocity.shape[1], velocity.shape[1], dtype=torch.bool, device=observed.device).triu(1)
    pair_known = pair_known & upper[None, :, :, None]
    pair_cos = (velocity[:, :, None]*velocity[:, None, :]).sum(-1)/(speed[:, :, None]*speed[:, None, :]).clamp_min(1e-12)
    angles = torch.where(pair_known, torch.rad2deg(torch.acos(pair_cos.clamp(-1, 1))), torch.zeros_like(pair_cos))
    max_turn = angles.amax((1, 2)) if velocity.shape[1] else last_speed*0
    ols8, rms, _ = fit_velocity(e, ts, suffix, 8)
    max_speed = torch.where(steps, speed, torch.zeros_like(speed)).max(1).values if speed.shape[1] else last_speed*0
    return dict(history_frames=e['valid'].sum(1), contiguous_history_frames=suffix.sum(1), velocity_observations=count,
        last_velocity_x_mps=last[..., 0], last_velocity_y_mps=last[..., 1], last_speed_mps=last_speed,
        mean_velocity_x_mps=mean[..., 0], mean_velocity_y_mps=mean[..., 1], mean_velocity_speed_mps=mean_speed,
        mean_scalar_speed_mps=mean_scalar_speed, velocity_variance_m2ps2=variance,
        adjacent_velocity_change_mps=mean_change, net_displacement_m=net, total_path_m=path,
        net_path_ratio=torch.where(path > 1e-9, net/path.clamp_min(1e-12), torch.ones_like(path)),
        net_path_ratio_defined=path > 1e-9, last_mean_angle_deg=angle, last_mean_angle_known=angle_known,
        maximum_turn_angle_deg=max_turn, turn_angle_observations=pair_known.sum((1, 2)), linear_fit_RMS_m=rms,
        terminal_mean_velocity_difference_mps=(last-mean).norm(dim=-1), maximum_speed_mps=max_speed,
        last_OLS8_velocity_difference_mps=(last-ols8).norm(dim=-1))


def reliability_groups(f):
    visible = f['history_frames'] > 0; known = f['velocity_observations'] > 0
    speed = f['mean_velocity_speed_mps']; var = f['velocity_variance_m2ps2']; rms = f['linear_fit_RMS_m']
    unstable = f['terminal_mean_velocity_difference_mps'] >= .5
    turn = ((f['turn_angle_observations'] > 0) & (f['maximum_turn_angle_deg'] >= 60)) | (f['last_mean_angle_known'] & (f['last_mean_angle_deg'] >= 60))
    return dict(all=visible, history_full8=visible & (f['history_frames'] == 8), history_partial=visible & (f['history_frames'] < 8),
        velocity_unknown=visible & ~known, history_speed_low=visible & known & (speed < .2),
        history_speed_medium=visible & known & (speed >= .2) & (speed < 1), history_speed_high=visible & known & (speed >= 1),
        history_quasi_stationary=visible & known & (f['maximum_speed_mps'] <= .1) & (f['total_path_m'] <= .2) & (f['net_displacement_m'] <= .15),
        history_out_and_back=visible & known & (f['total_path_m'] >= .5) & (f['net_path_ratio'] <= .5),
        history_direction_changing=visible & known & turn, history_terminal_unstable=visible & known & unstable,
        history_terminal_stable=visible & known & ~unstable,
        variance_low=visible & known & (var < .01), variance_medium=visible & known & (var >= .01) & (var < .25), variance_high=visible & known & (var >= .25),
        fit_residual_low=visible & known & (rms < .02), fit_residual_medium=visible & known & (rms >= .02) & (rms < .1), fit_residual_high=visible & known & (rms >= .1),
        direction_consistent=visible & known & (f['net_path_ratio'] >= .9) & (f['last_mean_angle_known']) & (f['last_mean_angle_deg'] < 30),
        direction_discordant=visible & known & (f['last_mean_angle_known']) & (f['last_mean_angle_deg'] >= 60))
