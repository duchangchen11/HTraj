import pytest
import torch
from src.models.kinematic_reliability import (NAMES, KinematicPredictor, estimated_velocity,
                                            reliability_features, reliability_groups)
from src.models.relation_routing import HistoricalCV


def line():
    t = torch.arange(8).double()[None]*.4
    x = torch.zeros(1, 8, 2, 2).double()
    x[:, :, 0, 0] = t*1.3+14
    x[:, :, 0, 1] = t*(-.4)-7
    x[:, :, 1] = 3
    return x, torch.ones(x.shape[:-1], dtype=torch.bool), t


@pytest.mark.parametrize('name', NAMES)
def test_exact_linear_velocity_and_anchor(name):
    x, v, t = line(); model = KinematicPredictor(name)
    assert sum(p.numel() for p in model.parameters()) == 0
    speed = estimated_velocity(x, v, t, name)
    torch.testing.assert_close(speed, torch.tensor([[[1.3, -.4], [0., 0.]]]).double())
    future = torch.arange(1, 13).double()*.4
    torch.testing.assert_close(model(x, v, t), x[:, -1:]+future[None, :, None, None]*speed[:, None])


def test_last2_exact_archived_cv():
    torch.manual_seed(17); x = torch.randn(2, 8, 4, 2).double(); v = torch.rand(2, 8, 4) > .2
    t = torch.arange(8).double()[None].expand(2, -1)*.4
    assert torch.equal(KinematicPredictor('CV-Last2')(x, v, t), HistoricalCV()(x, v, t))


@pytest.mark.parametrize('name', NAMES)
def test_translation_padding_permutation_future_isolation(name):
    x, v, t = line(); model = KinematicPredictor(name); y = model(x, v, t)
    offset = torch.tensor([103., -72.]).double()
    torch.testing.assert_close(model(x+offset, v, t), y+offset)
    perm = [1, 0]; torch.testing.assert_close(model(x[:, :, perm], v[:, :, perm], t), y[:, :, perm])
    z = torch.cat([x, torch.full((1, 8, 1, 2), float('nan')).double()], 2)
    mask = torch.cat([v, torch.zeros(1, 8, 1, dtype=torch.bool)], 2)
    yp = model(z, mask, t); torch.testing.assert_close(yp[:, :, :2], y)
    assert torch.equal(yp[:, :, -1], torch.zeros_like(yp[:, :, -1]))
    b = dict(observed=x, valid=v, timestamps=t, future=torch.randn(1, 12, 2, 2))
    before = (model(b['observed'], b['valid'], b['timestamps']), reliability_features(x, v, t))
    b['future'].fill_(float('nan'))
    assert torch.equal(before[0], model(b['observed'], b['valid'], b['timestamps']))
    assert all(torch.equal(value, reliability_features(x, v, t)[k]) for k, value in before[1].items())


@pytest.mark.parametrize('name', NAMES)
def test_missing_history_current_suffix_only(name):
    x, v, t = line(); v[:, 4] = False
    z = x.clone(); z[:, :4] += 500
    model = KinematicPredictor(name)
    torch.testing.assert_close(model(x, v, t), model(z, v, t))
    v[:, :-1] = False; x[:, :-1] = float('nan')
    torch.testing.assert_close(model(x, v, t), x[:, -1:].expand(-1, 12, -1, -1))
    assert torch.isfinite(model(x[:, -1:], v[:, -1:], t[:, -1:])).all()


def test_ols_uses_real_timestamps_and_last_origin():
    x, v, t = line(); t *= .5
    torch.testing.assert_close(estimated_velocity(x, v, t, 'CV-OLS8')[0, 0], torch.tensor([2.6, -.8]).double())
    # A quadratic path has OLS slope different from last2; the prediction
    # still anchors at actual final x rather than fitted endpoint.
    x[:, :, 0, 0] = (t**2)
    model = KinematicPredictor('CV-OLS8'); speed = estimated_velocity(x, v, t, 'CV-OLS8')
    torch.testing.assert_close(model(x, v, t)[:, 0], x[:, -1]+speed*.4)


def test_component_median_even_and_rotation_limit():
    x = torch.zeros(1, 3, 1, 2).double(); x[0, :, 0, 0] = torch.tensor([0., .4, 1.6])
    torch.testing.assert_close(estimated_velocity(x, name='CV-Median')[0, 0], torch.tensor([2., 0.]).double(), rtol=1e-7, atol=1e-7)
    increments = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]).double()*.4
    x = torch.cat([torch.zeros(1, 2).double(), increments.cumsum(0)]).reshape(1, 4, 1, 2)
    a = torch.tensor(torch.pi/4).double(); R = torch.stack([torch.stack([a.cos(), -a.sin()]), torch.stack([a.sin(), a.cos()])])
    med = estimated_velocity(x, name='CV-Median'); rotated = estimated_velocity(x@R.T, name='CV-Median')
    assert not torch.allclose(rotated, med@R.T)  # Componentwise median is axis-dependent.
    for name in ('CV-Last2', 'CV-OLS4', 'CV-OLS8'):
        torch.testing.assert_close(estimated_velocity(x@R.T, name=name), estimated_velocity(x, name=name)@R.T)


def test_reliability_line_static_outback_and_turn():
    x, v, t = line(); f = reliability_features(x, v, t); g = reliability_groups(f)
    torch.testing.assert_close(f['net_path_ratio'], torch.ones(1, 2).double())
    assert f['velocity_variance_m2ps2'].max() < 1e-25 and f['linear_fit_RMS_m'].max() < 1e-12
    assert g['history_quasi_stationary'][0, 1] and not g['history_quasi_stationary'][0, 0]
    z = torch.zeros(1, 8, 1, 2).double(); z[0, :, 0, 0] = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0]).double()
    ff = reliability_features(z); gg = reliability_groups(ff)
    assert gg['history_out_and_back'].all() and gg['history_direction_changing'].all()
    assert ff['mean_velocity_speed_mps'].item() == 0 and ff['mean_scalar_speed_mps'].item() > 0
    assert not ff['last_mean_angle_known'].any()


def test_terminal_noise_features_and_ols_response():
    x, v, t = line(); noisy = x.clone(); noisy[:, -1, 0, 1] += .8
    f = reliability_features(noisy, v, t); assert reliability_groups(f)['history_terminal_unstable'][0, 0]
    assert (estimated_velocity(noisy, v, t, 'CV-OLS8')-estimated_velocity(x, v, t, 'CV-OLS8')).norm() < (estimated_velocity(noisy, v, t, 'CV-Last2')-estimated_velocity(x, v, t, 'CV-Last2')).norm()


def test_nonfinite_and_invalid_model_rejected():
    x, v, t = line(); x[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError): KinematicPredictor()(x, v, t)
    with pytest.raises(ValueError): KinematicPredictor('CV-tuned')
