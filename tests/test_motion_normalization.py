import pytest
import torch
from src.models.ped_base import PedBase
from src.models.frozen_ego import CVResidual
from src.models.ego_encoder import motion_inputs
from src.models.motion_normalization import (TranslationNormalizedPedBase, HeadingNormalized,
                                             heading_frame, to_local, to_global)


def example():
    torch.manual_seed(17)
    x = torch.randn(2, 8, 3, 2).double().cumsum(1)
    valid = torch.ones(x.shape[:-1], dtype=torch.bool)
    ts = torch.arange(8).double()[None].expand(2, -1)*.4
    return x, valid, ts


def rotate(x, theta=.731):
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    R = torch.tensor([[c, -s], [s, c]], dtype=x.dtype)
    # Use a double-precision orthogonal matrix for stringent invariance tests.
    a = x.new_tensor(theta); R = torch.stack([torch.stack([a.cos(), -a.sin()]), torch.stack([a.sin(), a.cos()])])
    return x@R.T


@pytest.mark.parametrize('kind', ['M0', 'M2', 'M3'])
def test_translation_equivariance(kind):
    x, v, ts = example()
    model = (PedBase() if kind == 'M0' else HeadingNormalized(cv_residual=kind == 'M3')).double()
    shift = torch.tensor([103., -71.], dtype=x.dtype)
    torch.testing.assert_close(model(x+shift, v, ts), model(x, v, ts)+shift, rtol=1e-10, atol=1e-10)


def test_M1_is_not_new_ablation():
    assert TranslationNormalizedPedBase is PedBase
    x, v, ts = example()
    e = motion_inputs(x, v, ts); c = motion_inputs(x-x[:, -1:], v, ts)
    torch.testing.assert_close(e['motion_features'], c['motion_features'], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('cv', [False, True])
def test_rotation_equivariance_reliable_heading(cv):
    x, v, ts = example(); model = HeadingNormalized(cv_residual=cv).double()
    assert heading_frame(x, v, ts)['reliable'].all()
    torch.testing.assert_close(model(rotate(x), v, ts), rotate(model(x, v, ts)), rtol=1e-10, atol=1e-10)


def test_roundtrip_global_meter_coordinates():
    x, v, ts = example(); f = heading_frame(x, v, ts)
    torch.testing.assert_close(to_global(to_local(x-f['origin'][:, None], f['unit']), f['unit'])+f['origin'][:, None], x)
    model = HeadingNormalized().double()
    for p in model.parameters(): p.data.zero_()
    torch.testing.assert_close(model(x, v, ts), x[:, -1:].expand(-1, 12, -1, -1))


def test_no_velocity_and_near_zero_fallback():
    x, v, ts = example(); x[:] = 3.; v[:, :-1] = False
    x[:, :-1] = float('nan')
    f = heading_frame(x, v, ts)
    assert not f['reliable'].any() and not f['duration'].any()
    assert torch.equal(f['unit'], torch.tensor([1., 0.]).double().expand_as(f['unit']))
    y = HeadingNormalized(cv_residual=True).double()(x, v, ts)
    assert torch.isfinite(y).all()
    x, v, ts = example(); x[:] = 2.; x[:, :, :, 0] += torch.arange(8).double()[None, :, None]*1e-12
    assert not heading_frame(x, v, ts)['reliable'].any()
    assert torch.isfinite(HeadingNormalized().double()(x, v, ts)).all()


def test_fallback_rotation_limit_is_explicit():
    x, v, ts = example(); x[:] = 0.
    model = HeadingNormalized().double()
    for p in model.parameters(): p.data.zero_()
    model.decoder.head[-1].bias.data[::2] = 1.
    # Identity fallback gives global-axis corrections: no false equivariance claim.
    assert not torch.allclose(model(rotate(x), v, ts), rotate(model(x, v, ts)))


def test_gap_resets_frame_and_encoder():
    x, v, ts = example(); v[:, 4] = False
    f = heading_frame(x, v, ts)
    torch.testing.assert_close(f['mean_velocity'], (x[:, -1]-x[:, 5])/.8)
    z = x.clone(); z[:, :4] += 300
    model = HeadingNormalized().double()
    torch.testing.assert_close(model(x, v, ts), model(z, v, ts))


def test_timestamp_velocity_units_and_long_gap():
    x = torch.zeros(1, 8, 1, 2).double(); x[0, :, 0, 0] = torch.arange(8).double()*.2
    ts = torch.arange(8).double()[None]*.2
    f = heading_frame(x, timestamps=ts)
    torch.testing.assert_close(f['mean_velocity'], torch.tensor([[[1., 0.]]]).double())
    ts[0, -1] = 3.
    assert not heading_frame(x, timestamps=ts)['reliable'].any()


@pytest.mark.parametrize('cv', [False, True])
def test_padding_permutation_and_future_isolation(cv):
    x, v, ts = example(); v[:, :, -1] = False; x[:, :, -1] = float('nan')
    model = HeadingNormalized(cv_residual=cv).double(); y = model(x, v, ts)
    assert torch.equal(y[:, :, -1], torch.zeros_like(y[:, :, -1]))
    perm = [2, 0, 1]; torch.testing.assert_close(model(x[:, :, perm], v[:, :, perm], ts), y[:, :, perm])
    batch = dict(observed=x, observed_valid=v, timestamps=ts, future=torch.randn(2, 12, 3, 2))
    a = model(batch['observed'], batch['observed_valid'], batch['timestamps'])
    batch['future'].fill_(float('nan'))
    assert torch.equal(a, model(batch['observed'], batch['observed_valid'], batch['timestamps']))


@pytest.mark.parametrize('cv', [False, True])
def test_gradients_parameters_initialization_finite(cv):
    x, v, ts = example(); torch.manual_seed(17); model = HeadingNormalized(cv_residual=cv).double()
    torch.manual_seed(17); ref = PedBase().double()
    assert sum(p.numel() for p in model.parameters()) == 35352
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), ref.parameters()))
    y = model(x, v, ts); y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_heading_unit_length_threshold_boundary():
    x = torch.zeros(1, 8, 1, 2).double(); x[0, :, 0, 0] = torch.arange(8).double()*.4*.2
    a = heading_frame(x, threshold=.199); b = heading_frame(x, threshold=.201)
    assert a['reliable'].all() and not b['reliable'].any()
    assert torch.equal(a['unit'], b['unit'])  # +x direction equals identity.
    with pytest.raises(ValueError): heading_frame(x, threshold=0)


def test_evaluation_rows_feature_group_names_do_not_collide():
    from src.training.motion_generalization import attach_groups, evaluate
    from src.models.relation_routing import HistoricalCV
    x, v, ts = example()
    b = dict(observed=x, observed_valid=v, timestamps=ts,
             future=x[:, -1:].expand(-1, 12, -1, -1).clone(),
             future_valid=torch.ones(2, 12, 3, dtype=torch.bool),
             recording=['toy', 'toy'], sample_id=['a', 'b'],
             agent_ids=torch.arange(3)[None].expand(2, -1))
    attach_groups(b)
    assert not set(b['history_features']) & set(b['groups'])
    result = evaluate(HistoricalCV(), [b], torch.device('cpu'), rows=True)
    assert len(result['targets']) == 6
    assert all('heading_reliable_flag' in r and 'heading_reliable' in r for r in result['targets'])
