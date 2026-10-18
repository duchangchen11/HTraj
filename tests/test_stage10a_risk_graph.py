"""Stage10A engineering properties only; no measured forecasting accuracy."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from src.risk_dynamic_graph import GraphConfig, RiskGraphModel, build_graph
from src.risk_dynamic_graph.interaction_features import historical_motion
from src.risk_dynamic_graph.risk_estimator import closest_approach
from src.risk_dynamic_graph.data import inference_inputs
from src.data.preprocessing import Recording

C = GraphConfig(radius=10., max_neighbors=2, distance_scale=1.)


def data(n=5, batch=1):
    rng = torch.Generator().manual_seed(101)
    p = torch.randn(batch, 8, n, 2, generator=rng, dtype=torch.float64).cumsum(1) * .05
    p = p + torch.arange(n, dtype=p.dtype)[None, None, :, None] * .3
    m = torch.ones(batch, 8, n, dtype=torch.bool)
    ids = (torch.arange(n, dtype=torch.int64) * 7 + 3)[None].expand(batch, -1)
    return p, m, ids


@pytest.mark.parametrize('mode', ['E', 'S', 'D', 'R'])
@pytest.mark.parametrize('n', [1, 2, 5, 12])
def test_permutation_value_gradient_translation(mode, n):
    p, mask, ids = data(n)
    model = RiskGraphModel(mode, 16, C).double()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(32))
    p.requires_grad_(True)
    a = model(p, mask, ids)
    grad = torch.autograd.grad(a.square().sum(), p)[0]
    q = p.detach()[:, :, perm].clone().requires_grad_(True)
    b = model(q, mask[:, :, perm], ids[:, perm])
    gradq = torch.autograd.grad(b.square().sum(), q)[0]
    torch.testing.assert_close(a[:, perm], b, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grad[:, :, perm], gradq, rtol=1e-9, atol=1e-9)
    shift = torch.tensor([8., -11.], dtype=p.dtype)
    torch.testing.assert_close(model(p + shift, mask, ids), a + shift, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('mode', ['E', 'S', 'D', 'R'])
def test_padding_missing_reset_no_nan(mode):
    p, mask, ids = data(4)
    mask[:, :3, 1] = False
    mask[:, 3:5, 2] = False
    p[~mask] = float('nan')
    model = RiskGraphModel(mode, 16, C).double()
    out = model(p, mask, ids, diagnostics=True)
    padded = torch.cat([p, torch.full((1, 8, 2, 2), float('nan'), dtype=p.dtype)], 2)
    pm = torch.cat([mask, torch.zeros(1, 8, 2, dtype=torch.bool)], 2)
    pi = torch.cat([ids, torch.tensor([[999, 1001]])], 1)
    augmented = model(padded, pm, pi)
    torch.testing.assert_close(out['prediction'], augmented[:, :4], atol=1e-10, rtol=1e-10)
    assert torch.equal(augmented[:, 4:], torch.zeros_like(augmented[:, 4:]))
    assert not out['graph']['velocity_valid'][0, 3, 1]
    assert out['graph']['velocity_valid'][0, 4, 1]
    assert not out['graph']['velocity_valid'][0, 5, 2]
    for t, actor in ((3, 1), (5, 2)):
        inp = torch.tensor([[0., 0., 0.]], dtype=p.dtype)
        h = model.encoder(inp, torch.zeros(1, 16, dtype=p.dtype))
        torch.testing.assert_close(out['ego_trace'][:, t, actor], h)
    assert torch.isfinite(augmented).all()
    out['prediction'].square().mean().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


@pytest.mark.parametrize('mode', ['S', 'D', 'R'])
def test_causal_prefix(mode):
    p, mask, ids = data(6)
    q = p.clone()
    q[:, 4:] += 40.
    m = RiskGraphModel(mode, 16, C).double()
    g = build_graph(p, mask, ids, mode, C)
    gp = build_graph(p[:, :4], mask[:, :4], ids, mode, C)
    changed = build_graph(q, mask, ids, mode, C)
    for key in ('candidate_mask', 'edge_mask', 'risk', 'features', 't_cpa', 'dca'):
        torch.testing.assert_close(g[key][:, :4], gp[key], atol=0., rtol=0.)
        torch.testing.assert_close(g[key][:, :4], changed[key][:, :4], atol=0., rtol=0.)
    a = m(p, mask, ids, diagnostics=True)['ego_trace']
    b = m(q, mask, ids, diagnostics=True)['ego_trace']
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0., rtol=0.)


def test_cpa_numpy_independent_and_extremes():
    rng = np.random.default_rng(7)
    r = rng.normal(size=(100, 2))
    v = rng.normal(size=(100, 2))
    v[:2] = [0., 1e-12]
    known = np.ones(100, dtype=bool)
    known[2:5] = False
    g = closest_approach(torch.tensor(r), torch.tensor(v), torch.tensor(known))
    expected = []
    for ri, vi, k in zip(r, v, known):
        speed2 = float(np.dot(vi, vi))
        time = float(np.clip(-np.dot(ri, vi) / speed2, 0., 4.8)) if k and speed2 > 1e-8 else 0.
        d = float(np.linalg.norm(ri + time * vi))
        expected.append([time, d, np.exp(-.5 * d*d - time/4.8)])
    np.testing.assert_allclose(np.stack([g[k].numpy() for k in ('t_cpa','dca','risk')], 1), expected, atol=1e-12)
    assert all(torch.isfinite(g[k]).all() for k in ('t_cpa','dca','risk'))


def crossing():
    # Receiver at 0; nearest source recedes to x=1. Farther source approaches
    # from x=3 at -1m/s. R chooses future near approach; D chooses current near.
    t = torch.arange(8, dtype=torch.float64) * .4
    p = torch.zeros(1, 8, 3, 2, dtype=torch.float64)
    p[0, :, 1, 0] = 1. + (t - t[-1]) * .1
    p[0, :, 2, 0] = 3. - (t - t[-1])
    return p, torch.ones(1, 8, 3, dtype=torch.bool), torch.tensor([[10, 20, 30]])


def test_r_d_distinct_direction_and_determinism():
    p, mask, ids = crossing()
    conf = replace(C, max_neighbors=1, distance_scale=.5)
    d, r = [build_graph(p, mask, ids, mode, conf) for mode in ('D', 'R')]
    assert d['edge_mask'][0, -1, 0].tolist() == [False, True, False]
    assert r['edge_mask'][0, -1, 0].tolist() == [False, False, True]
    assert abs(float(r['t_cpa'][0, -1, 0, 2]) - 3.) < 1e-10
    assert float(r['dca'][0, -1, 0, 2]) < 1e-10
    perm = torch.tensor([2, 0, 1])
    shuffled = build_graph(p[:, :, perm], mask[:, :, perm], ids[:, perm], 'R', conf)
    assert torch.equal(r['edge_mask'][:, :, perm][:, :, :, perm], shuffled['edge_mask'])
    assert torch.equal(r['edge_mask'], build_graph(p, mask, ids, 'R', conf)['edge_mask'])
    parallel = torch.arange(8, dtype=p.dtype)[None, :, None, None] * .4
    p[:, :, :, 0:1] = parallel + torch.tensor([0., 1., 3.])[None, None, :, None]
    g = build_graph(p, mask, ids, 'R', conf)
    assert not g['approaching'][0, -1].any()
    assert g['t_cpa'][0, -1].eq(0).all()


@pytest.mark.parametrize('mode', ['S', 'D', 'R'])
def test_ties_stable_identity(mode):
    p, mask, ids = data(5)
    p[:] = 0.
    ids = torch.tensor([[100, 40, 20, 90, 70]])
    conf = replace(C, max_neighbors=1)
    g = build_graph(p, mask, ids, mode, conf)['edge_mask']
    assert ids[0, g[0, -1, 0]].tolist() == [20]
    perm = torch.tensor([4, 1, 0, 3, 2])
    gp = build_graph(p[:, :, perm], mask[:, :, perm], ids[:, perm], mode, conf)['edge_mask']
    assert torch.equal(g[:, :, perm][:, :, :, perm], gp)


def test_static_first_encounter_frozen_and_dynamic_updates():
    p, mask, ids = crossing()
    conf = replace(C, radius=10., max_neighbors=1)
    p[0, -1, 1, 0] = 9.
    p[0, -1, 2, 0] = .1
    s, d = [build_graph(p, mask, ids, mode, conf) for mode in ('S', 'D')]
    assert s['edge_mask'][0, :, 0, 1].all()
    assert d['edge_mask'][0, -1, 0, 2]
    mask[0, -1, 1] = False
    s = build_graph(p, mask, ids, 'S', conf)
    assert not s['edge_mask'][0, -1, 0].any()  # don't refill static slots


def test_no_message_returns_ego_and_neighbor_dependency():
    p, mask, ids = data(3)
    e = RiskGraphModel('E', 16, C).double()
    d = RiskGraphModel('D', 16, C).double()
    d.load_state_dict(e.state_dict(), strict=True)
    small = replace(C, max_neighbors=0)
    d.config = small
    torch.testing.assert_close(e(p, mask, ids), d(p, mask, ids), atol=0., rtol=0.)
    p.requires_grad_(True)
    assert torch.autograd.grad(e(p, mask, ids)[0, 0].sum(), p)[0][:, :, 1:].eq(0).all()
    d.config = C
    grad = torch.autograd.grad(d(p, mask, ids)[0, 0].sum(), p)[0]
    assert grad[:, :, 1:].abs().max() > 0


def raw_recording():
    by_frame = {}
    for t in range(20):
        by_frame[10*t] = {1: np.array([.2*t, 0.], np.float32)}
        if 3 <= t <= 9:
            by_frame[10*t][2] = np.array([1., .1*t], np.float32)
    return Recording('synthetic', 'HOTEL', np.arange(20)*10, by_frame)


def test_raw_future_counterfactual_not_tensor_only():
    rec = raw_recording()
    inp = inference_inputs(rec, 0)
    changed = copy.deepcopy(rec)
    for f in range(80, 200, 10):
        changed.by_frame[f] = {1: np.array([1000., -300.], np.float32),
                               999: np.array([0., 0.], np.float32)}
    # Removes neighbor future entirely, changes target future, adds future-only ID.
    other = inference_inputs(changed, 0)
    for key in inp:
        assert torch.equal(inp[key], other[key])
    for mode in ('S', 'D', 'R'):
        m = RiskGraphModel(mode, 16, C).double()
        a = {**inp, 'position': inp['position'].double()}
        b = {**other, 'position': other['position'].double()}
        assert torch.equal(m(**a), m(**b))
        assert torch.equal(build_graph(**a, mode=mode, config=C)['edge_mask'],
                           build_graph(**b, mode=mode, config=C)['edge_mask'])


def test_query_guard_history_only():
    class Guard(dict):
        def get(self, key, default=None):
            assert key in range(0, 80, 10)
            return super().get(key, default)
    r = raw_recording()
    r.by_frame = Guard(r.by_frame)
    inference_inputs(r, 0)


def test_invalid_inputs_rejected():
    p, mask, ids = data(3)
    with pytest.raises(ValueError):
        build_graph(p, mask, ids * 0)
    p[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        historical_motion(p, mask)
