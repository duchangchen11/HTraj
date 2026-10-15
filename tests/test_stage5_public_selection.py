"""Regression checks for the bounded public-baseline audit, not new models."""
import json
from pathlib import Path
import pytest
import torch
from scripts.stage5.common import ROOT, COMMITS, CACHE, metrics, check_non_eth_asset
from scripts.stage5.validate_socialvae import module, loader_for


def test_metric_minima_are_independent_and_mean_is_not_first_sample():
    truth = torch.zeros(3, 1, 2)
    pred = torch.zeros(2, 3, 1, 2)
    pred[0, :, 0, 0] = torch.tensor([0., 0., 3.])
    pred[1, :, 0, 0] = torch.tensor([3., 3., 0.])
    ade, fde = metrics(pred, truth)
    assert ade[:, 0].tolist() == [1., 2.]
    assert fde[:, 0].tolist() == [3., 0.]
    assert ade[:, 0].argmin() != fde[:, 0].argmin()
    with pytest.raises(AssertionError):
        metrics(pred[0], truth)


@pytest.mark.parametrize('asset', [
    'models/eth/ckpt-best', 'data/eth/test/biwi_eth.txt',
    'data/hotel/train/biwi_eth.txt', '../eth_ckpt.pth', 'models/univ/ckpt-best'])
def test_asset_boundary_rejects_unregistered_or_eth(asset):
    with pytest.raises(ValueError):
        check_non_eth_asset(asset)


def test_fixed_scope_and_no_extra_functional_candidate():
    d = json.loads((ROOT/'configs/stage5/functional_plan.json').read_text())
    assert d['functional_candidates'] == ['EqMotion', 'SocialVAE']
    assert d['SocialVAE']['samples_per_fold'] == 128
    assert not d['SocialVAE']['FPC']
    assert d['SocialVAE']['training_smoke']['steps'] == 1
    check_non_eth_asset('models/hotel/ckpt-best')


@pytest.fixture
def official():
    source = CACHE/'socialvae'/COMMITS['socialvae']/'social_vae.py'
    if not source.exists():
        pytest.skip('Pinned external source is a local ignored asset; fetch it to run integration checks')
    return module('stage5_test_official_model', 'social_vae.py'), module('stage5_test_official_data', 'data.py')


def example():
    x = torch.zeros(8, 2, 6)
    x[..., 0] = torch.arange(8)[:, None]*.2
    n = x[:, :, None].repeat(3, 1, 2, 1)[:20].clone()
    n[..., :2] += .4
    n[..., 2:4] = .2
    return x, n


def test_official_prediction_padding_neighbor_permutation_future_isolation(official):
    net, _ = official
    model = net.SocialVAE(12, 2, 32).eval()
    x, n = example()
    with torch.no_grad():
        base = model(x, n)
        altered = n.clone()
        altered[8:] = float('nan')
        assert torch.equal(base, model(x, altered))
        torch.testing.assert_close(base, model(x, n.flip(2)))
        torch.testing.assert_close(base, model(x, torch.cat((n, torch.full_like(n[:, :, :1], 1e9)), 2)))
        assert torch.equal(model(x, None), model(x, n[:, :, :0]))
    assert base.shape == (12, 2, 2) and torch.isfinite(base).all()


def test_official_train_backward_finite_and_no_input_jacobian_claim(official):
    net, _ = official
    model = net.SocialVAE(12, 2, 32).train()
    x, n = example()
    y = torch.randn(12, 2, 2)
    losses = model.loss(*model(x, y, n))
    losses['loss'].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0
    # Official enc() constructs physical features under no_grad. Learned-weight
    # gradients are tested, not an input-Jacobian certificate of social dependence.


def test_detect_raw_loader_future_boundary_dependency(official, tmp_path):
    _, data = official
    states = []
    for i, speed in enumerate([.2, 1.]):
        file = tmp_path/(str(i)+'.txt')
        rows = []
        for t in range(20):
            rows.append(f'{t*10} 0 {t*.1} 0\n')
            if t >= 7:
                rows.append(f'{t*10} 1 {1.7+(t-7)*speed} 0\n')
        file.write_text(''.join(rows))
        d, _ = loader_for(data.Dataloader, file, torch.device('cpu'))
        states.append(d.collate_fn([d.data[0]]))
    xa, ya, na = states[0]
    xb, yb, nb = states[1]
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert torch.equal(na[:8, ..., :2], nb[:8, ..., :2])
    assert not torch.equal(na[:8, ..., 2:4], nb[:8, ..., 2:4])


def test_audit_results_do_not_claim_eqmotion_or_paper_reproduction():
    # Lightweight committed records must remain usable without downloaded data.
    eq = json.loads((ROOT/'results/tables/stage5/eqmotion_strict_loading.json').read_text())
    assert eq['unique_ETH_UCY_implementations'] == 1
    assert all(not d['strict_passed'] and not d['forward_executed'] for d in eq['configs'])
    social = json.loads((ROOT/'results/tables/stage5/socialvae_execution.json').read_text())
    assert social['functional_candidates'] == ['EqMotion', 'SocialVAE']
    assert not social['ETH_evaluated'] and not social['formal_training'] and not social['paper_reproduced']
