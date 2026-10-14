from pathlib import Path
import hashlib
import json
import pytest
import torch
from src.models.tohi_a import TOHIA
from src.models.ped_base import PedBase
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS

NEW = ['TOHI-A0', 'TOHI-A-R', 'TOHI-A-N', 'TOHI-A-RN', 'FirstOrder-GNN-RN', 'Dynamic-GNN-RN', 'PedBase-Wide']


def tracks(people=6):
    return torch.randn(2, 1, people, 2)+torch.randn(2, 8, people, 2).mul(.05).cumsum(1)


def test_a0_is_numerically_identical_to_preserved_original():
    original = TOHIA(hidden_dim=16).eval()
    alias = DIAGNOSTIC_MODELS['TOHI-A0'](hidden_dim=16).eval()
    alias.load_state_dict(original.state_dict(), strict=True)
    x = tracks()
    torch.testing.assert_close(original(x), alias(x), rtol=0, atol=0)


def test_a0_backward_matches_preserved_original_on_cpu():
    torch.manual_seed(17)
    original = TOHIA(hidden_dim=16)
    alias = DIAGNOSTIC_MODELS['TOHI-A0'](hidden_dim=16)
    alias.load_state_dict(original.state_dict(), strict=True)
    x = tracks().requires_grad_()
    other = x.detach().clone().requires_grad_()
    original(x).square().mean().backward()
    alias(other).square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    for (name, parameter), (other_name, other_parameter) in zip(
            original.named_parameters(), alias.named_parameters()):
        assert name == other_name
        assert (parameter.grad is None) == (other_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize('name', NEW)
def test_variant_padding_permutation_and_backward(name):
    model = DIAGNOSTIC_MODELS[name](hidden_dim=16).eval()
    x = tracks()
    permutation = torch.tensor([3, 5, 0, 4, 2, 1])
    torch.testing.assert_close(model(x)[:, :, permutation], model(x[:, :, permutation]), rtol=1e-5, atol=2e-6)
    padded = torch.cat([x, torch.full((2, 8, 2, 2), torch.nan)], 2)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    valid[:, :, -2:] = False
    result = model(padded, valid)
    torch.testing.assert_close(result[:, :, :6], model(x))
    assert result[:, :, 6:].eq(0).all()
    result.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('name', ['TOHI-A-R', 'TOHI-A-N', 'TOHI-A-RN', 'FirstOrder-GNN-RN', 'Dynamic-GNN-RN'])
def test_exact_no_neighbor_ego_fallback_and_invalid_norm_bias(name):
    model = DIAGNOSTIC_MODELS[name](hidden_dim=16).eval()
    for normalizer in model.fusion.normalizers:
        normalizer.bias.data.fill_(100)
    base = PedBase(hidden_dim=16).eval()
    base.encoder.load_state_dict(model.encoder.state_dict())
    base.decoder.load_state_dict(model.decoder.state_dict())
    x = tracks(people=1)
    prediction, debug = model(x, return_debug=True)
    torch.testing.assert_close(prediction, base(x), rtol=0, atol=0)
    assert debug['used_order_states'].eq(0).all()
    assert debug['residual_gate'].eq(0).all()


@pytest.mark.parametrize('name', ['TOHI-A-R', 'TOHI-A-N', 'TOHI-A-RN'])
def test_propagation_and_graph_are_unchanged_by_fusion(name):
    torch.manual_seed(17)
    original = TOHIA(hidden_dim=16).eval()
    torch.manual_seed(17)
    variant = DIAGNOSTIC_MODELS[name](hidden_dim=16).eval()
    x = tracks()
    _, a = original(x, return_debug=True)
    _, b = variant(x, return_debug=True)
    torch.testing.assert_close(a['order_states'], b['order_states'], rtol=0, atol=0)
    assert torch.equal(a['order_support'], b['order_support'])
    for old, new in zip(a['graph_frames'], b['graph_frames']):
        assert torch.equal(old.source, new.source) and torch.equal(old.target, new.target)
        torch.testing.assert_close(old.weight, new.weight, rtol=0, atol=0)


@pytest.mark.parametrize('name', ['TOHI-A-R', 'TOHI-A-RN', 'FirstOrder-GNN-RN', 'Dynamic-GNN-RN'])
def test_social_branch_changes_prediction_and_has_nonzero_gradient(name):
    model = DIAGNOSTIC_MODELS[name](hidden_dim=16).eval()
    x = tracks()
    prediction, debug = model(x, return_debug=True)
    observed_gate = debug['residual_gate'][debug['order_support_last'].any(-1)]
    torch.testing.assert_close(observed_gate, torch.full_like(observed_gate, .05))
    assert (debug['fused_context']-debug['hidden'][:, -1]).norm().item() > 1e-7
    disabled = model(x, intervention={'disable_residual': True})
    assert (prediction-disabled).abs().max().item() > 1e-7
    prediction.square().mean().backward()
    assert model.fusion.delta[0].weight.grad.norm().item() > 0
    assert model.fusion.gate.bias.grad.abs().sum().item() > 0


def test_normalization_controls_scale_but_keeps_invalid_zero():
    model = DIAGNOSTIC_MODELS['TOHI-A-RN'](hidden_dim=16)
    ego = torch.randn(1, 2, 16)
    states = torch.randn(1, 2, 3, 16)*torch.tensor([1, 100, 10000])[None, None, :, None]
    support = torch.tensor([[[True, True, False], [True, False, True]]])
    context, beta, debug = model.fusion(ego, states, support)
    norm = debug['used_order_states'].norm(dim=-1)
    torch.testing.assert_close(norm[support], torch.full_like(norm[support], 4.), atol=1e-3, rtol=1e-3)
    assert norm[~support].eq(0).all()
    assert beta[~support].eq(0).all()


def test_ablations_cap_orders_remove_messages_and_degree_preserving_shuffle():
    model = DIAGNOSTIC_MODELS['TOHI-A-RN'](hidden_dim=16).eval()
    x = tracks(people=20)*.5
    _, full = model(x, return_debug=True)
    _, cap = model(x, return_debug=True, intervention={'order_cap': 1})
    torch.testing.assert_close(full['order_states'], cap['order_states'])
    assert cap['order_weights'][:, :, 1:].eq(0).all()
    _, removed = model(x, return_debug=True, intervention={'remove_messages': True})
    assert not removed['order_support'].any()
    torch.testing.assert_close(removed['fused_context'], removed['hidden'][:, -1])
    _, shuffled = model(x, return_debug=True, intervention={'shuffle_graph': True})
    changed = False
    for old, new in zip(full['graph_frames'], shuffled['graph_frames']):
        assert torch.equal(old.target, new.target)
        assert not (new.source == new.target).any()
        assert torch.equal(new.source//20, new.target//20)
        n = 40
        old_keys, new_keys = old.target*n+old.source, new.target*n+new.source
        changed |= bool((~torch.isin(new_keys, old_keys)).any())
    assert changed


def test_fusion_prefix_and_gt_isolation_still_hold():
    model = DIAGNOSTIC_MODELS['TOHI-A-RN'](hidden_dim=16).eval()
    x = tracks().requires_grad_()
    changed = x.detach().clone()
    changed[:, 4:] += 100
    _, original = model(x, return_debug=True)
    _, modified = model(changed, return_debug=True)
    torch.testing.assert_close(original['order_states'][:, :4], modified['order_states'][:, :4])
    grad = torch.autograd.grad(original['order_states'][:, 3].sum(), x)[0]
    assert grad[:, 4:].eq(0).all()
    with pytest.raises(TypeError):
        model(x, future=torch.randn(2, 12, 6, 2))


def test_all_stage2_protected_files_remain_byte_identical():
    protected = json.loads(Path('configs/stage2p5/protected_artifacts.json').read_text())
    for path, expected in protected.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
