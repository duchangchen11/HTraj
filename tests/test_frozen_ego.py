import copy
import pytest
import torch
from src.models.ped_base import PedBase
from src.models.frozen_ego import CVResidual, FrozenEgoResidual, random_candidates
from src.models.ego_encoder import motion_inputs
from src.models.temporal_graph import select_candidates
from src.evaluation.frozen_ego import history_groups


def scene(people=3, batch=1):
    origin = torch.arange(people).float()[None, None, :, None].expand(batch, 8, people, 2)*.5
    time = torch.arange(8).float()[None, :, None, None]
    observed = origin + time*torch.tensor([.1, .02])
    valid = torch.ones(batch, 8, people, dtype=torch.bool)
    return observed, valid


def nonzero_branch(model):
    with torch.no_grad():
        model.delta[-1].weight.fill_(.03)
        model.delta[-1].bias.fill_(.2)


def test_cv_residual_exact_parameter_and_initialization_match():
    torch.manual_seed(17); base = PedBase()
    torch.manual_seed(17); residual = CVResidual()
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in residual.parameters()) == 35352
    for k, v in base.state_dict().items():
        assert torch.equal(v, residual.state_dict()[k])
    observed, valid = scene()
    p, e = base(observed, valid, return_debug=True)
    r, debug = residual(observed, valid, return_debug=True)
    torch.testing.assert_close(r, debug['cv_prediction'] + (p-e['positions'][:, -1, None]))


def test_cv_residual_missing_velocity_is_explicit_stationary_fallback():
    observed, valid = scene(); valid[:, :-1, 0] = False
    _, debug = CVResidual()(observed, valid, return_debug=True)
    assert not debug['continuity'][0, -1, 0]
    assert debug['motion_features'][0, -1, 0, 4] == 1  # visible
    assert debug['motion_features'][0, -1, 0, 5] == 0  # speed unknown
    assert torch.equal(debug['cv_prediction'][0, :, 0], observed[0, -1, 0].expand(12, 2))


@pytest.mark.parametrize('mode', ['Social-Previous', 'Social-Current', 'Social-Random'])
def test_unsupported_social_output_equals_same_frozen_anchor_exactly(mode):
    observed, valid = scene(); observed[:, :, 2] += 50
    model = FrozenEgoResidual(CVResidual(), mode); nonzero_branch(model)
    prediction, debug = model(observed, valid, return_debug=True)
    assert not debug['has_message'][0, 2] and debug['has_message'][0, :2].all()
    assert torch.equal(prediction[:, :, 2], debug['base_prediction'][:, :, 2])
    removed, d = model(observed, valid, return_debug=True, remove_messages=True)
    assert torch.equal(removed, d['base_prediction'])
    assert torch.count_nonzero(d['correction']) == 0


@pytest.mark.parametrize('mode', ['Ego-Residual', 'Social-Previous', 'Social-Current', 'Social-Random'])
def test_optimizer_cannot_change_anchor_encoder_decoder_or_mode(mode):
    observed, valid = scene(); model = FrozenEgoResidual(CVResidual(), mode)
    before = copy.deepcopy(model.anchor.state_dict()); model.train()
    assert not model.anchor.training and all(not p.requires_grad for p in model.anchor.parameters())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    for _ in range(2):
        optimizer.zero_grad(); prediction = model(observed, valid); prediction.square().mean().backward(); optimizer.step()
    assert all(p.grad is None for p in model.anchor.parameters())
    assert all(torch.equal(v, model.anchor.state_dict()[k]) for k, v in before.items())


def test_same_capacity_same_zero_initial_output_for_all_residual_information_controls():
    observed, valid = scene()
    counts = []
    anchor = CVResidual()
    for mode in FrozenEgoResidual.MODES[1:]:
        torch.manual_seed(17); model = FrozenEgoResidual(copy.deepcopy(anchor), mode)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        prediction, d = model(observed, valid, return_debug=True)
        assert torch.equal(prediction, d['base_prediction'])
    assert counts == [14425]*4


def test_previous_current_use_expected_strictly_observed_prefix_payload():
    observed, valid = scene(2)
    encoded = motion_inputs(observed, valid)
    encoded['hidden'] = torch.zeros(1, 8, 2, 64)
    encoded['hidden'][..., 0] = torch.arange(8)[None, :, None]
    base = torch.zeros(1, 12, 2, 2)
    means = []
    for mode in ['Social-Previous', 'Social-Current']:
        model = FrozenEgoResidual(PedBase(), mode)
        with torch.no_grad():
            model.payload[0].weight.zero_(); model.payload[0].bias.zero_(); model.payload[0].weight[0, 0] = 1
        _, d = model.forward_encoded(encoded, base, return_debug=True)
        means.append(d['pooled_message'][0, 0, 0])
    assert torch.equal(means[0], torch.tensor(3.)) and torch.equal(means[1], torch.tensor(4.))
    model = FrozenEgoResidual(PedBase(), 'Social-Current')
    _, a = model.encode_anchor(observed, valid)
    later = observed.clone(); later[:, 5:] += 100
    _, b = model.encode_anchor(later, valid)
    assert torch.equal(a['hidden'][:, :5], b['hidden'][:, :5])


def test_legitimate_random_graph_preserves_scene_radius_budget_and_no_self():
    observed, valid = scene(14, 2)
    observed *= .1
    observed[:, :, 13] += 50
    e = motion_inputs(observed, valid)
    real, _ = select_candidates(e['positions'][:, 7], e['velocity'][:, 7], e['continuity'][:, 7])
    random = random_candidates(e, 7, real)
    assert torch.equal(real.sum(-1), random.sum(-1))
    assert (random & ~real).any()
    assert not random.diagonal(dim1=-2, dim2=-1).any()
    distance = (observed[:, 7, :, None]-observed[:, 7, None, :]).norm(dim=-1)
    assert (distance[random] <= 4).all() and not random[:, :, 13].any()
    e['continuity'][:, 7, 12] = False
    real, _ = select_candidates(e['positions'][:, 7], e['velocity'][:, 7], e['continuity'][:, 7])
    random = random_candidates(e, 7, real)
    assert not random[:, :, 12].any() and not random[:, 12].any()


@pytest.mark.parametrize('mode', ['Ego-Residual', 'Social-Previous', 'Social-Current', 'Social-Random'])
def test_padding_permutation_and_cached_forward_agree(mode):
    observed, valid = scene(11); observed *= .1
    model = FrozenEgoResidual(CVResidual(), mode); nonzero_branch(model); model.eval()
    original = model(observed, valid)
    permutation = torch.randperm(11)
    torch.testing.assert_close(model(observed[:, :, permutation], valid[:, :, permutation]), original[:, :, permutation], atol=1e-6, rtol=1e-6)
    padded = torch.cat([observed, torch.full((1, 8, 2, 2), float('nan'))], 2)
    mask = torch.cat([valid, torch.zeros(1, 8, 2, dtype=torch.bool)], 2)
    p = model(padded, mask)
    torch.testing.assert_close(p[:, :, :11], original, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(p[:, :, 11:]) == 0
    base, e = model.encode_anchor(observed, valid)
    cached = dict(observed=observed, observed_valid=valid, timestamps=torch.arange(8, dtype=torch.float64)[None]*.4, anchor_prediction=base, anchor_hidden=e['hidden'], future=torch.randn(1, 12, 11, 2))
    torch.testing.assert_close(model.forward_cached(cached), original, atol=1e-6, rtol=1e-6)
    cached['future'] *= 100
    assert torch.equal(model.forward_cached(cached), model.forward_cached(dict(cached, future=torch.zeros_like(cached['future']))))


def test_history_groups_invariant_to_future_ground_truth():
    observed, valid = scene()
    batch = dict(observed=observed, observed_valid=valid, timestamps=torch.arange(8, dtype=torch.float64)[None]*.4, future=torch.randn(1, 12, 3, 2))
    a, _ = history_groups(batch)
    b, _ = history_groups(dict(batch, future=batch['future']*100))
    assert all(torch.equal(a[g], b[g]) for g in a)
    assert a['history_CV_like'].all() and not a['history_velocity_changing'].any()


def test_cv_output_anchor_does_not_fall_back_to_neural_decoder_predictions():
    from src.models.relation_routing import HistoricalCV
    observed, valid = scene(); observed[:, :, 2] += 50
    anchor = PedBase()
    with torch.no_grad():
        anchor.decoder.head[-1].bias.fill_(100)
    model = FrozenEgoResidual(anchor, 'Social-Current', cv_output=True)
    nonzero_branch(model)
    prediction, debug = model(observed, valid, return_debug=True)
    cv = HistoricalCV()(observed, valid)
    assert torch.equal(debug['base_prediction'], cv)
    assert torch.equal(prediction[:, :, 2], cv[:, :, 2])
    assert not torch.equal(anchor(observed, valid), cv)
