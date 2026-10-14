import pytest
import torch
from src.models import MODEL_REGISTRY
from src.models.ego_encoder import motion_inputs
from src.models.order_state import OrderState
from src.models.temporal_graph import GraphFrame, select_candidates
from src.evaluation.metrics import masked_l2_loss, metric_sums


MODELS = list(MODEL_REGISTRY)


def tracks(batch=2, people=4, steps=8):
    initial = torch.randn(batch, 1, people, 2)
    motion = torch.randn(batch, steps, people, 2)*0.08
    return initial+motion.cumsum(1)


@pytest.mark.parametrize('name', MODELS)
@pytest.mark.parametrize('people', [1, 4])
def test_single_and_multi_person_shapes(name, people):
    model = MODEL_REGISTRY[name](hidden_dim=16)
    result = model(tracks(people=people))
    assert result.shape == (2, 12, people, 2)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize('name', MODELS)
def test_padding_ignores_arbitrary_invalid_coordinates(name):
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    x = tracks(batch=1, people=3)
    padded = torch.cat([x, torch.full((1, 8, 3, 2), torch.nan)], 2)
    valid = torch.ones(1, 8, 6, dtype=torch.bool)
    valid[:, :, 3:] = False
    torch.testing.assert_close(model(x), model(padded, valid)[:, :, :3])
    assert model(padded, valid)[:, :, 3:].eq(0).all()


@pytest.mark.parametrize('name', MODELS)
def test_agent_permutation_equivariance(name):
    model = MODEL_REGISTRY[name](hidden_dim=16, max_neighbors=2).eval()
    x = tracks(people=6)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    torch.testing.assert_close(model(x)[:, :, order], model(x[:, :, order]), rtol=1e-5, atol=2e-6)


def test_topk_tie_boundary_has_no_index_tiebreak():
    x, v = torch.zeros(1, 6, 2), torch.zeros(1, 6, 2)
    valid = torch.ones(1, 6, dtype=torch.bool)
    adjacency, _ = select_candidates(x, v, valid, max_neighbors=2)
    assert not adjacency.any()  # Whole indistinguishable boundary group dropped.


@pytest.mark.parametrize('name', MODELS)
def test_no_neighbors_exact_shared_ego_fallback(name):
    x = tracks(batch=1, people=1)
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    base = MODEL_REGISTRY['PedBase'](hidden_dim=16).eval()
    base.encoder.load_state_dict(model.encoder.state_dict())
    base.decoder.load_state_dict(model.decoder.state_dict())
    torch.testing.assert_close(model(x), base(x))
    if name in ('FirstOrder-GNN', 'TOHI-Net-A'):
        _, debug = model(x, return_debug=True)
        assert debug['order_states'].eq(0).all()
        assert not debug['order_support'].any()


def oracle_run(events, people=4, steps=8, zero_projection=False):
    module = OrderState(hidden_dim=1, max_order=3)
    with torch.no_grad():
        for projection in module.projections:
            projection.weight.fill_(0 if zero_projection else 1)
        for gate in module.gates:
            for parameter in gate.parameters():
                parameter.zero_()  # sigmoid(0) = 0.5
    hidden = torch.zeros(1, steps, people, 1, requires_grad=True)
    with torch.no_grad():
        hidden[:, :, 0] = 1
    continuity = torch.ones(1, steps, people, dtype=torch.bool)
    continuity[:, 0] = False
    frames = [GraphFrame.artificial(1, people, events.get(t, [])) for t in range(steps)]
    states, support = module(hidden, frames, continuity)
    return states, support, hidden


def test_first_order_message_and_numeric_value():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1.0)]})
    assert states[0, 1, 1, 0, 0].item() == 1
    assert support[0, 1, 1, 0]
    assert states[:, :, :, 1:].eq(0).all()


def test_second_order_forward_time_path():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1.0)], 3: [(0, 1, 2, 1.0)]})
    assert support[0, 3, 2, 1]
    assert states[0, 3, 2, 1, 0].item() == 0.5  # one waiting frame


def test_reverse_time_blocks_same_second_order_path():
    states, support, _ = oracle_run({1: [(0, 1, 2, 1.0)], 3: [(0, 0, 1, 1.0)]})
    assert not support[0, :, 2, 1].any()
    assert states[0, :, 2, 1].eq(0).all()


def test_same_frame_chain_cannot_read_just_written_order():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1.0), (0, 1, 2, 1.0)]})
    assert not support[0, :, 2, 1].any()
    assert states[0, :, 2, 1].eq(0).all()


def test_third_order_path():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1)], 2: [(0, 1, 2, 1)], 3: [(0, 2, 3, 1)]})
    assert support[0, 3, 3, 2]
    assert states[0, 3, 3, 2, 0].item() == 1


def test_waiting_does_not_increase_social_order():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1)]})
    assert support[0, -1, 1, 0]
    assert states[0, -1, 1, 0, 0].item() == 0.5**6
    assert not support[:, :, :, 1:].any()


def test_topology_exists_even_if_learned_message_is_zero():
    states, support, _ = oracle_run({1: [(0, 0, 1, 1)], 2: [(0, 1, 2, 1)]}, zero_projection=True)
    assert support[0, 2, 2, 1]
    assert states.eq(0).all()


def test_reappearance_resets_ego_and_social_state():
    model = MODEL_REGISTRY['TOHI-Net-A'](hidden_dim=16).eval()
    x = tracks(batch=1, people=2)
    valid = torch.ones(1, 8, 2, dtype=torch.bool)
    valid[:, 3] = False
    _, full = model(x, valid, return_debug=True)
    _, restarted = model(x[:, 4:], valid[:, 4:], return_debug=True)
    torch.testing.assert_close(full['hidden'][:, 4:], restarted['hidden'])
    torch.testing.assert_close(full['order_states'][:, 4:], restarted['order_states'])
    assert not full['order_support'][:, 4].any()


@pytest.mark.parametrize('name', MODELS)
def test_prefix_invariance_and_gradient_support(name):
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    x = tracks(batch=1, people=3).requires_grad_()
    changed = x.detach().clone()
    changed[:, 4:] += torch.randn_like(changed[:, 4:])*100
    _, original = model(x, return_debug=True)
    _, modified = model(changed, return_debug=True)
    torch.testing.assert_close(original['hidden'][:, :4], modified['hidden'][:, :4])
    objective = original['hidden'][:, 3].sum()
    if name in ('FirstOrder-GNN', 'TOHI-Net-A'):
        torch.testing.assert_close(original['order_states'][:, :4], modified['order_states'][:, :4])
        objective = objective+original['order_states'][:, 3].sum()
    if name == 'Dynamic-GNN':
        torch.testing.assert_close(original['social_states'][:, :4], modified['social_states'][:, :4])
        objective = objective+original['social_states'][:, 3].sum()
    gradient = torch.autograd.grad(objective, x)[0]
    assert gradient[:, 4:].eq(0).all()
    if 'graph_frames' in original:
        for a, b in zip(original['graph_frames'][:4], modified['graph_frames'][:4]):
            assert torch.equal(a.source, b.source) and torch.equal(a.target, b.target)
            torch.testing.assert_close(a.weight, b.weight)


def test_pedbase_never_reads_neighbor_hidden_states():
    model = MODEL_REGISTRY['PedBase'](hidden_dim=16).eval()
    x = tracks(batch=1, people=3)
    changed = x.clone()
    changed[:, :, 1:] += 100
    torch.testing.assert_close(model(x)[:, :, 0], model(changed)[:, :, 0])


@pytest.mark.parametrize('name', MODELS)
def test_future_gt_not_in_model_signature_or_prediction(name):
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    batch = dict(observed=tracks(), future=torch.randn(2, 12, 4, 2))
    prediction = model(batch['observed'])
    batch['future'] *= 1000
    torch.testing.assert_close(prediction, model(batch['observed']))
    with pytest.raises(TypeError):
        model(batch['observed'], future=batch['future'])


@pytest.mark.parametrize('name', MODELS)
def test_finite_loss_and_backward(name):
    model = MODEL_REGISTRY[name](hidden_dim=16)
    x = tracks(people=5)
    predicted = model(x)
    valid = torch.ones(2, 12, 5, dtype=torch.bool)
    loss = masked_l2_loss(predicted, torch.randn_like(predicted), valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.gru.weight_hh.grad is not None


@pytest.mark.parametrize('name', MODELS)
def test_numerical_stability_coincident_and_large_coordinates(name):
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    for x in [torch.zeros(2, 8, 5, 2), tracks(people=5)*1000]:
        assert torch.isfinite(model(x)).all()
    x = tracks()
    x[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        model(x)


@pytest.mark.parametrize('name', MODELS)
def test_scene_and_call_state_isolation(name):
    model = MODEL_REGISTRY[name](hidden_dim=16).eval()
    x = tracks(batch=2, people=3)
    together = model(x)
    torch.testing.assert_close(together, torch.cat([model(x[:1]), model(x[1:])]), rtol=1e-5, atol=2e-6)
    model(tracks(batch=1, people=7))
    torch.testing.assert_close(together, model(x))


def test_metrics_use_separate_masks_and_final_horizon():
    prediction = torch.zeros(1, 3, 2, 2)
    target = torch.zeros_like(prediction)
    target[..., 0] = torch.tensor([[[1., 2.], [1., 4.], [1., 100.]]])
    valid = torch.tensor([[[True, True], [True, True], [True, False]]])
    sums = metric_sums(prediction, target, valid)
    assert sums['ade_sum'].item() == 4  # track means 1 and 3
    assert sums['ade_count'].item() == 2
    assert sums['fde_sum'].item() == 1 and sums['fde_count'].item() == 1
    assert masked_l2_loss(prediction, target, valid).item() == 2


def test_real_timestamps_and_long_gap_reset():
    x = torch.zeros(1, 3, 1, 2)
    x[:, :, :, 0] = torch.tensor([0., .4, 1.2])[None, :, None]
    encoded = motion_inputs(x, timestamps=torch.tensor([[0., .4, 1.2]], dtype=torch.float64))
    assert encoded['velocity'][0, 1, 0, 0].item() == 1
    assert not encoded['continuity'][0, 2, 0]
    assert encoded['velocity'][0, 2, 0].eq(0).all()
