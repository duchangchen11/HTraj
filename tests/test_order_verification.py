import pytest
import torch
from src.models.order_state import OrderState
from src.models.social_diagnostics_models import TOHIResidual
from src.models.order_verification import MeasuredOrderState, MODEL_SPECS, build_order_model
from src.models.temporal_graph import GraphFrame
from src.evaluation.order_verification import verification_groups, VERIFICATION_GROUPS, extract_measured_debug


def oracle(events, mode, people=4, steps=5):
    original = OrderState(hidden_dim=1, max_order=3)
    with torch.no_grad():
        for projection in original.projections:
            projection.weight.fill_(1)
        for gate in original.gates:
            for p in gate.parameters():
                p.zero_()
    model = MeasuredOrderState(original, 3, mode)
    hidden = torch.zeros(1, steps, people, 1, requires_grad=True)
    with torch.no_grad():
        hidden[:, :, 0] = 1
    continuity = torch.ones(1, steps, people, dtype=torch.bool)
    continuity[:, 0] = False
    frames = [GraphFrame.artificial(1, people, events.get(t, [])) for t in range(steps)]
    return (*model(hidden, frames, continuity, return_parts=True), hidden)


@pytest.mark.parametrize('mode', ['strict', 'nondecreasing'])
def test_positive_and_reverse_temporal_chains(mode):
    state, support, parts, h = oracle({1:[(0,0,1,1)],3:[(0,1,2,1)]}, mode)
    assert support[0,3,2,1] and state[0,3,2,1,0] == .5
    gradient = torch.autograd.grad(state[0,3,2,1].sum(), h)[0]
    assert gradient[0,0,0,0] == .5 and gradient[:,3:].eq(0).all()
    reverse, mask, _, _ = oracle({1:[(0,1,2,1)],3:[(0,0,1,1)]}, mode)
    assert not mask[:,:,2,1].any() and reverse[:,:,2,1].eq(0).all()


def test_same_frame_is_only_allowed_in_nondecreasing_mode():
    events={1:[(0,0,1,1),(0,1,2,1),(0,2,3,1)]}
    strict, strict_mask, _, _ = oracle(events,'strict')
    relaxed, relaxed_mask, _, h = oracle(events,'nondecreasing')
    assert not strict_mask[:,:,:,1:].any()
    assert relaxed_mask[0,1,2,1] and relaxed[0,1,2,1,0] == 1
    assert relaxed_mask[0,1,3,2] and relaxed[0,1,3,2,0] == 1
    gradient=torch.autograd.grad(relaxed[0,1,3,2].sum(), h)[0]
    assert gradient[0,0,0,0] == 1 and gradient[:,1:].eq(0).all()


@pytest.mark.parametrize('mode', ['strict','nondecreasing'])
def test_waiting_does_not_create_extra_hops_and_multiple_paths_sum(mode):
    only, mask, _, _ = oracle({1:[(0,0,1,1)]},mode)
    assert not mask[:,:,:,1:].any() and only[0,-1,1,0,0] == .5**3
    multiple, support, _, _ = oracle({1:[(0,0,1,1),(0,0,2,1)],3:[(0,1,3,1),(0,2,3,1)]},mode)
    assert multiple[0,3,3,1,0] == 1 and support[0,3,3,1]


def test_strict_instrumentation_is_exact_original_forward_and_backward():
    torch.manual_seed(17); original=TOHIResidual(hidden_dim=16)
    torch.manual_seed(17); measured=build_order_model('TOHI-H3',hidden_dim=16)
    a=torch.randn(2,8,6,2).cumsum(1)*.02
    b=a.clone()
    torch.testing.assert_close(original(a),measured(b),rtol=0,atol=0)
    original(a).square().mean().backward(); measured(b).square().mean().backward()
    for (name,p),(other_name,other_p) in zip(original.named_parameters(),measured.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (other_p.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,other_p.grad,rtol=0,atol=0)


def test_all_common_parameters_start_identically_after_actual_truncation():
    models={}
    for name in MODEL_SPECS:
        torch.manual_seed(17); models[name]=build_order_model(name)
    reference=models['TOHI-H3'].state_dict()
    for name,model in models.items():
        assert model.encoder.hidden_dim == 64
        assert len(model.order_state.projections) == MODEL_SPECS[name]['order']
        assert len(model.fusion.scores) == MODEL_SPECS[name]['order']
        for key,value in model.state_dict().items():
            if '-Cap' in name and key.startswith('fusion.delta.'):
                continue
            torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
    assert abs(sum(p.numel() for p in models['TOHI-H1-Cap'].parameters())-115997) <= 2
    assert abs(sum(p.numel() for p in models['TOHI-H2-Cap'].parameters())-115997) <= 63


def test_nondecreasing_h1_has_identical_dependency_as_strict_h1():
    torch.manual_seed(17); a=build_order_model('TOHI-H1',hidden_dim=16)
    torch.manual_seed(17); b=build_order_model('TOHI-H1',hidden_dim=16)
    b.order_state.mode='nondecreasing'
    x=torch.randn(1,8,4,2)*.1
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)


@pytest.mark.parametrize('name', list(MODEL_SPECS))
def test_model_masks_prefix_permutation_and_finite_gradients(name):
    model=build_order_model(name,hidden_dim=16)
    x=torch.randn(1,8,5,2).cumsum(1)*.02
    pred,debug=model(x,return_debug=True)
    changed=x.clone(); changed[:,4:] += 20
    _,other=model(changed,return_debug=True)
    torch.testing.assert_close(debug['order_states'][:,:4],other['order_states'][:,:4])
    order=torch.tensor([2,4,0,1,3])
    torch.testing.assert_close(pred[:,:,order],model(x[:,:,order]),atol=2e-6,rtol=1e-5)
    padded=torch.cat([x,torch.full((1,8,1,2),torch.nan)],2)
    valid=torch.ones(1,8,6,dtype=torch.bool); valid[:,:,-1]=False
    torch.testing.assert_close(pred,model(padded,valid)[:,:,:5])
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert (model(x,intervention={'disable_residual':True})-pred).abs().max() > 0
    alone,_=model(x[:,:,:1],return_debug=True)
    encoded=model.encoder(x[:,:,:1])
    torch.testing.assert_close(alone,model.decoder(encoded['hidden'][:,-1],encoded['positions'][:,-1],encoded['valid'][:,-1]),rtol=0,atol=0)


def test_decomposition_and_history_groups_are_observation_only():
    model=build_order_model('TOHI-H3',hidden_dim=16)
    x=torch.randn(1,8,4,2)*.1
    valid=torch.ones(1,8,4,dtype=torch.bool); valid[:,:3,1]=False
    batch=dict(observed=x,observed_valid=valid,timestamps=torch.arange(8)[None].double()*.4,
               future=torch.randn(1,12,4,2))
    groups=verification_groups(batch)
    assert set(groups)==set(VERIFICATION_GROUPS)
    assert groups['history_full8'][0].tolist()==[True,False,True,True]
    batch['future']*=1000
    assert all(torch.equal(mask,verification_groups(batch)[key]) for key,mask in groups.items())
    _,debug=model(x,valid,return_debug=True)
    torch.testing.assert_close(debug['order_states'],debug['propagation_parts']['carry']+debug['propagation_parts']['message'])
    values=extract_measured_debug(model,debug)
    assert all(torch.isfinite(v).all() for v in values.values())
    assert 'q3_before_L2' in values and 'q3_message_L2' in values and 'residual_correction_L2' in values


def test_same_projection_aggregation_budget(monkeypatch):
    import src.models.order_verification as module
    original=module.aggregate
    counts=[]
    for mode in ['strict','nondecreasing']:
        count=[0]
        def wrapped(*args):
            count[0]+=1
            return original(*args)
        monkeypatch.setattr(module,'aggregate',wrapped)
        oracle({1:[(0,0,1,1),(0,1,2,1)]},mode)
        counts.append(count[0])
    assert counts==[15,15]
