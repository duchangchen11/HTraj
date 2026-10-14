import pytest
import torch
from src.models.relation_routing import build_routing_model,receiver_candidates,HistoricalCV
from src.models.order_verification import build_order_model
from src.models.ego_encoder import motion_inputs
from src.evaluation.relation_routing import routing_groups


def entry():
    x=torch.zeros(1,8,3,2); x[0,:,1,0]=torch.arange(8)*.1+1.;x[0,:,2,0]=10.
    mask=torch.ones(1,8,3,dtype=torch.bool);mask[0,:7,0]=False
    return x,mask


def test_new_target_receives_and_unknown_is_explicit():
    x,m=entry();model=build_routing_model('Receiver-Fix',16)
    _,d=model(x,m,return_debug=True);f=d['graph_frames'][-1]
    edge=(f.target==0) & (f.source==1)
    assert edge.sum()==1 and f.features[edge,6].item()==2
    assert f.features[edge,2:5].eq(0).all() # placeholders marked unknown; no CPA
    assert d['order_support'][0,-1,0,0]
    assert d['propagation_parts']['carry'][0,-1,0].eq(0).all()
    assert d['propagation_parts']['message'][0,-1,0].abs().sum()>0


def test_source_without_history_cannot_send_or_forge_motion():
    x,m=entry();m[0,:7,1]=False
    e=motion_inputs(x,m);mask,_=receiver_candidates(e,7)
    assert not mask[0,:,1].any() and not e['continuity'][0,7,1]
    assert not mask[0,0].any()


@pytest.mark.parametrize('name',['Receiver-Fix','Relation-Routing','Routing-Current','Routing-NoUpdate','Routing-Uniform','Routing-Random'])
def test_no_neighbors_exact_ego_and_finite_gradients(name):
    x=torch.zeros(1,8,1,2,requires_grad=True);model=build_routing_model(name,16)
    p,d=model(x,return_debug=True)
    assert torch.equal(d['fused_context'],d['hidden'][:,-1])
    assert d['residual_gate'].eq(0).all()
    p.square().sum().backward();assert torch.isfinite(x.grad).all()


def test_receiver_fix_same_known_graph_and_numeric_as_h1():
    torch.manual_seed(17);old=build_order_model('TOHI-H1',16)
    torch.manual_seed(17);new=build_routing_model('Receiver-Fix',16)
    x=torch.randn(1,8,4,2)*.1
    a,da=old(x,return_debug=True);b,db=new(x,return_debug=True)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for f,g in zip(da['graph_frames'],db['graph_frames']):
        assert torch.equal(f.source,g.source) and torch.equal(f.features,g.features)
    a.sum().backward();b.sum().backward()
    for (n,p),(k,q) in zip(old.named_parameters(),new.named_parameters()):
        assert p.shape==q.shape and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


@pytest.mark.parametrize('name',['Receiver-Fix','Relation-Routing','Routing-Current','Routing-NoUpdate','Routing-Uniform','Routing-Random'])
def test_padding_and_permutation(name):
    x,m=entry(); model=build_routing_model(name,16).eval();p=model(x,m)
    perm=torch.tensor([2,0,1]);q=model(x[:,:,perm],m[:,:,perm]);torch.testing.assert_close(p[:,:,perm],q,atol=1e-6,rtol=1e-6)
    pad=torch.cat([x,torch.full((1,8,2,2),float('nan'))],2);mask=torch.cat([m,torch.zeros(1,8,2,dtype=torch.bool)],2)
    q=model(pad,mask);torch.testing.assert_close(p,q[:,:,:3],atol=1e-6,rtol=1e-6);assert q[:,:,3:].eq(0).all()


def test_prefix_causality_and_memory_reset():
    model=build_routing_model('Relation-Routing',16);x,m=entry();m[:]=True
    e=model.encoder(x,m);frames,details=model.graph.sequence(e)
    assert details[2]['memory_inherited'].any()
    # Pair disappears for a step, then returns: no inherited edge state.
    z=x.clone();z[:,3,0,0]=20.;e2=model.encoder(z,m);f,d=model.graph.sequence(e2)
    for pair,keep in zip(d[4]['pair'],d[4]['memory_inherited']):
        if int(pair) in (1,3):assert not keep
    z=x.clone();z[:,5:]+=100.;e3=model.encoder(z,m);f3,d3=model.graph.sequence(e3)
    for t in range(5):
        torch.testing.assert_close(details[t]['edge_state'],d3[t]['edge_state'],atol=0,rtol=0)
    m[:,3,0]=False;e4=model.encoder(x,m);f4,d4=model.graph.sequence(e4)
    for pair,keep in zip(d4[4]['pair'],d4[4]['memory_inherited']):
        if int(pair)//3==0:assert not keep


def test_no_update_same_params_but_history_dependency_differs():
    torch.manual_seed(17);a=build_routing_model('Relation-Routing',16)
    torch.manual_seed(17);b=build_routing_model('Routing-NoUpdate',16)
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    x,m=entry();m[:]=True
    fa,da=a.graph.sequence(a.encoder(x,m));fb,db=b.graph.sequence(b.encoder(x,m))
    assert all(d['before'].eq(0).all() for d in db)
    assert da[-1]['before'].abs().sum()>0
    assert not torch.equal(da[-1]['edge_state'],db[-1]['edge_state'])


def test_future_gt_not_in_graph_api_and_groups_partition():
    x,m=entry();b=dict(observed=x,observed_valid=m,timestamps=torch.arange(8,dtype=torch.float64)[None]*.4)
    g=routing_groups(b);b.update(future=torch.randn(1,12,3,2)*1e6,future_valid=torch.zeros(1,12,3,dtype=torch.bool))
    h=routing_groups(b);assert all(torch.equal(g[k],h[k]) for k in g)
    assert g['one_frame_with_neighbors'][0,0]
    model=build_routing_model('Receiver-Fix',16);assert 'future' not in __import__('inspect').signature(model.forward).parameters


def test_cv_no_future_velocity_and_stationary_fallback():
    x,m=entry();p=HistoricalCV()(x,m)
    assert p[0,:,0].eq(x[0,-1,0]).all()
    torch.testing.assert_close(p[0,-1,1],x[0,-1,1]+torch.tensor([1.2,0.]),atol=1e-6,rtol=1e-6)


def test_temporal_gap_resets_carry_and_relation_memory():
    x,m=entry();m[:]=True;t=torch.arange(8,dtype=torch.float64)[None]*.4;t[:,4:]+=2
    model=build_routing_model('Relation-Routing',16);_,d=model(x,m,t,True)
    assert d['propagation_parts']['carry'][:,4].eq(0).all()
    assert not d['relation_details'][4]['memory_inherited'].any()


def test_reappearing_receiver_resets_previously_nonzero_social_carry():
    x,m=entry();m[:]=True;m[:,4:7,0]=False
    model=build_routing_model('Receiver-Fix',16);_,d=model(x,m,return_debug=True)
    assert d['order_states'][0,3,0].abs().sum()>0
    assert d['propagation_parts']['before'][0,7,0].eq(0).all()
    assert d['propagation_parts']['carry'][0,7,0].eq(0).all()
    assert d['order_support'][0,7,0,0]


def test_random_control_legality_and_incoming_budget():
    torch.manual_seed(7);x=torch.randn(2,8,6,2).cumsum(1)*.1;m=torch.ones(2,8,6,dtype=torch.bool);m[0,:7,0]=False
    e=motion_inputs(x,m)
    for t in range(8):
        real,_=receiver_candidates(e,t);random,_=receiver_candidates(e,t,random=True)
        assert torch.equal(real.sum(-1),random.sum(-1))
        assert not random.diagonal(dim1=-2,dim2=-1).any()
        b,i,j=random.nonzero(as_tuple=True)
        assert e['continuity'][b,t,j].all() and e['valid'][b,t,i].all()


def test_pair_history_affects_routing_with_current_motion_and_ego_fixed():
    x=torch.zeros(1,8,3,2);x[:,:,1,0]=1.;x[:,:,2,0]=-1.
    z=x.clone();z[:,2,1,1]=.3
    ea=motion_inputs(x);eb=motion_inputs(z)
    # Freeze ego representations explicitly, isolating relation-state memory.
    ea['hidden']=torch.zeros(1,8,3,16);eb['hidden']=ea['hidden'].clone()
    torch.manual_seed(4);history=build_routing_model('Relation-Routing',16)
    torch.manual_seed(4);current=build_routing_model('Routing-NoUpdate',16)
    ha,_=history.graph.sequence(ea);hb,_=history.graph.sequence(eb)
    ca,_=current.graph.sequence(ea);cb,_=current.graph.sequence(eb)
    assert torch.equal(ha[-1].source,hb[-1].source)
    assert torch.equal(ha[-1].features,hb[-1].features)
    assert not torch.equal(ha[-1].weight,hb[-1].weight)
    assert torch.equal(ca[-1].weight,cb[-1].weight)
