import copy
import numpy as np
import torch
from src.public_training.data import select_window,graph_from_history,permute_graph_input,restore_predictions,displacement
from src.public_training.model import make_model,forward,nll,gaussian_trajectories


def test_graph_matches_official_and_future_values_cannot_enter():
    import networkx as nx
    from src.evaluation.public_baseline import load_module
    if not hasattr(nx,'from_numpy_matrix'):nx.from_numpy_matrix=nx.from_numpy_array
    upstream=load_module('stage4p1_official_utils_test','third_party/stage4/social_stgcnn/utils.py')
    torch.manual_seed(17);x=torch.randn(8,4,2).cumsum(0);rel=displacement(x)
    _,a=upstream.seq_to_graph(x.permute(1,2,0),rel.permute(1,2,0))
    torch.testing.assert_close(graph_from_history(x),a,atol=2e-6,rtol=2e-6)


def test_B_context_uses_history_presence_not_future():
    frames=list(range(0,200,10));by={f:{1:[f*.01,0],2:[0,f*.01],3:[1,1]} for f in frames}
    original=select_window(by,frames,'B_history_context_complete_targets')
    for f in frames[8:]:del by[f][3];by[f][1]=[10000.,-4000.]
    changed=select_window(by,frames,'B_history_context_complete_targets')
    assert original['agent_ids']==changed['agent_ids']==[1,2,3]
    assert changed['target_ids']==[1,2]
    assert torch.equal(original['observed'],changed['observed'])
    assert torch.equal(graph_from_history(original['observed']),graph_from_history(changed['observed']))
    offline=select_window(by,frames,'A_offline_complete20');assert offline['agent_ids']==[1,2]


def test_permutation_matrix_and_restore_are_exact():
    order=torch.tensor([2,0,3,1]);P=torch.eye(4)[order]
    v=torch.arange(64.).reshape(1,2,8,4);a=torch.randn(8,4,4)
    vv,aa=permute_graph_input(v,a,order)
    torch.testing.assert_close(aa,P[None]@a@P.T[None])
    assert torch.equal(restore_predictions(vv,order),v)
    assert torch.equal(restore_predictions(aa,order).index_select(-2,torch.argsort(order)),a)


def test_nosocial_has_equal_parameters_and_zero_cross_person_jacobian():
    for training in (False,True):
        torch.manual_seed(17);m=make_model('NoSocial');m.train(training)
        x=torch.randn(1,2,8,4,requires_grad=True);a=torch.randn(8,4,4)
        original=forward(m,x,a);original[:,0].sum().backward()
        assert torch.equal(x.grad[...,1:],torch.zeros_like(x.grad[...,1:]))
        changed=x.detach().clone();changed[...,1:]+=40
        torch.testing.assert_close(forward(m,changed,a)[:,0],original[:,0],atol=1e-6,rtol=1e-6)
    assert sum(p.numel() for p in make_model('Full').parameters())==sum(p.numel() for p in m.parameters())==7563


def test_nosocial_permutation_and_single_node_and_checkpoint(tmp_path):
    torch.manual_seed(17);m=make_model('NoSocial').eval();x=torch.randn(1,2,8,4);a=torch.eye(4)[None].repeat(8,1,1)
    p=torch.tensor([3,1,0,2]);v,b=permute_graph_input(x,a,p)
    with torch.no_grad():
        y=forward(m,x,a);yy=restore_predictions(forward(m,v,b).permute(2,0,1)[None],p)[0].permute(1,2,0)
        torch.testing.assert_close(yy,y,atol=1e-6,rtol=1e-6)
        assert forward(m,x[...,:1],a[:,:1,:1]).shape==(12,1,5)
        path=tmp_path/'model.pt';torch.save(m.state_dict(),path);other=make_model('NoSocial').eval();other.load_state_dict(torch.load(path,weights_only=True))
        assert torch.equal(forward(other,x,a),y)


def test_official_nll_gradients_and_correlated_sampling():
    torch.manual_seed(17);m=make_model('Full');v=torch.randn(1,2,8,3);a=torch.eye(3)[None].repeat(8,1,1)
    y=forward(m,v,a);loss=nll(y,torch.randn(12,3,2));loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    pars=torch.zeros(12,3,5);noise=torch.zeros(20,12,3,2);last=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    mean,samples=gaussian_trajectories(pars,last,noise)
    assert mean.shape==(12,3,2) and samples.shape==(20,12,3,2)
    assert torch.equal(mean,last.expand_as(mean)) and torch.equal(samples,mean.expand_as(samples))


def test_identity_graph_does_not_remove_txpcnn_social_path():
    torch.manual_seed(17);model=make_model('Full').eval();x=torch.randn(1,2,8,4,requires_grad=True)
    graph=torch.eye(4)[None].expand(8,-1,-1);forward(model,x,graph)[:,0].sum().backward()
    assert x.grad[...,1:].abs().max()>1e-8
