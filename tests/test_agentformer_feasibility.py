import random
import numpy as np
import pytest
import torch
from src.agentformer_feasibility.common import nested_equal,rng_state,restore_rng
from src.agentformer_feasibility.diagnosis import cv_predictions,history_labels,candidate_errors,context_bin,step_summary


def test_cv_units_and_last_position_anchor():
    t=np.arange(8)*.4
    x=np.stack([np.c_[2+1.5*t,-3+.3*t],np.ones((8,2))*4])
    expected=x[:,-1,None]+np.array([[1.5,.3],[0,0]])[:,None]*np.arange(1,13)[None,:,None]*.4
    for y in cv_predictions(x).values():np.testing.assert_allclose(y,expected,atol=1e-12)
    with pytest.raises(ValueError):cv_predictions(x[:,:7])


def test_history_speed_thresholds_and_translation_invariance():
    speeds=np.array([0,.049,.051,.399,.401,1.799,1.801])
    x=np.zeros((len(speeds),8,2));x[:,:,0]=speeds[:,None]*np.arange(8)[None]*.4
    s,labels=history_labels(x)
    np.testing.assert_allclose(s,speeds)
    assert labels.tolist()==['stationary','stationary','low','low','normal','normal','fast']
    np.testing.assert_array_equal(history_labels(x+100)[1],labels)
    assert [context_bin(n) for n in [0,1,3,4,9,10]]==['none','1-3','1-3','4-9','4-9','10+']


def test_candidates_independent_winners_and_dispersion():
    y=np.zeros((1,12,2));p=np.full((1,20,12,2),100.)
    p[0,0,:,0]=1;p[0,0,:,1]=0;p[0,0,-1,0]=10
    p[0,1,:,0]=2;p[0,1,:,1]=0;p[0,1,-1,0]=0
    r=candidate_errors(p,y)
    assert r['ADE_winner'][0]==0 and r['FDE_winner'][0]==1
    assert r['minADE20'][0]==1.75 and r['minFDE20'][0]==0
    np.testing.assert_allclose(r['ADE_best_step'].mean(1),r['minADE20'])
    assert r['endpoint_RMS'][0]>0 and r['endpoint_pairwise_mean'][0]>0
    with pytest.raises(ValueError):candidate_errors(p[:,:19],y)


def test_rng_and_adam_scheduler_strict_restoration():
    torch.manual_seed(1);np.random.seed(1);random.seed(1)
    model=torch.nn.Linear(2,1);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,10,.5)
    (model(torch.ones(1,2)).sum()**2).backward();opt.step();scheduler.step()
    state=rng_state();a=(random.random(),np.random.rand(),torch.rand(2));restore_rng(state)
    b=(random.random(),np.random.rand(),torch.rand(2));assert nested_equal(a,b)
    clone=torch.nn.Linear(2,1);clone.load_state_dict(model.state_dict(),strict=True)
    op=torch.optim.Adam(clone.parameters(),lr=1e-4);sc=torch.optim.lr_scheduler.StepLR(op,10,.5)
    op.load_state_dict(opt.state_dict());sc.load_state_dict(scheduler.state_dict())
    assert nested_equal(opt.state_dict(),op.state_dict()) and nested_equal(scheduler.state_dict(),sc.state_dict())
    assert len(op.state)>0


def test_warmup_only_affects_timing_summary_not_update_budget():
    r=step_summary([dict(step_seconds=float(t)) for t in range(1,21)],10)
    assert r['updates']==20 and r['steady_step_count']==10 and r['steady_mean_seconds']==15.5


def test_history_neighbor_groups_ignore_all_future_changes():
    from src.agentformer_feasibility.diagnosis import history_neighbor_counts
    raw=np.array([[t,i,float(i),0] for t in range(20) for i in [1,2,8]],dtype=float)
    a=history_neighbor_counts(raw,7,[1,2])
    changed=raw.copy();changed[changed[:,0]>7,2:]+=1000
    changed=changed[~((changed[:,0]>7)&(changed[:,1]==2))]
    changed=np.vstack([changed,[[12,99,0,0],[13,99,0,0]]])
    b=history_neighbor_counts(changed,7,[1,2])
    np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]==3
    assert a[0].tolist()==[1,1]


def test_observational_norm_guard_never_changes_gradients():
    from scripts.stage7p4.train_epoch import finite_norm
    p=torch.nn.Parameter(torch.tensor([3.,4.]));p.grad=torch.tensor([5.,12.])
    saved=p.grad.clone()
    assert finite_norm([p],grad=True)==13
    torch.testing.assert_close(p.grad,saved,atol=0,rtol=0)
    p.grad[0]=float('inf')
    with pytest.raises(FloatingPointError):finite_norm([p],grad=True)
