import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.preprocessing import Recording
from src.socialvae_causal.data import history_context,raw_example,collate_examples,PADDING
from src.socialvae_causal.evaluation import cv_prediction,trajectory_errors
from src.socialvae_causal.official import load_official,SOURCE
from scripts.stage5p1.audit_raw_inputs import scenario,change,write_raw


@pytest.fixture
def net():
    if not (SOURCE/'social_vae.py').exists():pytest.skip('Pinned external SocialVAE source required')
    return load_official().SocialVAE(12,2,32).eval()


def raw(tmp_path,name='multi_person',variant=None):
    rows=scenario(name)
    if variant:rows=change(rows,variant)
    file=tmp_path/(name+'_'+str(variant)+'.txt');write_raw(file,rows)
    return Recording.read(file,'synthetic')


@pytest.mark.parametrize('name',['stage5_entrant','complete_neighbor','partial_neighbor','single_person','multi_person','stationary','low_speed'])
@pytest.mark.parametrize('variant',['target_future','neighbor_future','future_only_person','extra_prewindow'])
def test_raw_counterfactual_input_and_prediction_invariance(tmp_path,net,name,variant):
    a=raw_example(raw(tmp_path,name),0,0)
    b=raw_example(raw(tmp_path,name,variant),0,0)
    assert np.array_equal(a['x'],b['x']) and np.array_equal(a['neighbor'],b['neighbor'])
    assert np.array_equal(a['context_ids'],b['context_ids'])
    ax,ay,an=collate_examples([a]);bx,by,bn=collate_examples([b])
    with torch.no_grad():assert torch.equal(net(ax,an),net(bx,bn))


def test_unknown_velocity_is_not_stationary_and_resets_after_gap(tmp_path):
    c=history_context(raw(tmp_path,'stage5_entrant'),0)
    i=int(np.where(c.ids==1)[0][0])
    assert c.position_valid[7,i] and not c.velocity_valid[7,i]
    assert np.all(c.packet[7,i]==PADDING)
    s=history_context(raw(tmp_path,'stationary'),0)
    j=int(np.where(s.ids==1)[0][0])
    assert s.velocity_valid[7,j] and np.array_equal(s.state[7,j,2:4],np.zeros(2))
    assert np.all(np.abs(s.packet[7,j,:2])<1e8)
    g=history_context(raw(tmp_path,'reappearing_neighbor'),0)
    j=int(np.where(g.ids==1)[0][0])
    assert g.position_valid[6,j] and not g.velocity_valid[6,j]
    assert np.all(g.packet[6,j]==PADDING) and g.velocity_valid[7,j]
    assert not g.acceleration_valid[7,j]


def test_boundary_flags_and_exact_eight_raw_queries(tmp_path):
    r=raw(tmp_path,'complete_neighbor')
    class Guard(dict):
        def get(self,key,default=None):
            assert key in range(0,80,10),key
            return super().get(key,default)
    r.by_frame=Guard(r.by_frame)
    c=history_context(r,0)
    assert c.input_support==tuple(range(0,80,10))
    assert not c.velocity_valid[0].any() and not c.acceleration_valid[:2].any()
    assert c.velocity_valid[1:].all() and c.acceleration_valid[2:].all()
    assert np.array_equal(c.state[1,:,4:6],np.zeros((2,2)))


def test_complete_targets_and_future_eligibility_do_not_select_neighbors(tmp_path):
    r=raw(tmp_path,'complete_neighbor');a=raw_example(r,0,0)
    changed=copy.deepcopy(r)
    for t in range(80,200,10):changed.by_frame[t].pop(1)
    b=raw_example(changed,0,0)
    assert np.array_equal(a['neighbor'],b['neighbor'])
    assert b['context_ids'].tolist()==[1]
    for t in range(80,200,10):changed.by_frame[t].pop(0)
    inference=raw_example(changed,0,0,require_supervised=False)
    assert np.array_equal(a['x'],inference['x']) and np.array_equal(a['neighbor'],inference['neighbor'])
    with pytest.raises(ValueError):raw_example(changed,0,0)
    entrant=raw(tmp_path,'stage5_entrant')
    with pytest.raises(ValueError):raw_example(entrant,0,1)


def test_unused_initial_velocities_and_padding_do_not_make_unknown_zero_motion(tmp_path,net):
    a=raw_example(raw(tmp_path,'complete_neighbor'),0,0)
    x,y,n=collate_examples([a])
    xx,nn=x.clone(),n.clone();xx[0,:,2:4]=123.;nn[0,:,:,2:4]=-321.
    with torch.no_grad():assert torch.equal(net(x,n),net(xx,nn))
    one=raw_example(raw(tmp_path,'single_person'),0,0)
    sx,sy,sn=collate_examples([one]);bx,by,bn=collate_examples([one,a])
    with torch.no_grad():torch.testing.assert_close(net(sx,sn)[:,0],net(bx,bn)[:,0],atol=2e-6,rtol=2e-6)
    assert np.all(a['neighbor'][8:]==PADDING)


def test_supervised_posterior_may_use_y_but_prior_cannot(tmp_path,net):
    a=raw_example(raw(tmp_path,'multi_person'),0,0);x,y,n=collate_examples([a])
    with torch.no_grad():
        prior=net.enc(x,n[:8]);h,b=net.enc(x,n,y=y);changed_h,changed_b=net.enc(x,n,y=y+5.)
    torch.testing.assert_close(prior,h)
    torch.testing.assert_close(h,changed_h)
    assert not torch.equal(b,changed_b)
    net.train();loss=net.loss(*net(x,y,n));loss['loss'].backward()
    assert all(torch.isfinite(v) for v in loss.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_global_meter_anchor_and_no_nan_with_empty_neighbors(tmp_path,net):
    a=raw_example(raw(tmp_path,'single_person'),0,0);x,y,n=collate_examples([a])
    for p in net.parameters():p.data.zero_()
    with torch.no_grad():p=net(x,n)
    assert torch.equal(p,x[-1,:,:2][None].expand(12,-1,-1))
    assert torch.isfinite(p).all()


@pytest.mark.parametrize('kind',['CV-Last2','CV-OLS4','CV-OLS8'])
def test_cv_formulas_use_physical_dt_and_complete_eight(kind):
    velocity=np.asarray([1.2,-.3]);x=np.arange(8)[:,None]*.4*velocity+[2.,3.]
    expected=x[-1]+np.arange(1,13)[:,None]*.4*velocity
    np.testing.assert_allclose(cv_prediction(x,kind),expected,atol=1e-12)
    with pytest.raises(ValueError):cv_prediction(x[:7],kind)


def test_k20_separate_best_ade_and_fde():
    p=torch.zeros(2,3,1,2);y=torch.zeros(3,1,2)
    p[0,:,0,0]=torch.tensor([0.,0.,3.]);p[1,:,0,0]=torch.tensor([3.,3.,0.])
    a,f=trajectory_errors(p,y)
    assert int(a[:,0].argmin())==0 and int(f[:,0].argmin())==1
