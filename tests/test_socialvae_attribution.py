"""Isolation needs interventions AND Jacobians: upstream detaches raw features."""
import copy
import pytest
import torch
from src.socialvae_attribution.model import SocialVAENoSocial,OfficialSocialVAE,structural_parameter_masks
from src.socialvae_baseline.evaluation import fixed_draws,batch_noise


def packet(B=2,M=3):
    time=torch.arange(8,dtype=torch.float32)[:,None]
    x=torch.zeros(8,B,6);x[:,:,:2]=torch.stack((time*.2+torch.arange(B)[None,:]*.1,time*.05+torch.zeros(1,B)),dim=-1)
    x[1:,:,2:4]=x[1:,:,:2]-x[:-1,:,:2]
    n=torch.full((20,B,M,6),1e9)
    for j in range(M):
        n[:8,:,j,:2]=x[:,:,:2]+torch.tensor([.3+j*.2,.4])
        n[:8,:,j,2:4]=torch.tensor([.1,.0]);n[:8,:,j,4:6]=0.
    y=x[-1,:,:2][None]+torch.arange(1,13)[:,None,None]*torch.tensor([.16,.03])[None,None,:]
    return x,y,n


def variants(n):
    p=n.clone();p[:8,:,:,:2]+=.5
    v=n.clone();v[:8,:,:,2:4]+=.3
    entry=n.clone();entry[:6,:,:,0:6]=1e9
    future=n.clone();future[8:]=torch.randn_like(future[8:])
    return {'position':p,'velocity':v,'number_empty':n[:,:,:0],
        'number_more':n.repeat(1,1,2,1),'order':n.flip(2),'entry':entry,
        'future_packet':future,'all_nonfinite':torch.full_like(n,float('nan'))}


@pytest.fixture
def models():
    torch.set_num_threads(1);torch.manual_seed(17)
    full=OfficialSocialVAE(12,2,256)
    ns=SocialVAENoSocial(12,2,256);ns.load_state_dict(full.state_dict(),strict=True)
    return full,ns


@pytest.mark.parametrize('mode',['K1','K20','posterior','training_loss'])
def test_no_social_interventions_exactly_invariant(models,mode):
    full,net=models;x,y,n=packet()
    items=[dict(recording='synthetic',start_frame=0,actor=i) for i in range(2)]
    def run(neighbor):
        if mode=='K1':net.eval();return (net(x,neighbor),)
        if mode=='K20':
            net.eval()
            with fixed_draws(net.p_z,batch_noise(items,False,'cpu')):return (net(x,neighbor,20),)
        net.train()
        if mode=='posterior':
            h,b=net.enc(x,neighbor,y=y);d=net.q_z(net.rnn_fy_init(h),b[0]);return h,b,d.loc,d.scale
        with fixed_draws(net.q_z,batch_noise(items,True,'cpu')):return tuple(net.loss(*net(x,y,neighbor)).values())
    reference=run(n)
    for label,changed in variants(n).items():
        output=run(changed)
        assert all(torch.equal(a,b) for a,b in zip(reference,output)),(mode,label)
        assert all(torch.isfinite(a).all() for a in output)


@pytest.mark.parametrize('training',[False,True])
def test_no_social_raw_neighbor_jacobian_is_unused(models,training):
    _,net=models;x,y,n=packet();n.requires_grad_();net.train(training)
    if training:output=net.loss(*net(x,y,n))['loss']
    else:output=net(x,n).sum()
    assert torch.autograd.grad(output,n,allow_unused=True)[0] is None


def test_full_detach_does_not_imply_no_social_dependence(models):
    net,_=models;net.eval();x,y,n=packet();n.requires_grad_()
    baseline=net(x,n)
    assert torch.autograd.grad(baseline.sum(),n,allow_unused=True)[0] is None
    # Numerical intervention is essential: raw features are constructed in no_grad.
    for label in ['position','velocity','number_empty','entry']:
        other=variants(n.detach())[label]
        assert not torch.equal(net(x,other),baseline),label
    torch.testing.assert_close(net(x,n.flip(2)),baseline,rtol=1e-5,atol=1e-5)


def test_no_social_equals_full_with_all_social_features_masked(models):
    full,ns=models;x,y,n=packet();n.fill_(1e9)
    full.eval();ns.eval()
    assert torch.equal(full(x,n),ns(x,n))
    full.train();ns.train()
    torch.manual_seed(9);a=full.loss(*full(x,y,n))
    torch.manual_seed(9);b=ns.loss(*ns(x,y,n))
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_loss_parameter_gradients_and_dead_columns(models):
    full,ns=models;x,y,n=packet()
    for net,label in [(full,'Full'),(ns,'NoSocial')]:
        net.train();loss=net.loss(*net(x,y,n));loss['loss'].backward()
        masks=structural_parameter_masks(net,label)
        assert sum(p.numel() for p in net.parameters())==2144002
        assert sum(int(v.sum()) for v in masks.values())==({'Full':2045698,'NoSocial':1459906}[label])
        for name,p in net.named_parameters():
            if not masks[name].any():assert p.grad is None
            else:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert not p.grad.detach().cpu()[~masks[name]].count_nonzero()


def test_future_neighbor_inputs_do_not_enter_causal_full_posterior(models):
    full,_=models;x,y,n=packet();full.train()
    # Protocol packets, not arbitrary official future neighbor slots, are sealed.
    assert torch.all(n[8:]==1e9)
    h,b=full.enc(x,n,y=y)
    h2,b2=full.enc(x,n,y=y+1.)
    assert torch.equal(h,h2) and not torch.equal(b,b2)


def test_lifetime_gradient_tracker_and_unchanged_dead_weights(models):
    from src.socialvae_attribution.training import GradientTracker
    _,net=models;x,y,n=packet();tracker=GradientTracker(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=.0005)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        net.loss(*net(x,y,n))['loss'].backward();optimizer.step();tracker.update(net)
    rows=tracker.snapshot(net,1)
    assert all(r['dead_parameter_values_unchanged'] for r in rows)
    assert all(r['optimizer_steps_observed']==2 for r in rows)
    assert sum(r['ever_nonzero_gradient_scalars'] for r in rows)>0
    assert all(not r['gradient_connected_ever'] for r in rows if r['parameter'].startswith(('embed_n.','embed_k.','embed_q.','rnn_fx_init.')))
