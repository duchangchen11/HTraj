"""Target-only capacity, not another social mechanism; exact isolation gates."""
import pytest
import torch
from src.socialvae_capacity.model import SocialVAEEgoCapacity,structural_parameter_masks
from src.socialvae_capacity.training import GradientTracker
from src.socialvae_baseline.evaluation import fixed_draws,batch_noise
from test_socialvae_attribution import packet,variants

@pytest.fixture
def net():
    torch.set_num_threads(1);torch.manual_seed(17)
    return SocialVAEEgoCapacity()

@pytest.mark.parametrize('mode',['K1','K20','posterior','training_loss','parameter_gradients'])
def test_capacity_neighbor_isolation(net,mode):
    x,y,n=packet();items=[dict(recording='synthetic',start_frame=0,actor=i) for i in range(2)]
    def run(neighbor):
        net.zero_grad(set_to_none=True)
        if mode=='K1':net.eval();return (net(x,neighbor),)
        if mode=='K20':
            net.eval()
            with fixed_draws(net.p_z,batch_noise(items,False,'cpu')):return (net(x,neighbor,20),)
        net.train()
        if mode=='posterior':
            h,b=net.enc(x,neighbor,y=y);q=net.q_z(net.rnn_fy_init(h),b[0]);return h,b,q.loc,q.scale
        with fixed_draws(net.q_z,batch_noise(items,True,'cpu')):loss=net.loss(*net(x,y,neighbor))
        if mode=='training_loss':return tuple(loss.values())
        loss['loss'].backward()
        return tuple(p.grad.clone() for p in net.parameters() if p.grad is not None)
    reference=run(n)
    for label,changed in variants(n).items():
        output=run(changed)
        assert all(torch.equal(a,b) for a,b in zip(reference,output)),(mode,label)
        assert all(torch.isfinite(a).all() for a in output)

@pytest.mark.parametrize('training',[False,True])
def test_capacity_raw_neighbor_jacobian_unused(net,training):
    x,y,n=packet();n.requires_grad_();net.train(training)
    output=net.loss(*net(x,y,n))['loss'] if training else net(x,n).sum()
    assert torch.autograd.grad(output,n,allow_unused=True)[0] is None

def test_capacity_count_and_real_gradient(net):
    x,y,n=packet();net.train();tracker=GradientTracker(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=.0005)
    before={k:p.detach().clone() for k,p in net.ego_capacity.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);net.loss(*net(x,y,n))['loss'].backward()
        for p in net.ego_capacity.parameters():assert p.grad is not None and p.grad.count_nonzero()>0
        tracker.update(net);optimizer.step()
    masks=structural_parameter_masks(net)
    assert sum(p.numel() for p in net.parameters())==2669570
    assert sum(int(v.sum()) for v in masks.values())==1985474
    assert abs(1985474/2045698-1)<.05
    assert sum(p.numel() for p in net.parameters() if p.grad is not None)==2182082
    for k,p in net.ego_capacity.named_parameters():assert not torch.equal(p,before[k])
    rows=tracker.snapshot(net,1)
    assert sum(r['ever_nonzero_gradient_scalars'] for r in rows if r['parameter'].startswith('ego_capacity.'))>0

def test_capacity_checkpoint_strict_roundtrip(net,tmp_path):
    x,y,n=packet();optimizer=torch.optim.Adam(net.parameters(),lr=.0005)
    net.train();net.loss(*net(x,y,n))['loss'].backward();optimizer.step()
    path=tmp_path/'checkpoint.pth';torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict()),path)
    clone=SocialVAEEgoCapacity();saved=torch.load(path,weights_only=True);clone.load_state_dict(saved['model'],strict=True)
    opt=torch.optim.Adam(clone.parameters(),lr=.0005);opt.load_state_dict(saved['optimizer'])
    net.eval();clone.eval();assert torch.equal(net(x,n),clone(x,n))
    net.train();clone.train();torch.manual_seed(7);a=net.loss(*net(x,y,n))
    torch.manual_seed(7);b=clone.loss(*clone(x,y,n));assert all(torch.equal(a[k],b[k]) for k in a)

def test_capacity_history_encoder_cannot_read_own_future(net):
    x,y,n=packet();net.train();a,b=net.enc(x,n,y=y);c,d=net.enc(x,n,y=y+3)
    assert torch.equal(a,c) and not torch.equal(b,d)
