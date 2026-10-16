"""The Full network executes, but its context contains no real neighbor data."""
import pytest
import torch
from src.socialvae_capacity.placebo import SocialVAEPlacebo,synthetic_context,structural_parameter_masks
from src.socialvae_attribution.model import OfficialSocialVAE
from src.socialvae_baseline.evaluation import fixed_draws,batch_noise
from test_socialvae_attribution import packet,variants

@pytest.fixture
def net():
 torch.set_num_threads(1);torch.manual_seed(17)
 return SocialVAEPlacebo(12,2,256)

@pytest.mark.parametrize('mode',['K1','K20','posterior','loss','gradients'])
def test_placebo_real_neighbor_isolation(net,mode):
 x,y,n=packet();items=[dict(recording='synthetic',start_frame=0,actor=i) for i in range(2)]
 def run(neighbor):
  net.zero_grad(set_to_none=True)
  if mode=='K1':net.eval();return (net(x,neighbor),)
  if mode=='K20':
   net.eval()
   with fixed_draws(net.p_z,batch_noise(items,False,'cpu')):return (net(x,neighbor,20),)
  net.train()
  if mode=='posterior':
   h,b=net.enc(x,neighbor,y=y);d=net.q_z(net.rnn_fy_init(h),b[0]);return h,b,d.loc,d.scale
  with fixed_draws(net.q_z,batch_noise(items,True,'cpu')):loss=net.loss(*net(x,y,neighbor))
  if mode=='loss':return tuple(loss.values())
  loss['loss'].backward();return tuple(p.grad.clone() for p in net.parameters() if p.grad is not None)
 reference=run(n)
 for label,other in variants(n).items():
  assert all(torch.equal(a,b) for a,b in zip(reference,run(other))),(mode,label)

@pytest.mark.parametrize('training',[False,True])
def test_placebo_neighbor_input_jacobian_unused(net,training):
 x,y,n=packet();n.requires_grad_();net.train(training)
 output=net.loss(*net(x,y,n))['loss'] if training else net(x,n).sum()
 assert torch.autograd.grad(output,n,allow_unused=True)[0] is None

def test_synthetic_packet_prefix_causality_and_kinematics():
 x,y,n=packet();p=synthetic_context(x,True)
 assert p.shape==(20,2,4,6) and torch.all(p[8:]==1e9)
 assert torch.isfinite(p).all()
 assert ((p[:8,:,:,:2]-x[:,:,:2].unsqueeze(2)).norm(dim=-1)<1.1).all()
 assert torch.equal(p[1:8,:,:,2:4],p[1:8,:,:,:2]-p[:7,:,:,:2])
 assert torch.equal(p[2:8,:,:,4:6],p[2:8,:,:,2:4]-p[1:7,:,:,2:4])
 for t in range(7):
  changed=x.clone();changed[t+1:]+=3.
  assert torch.equal(p[:t+1],synthetic_context(changed,True)[:t+1])
 assert torch.equal(p[:8],synthetic_context(x,False))

def test_placebo_is_exact_full_network_on_synthetic_input(net):
 x,y,n=packet();full=OfficialSocialVAE(12,2,256);full.load_state_dict(net.state_dict(),strict=True)
 assert sum(p.numel() for p in net.parameters())==2144002
 assert sum(int(v.sum()) for v in structural_parameter_masks(net).values())==2045698
 net.eval();full.eval();assert torch.equal(net(x,n),full(x,synthetic_context(x)))
 net.train();full.train();torch.manual_seed(7);a=net.loss(*net(x,y,n))
 torch.manual_seed(7);b=full.loss(*full(x,y,synthetic_context(x,True)))
 assert all(torch.equal(a[k],b[k]) for k in a)
 a['loss'].backward()
 for prefix in ['rnn_fx_init.','embed_n.','embed_k.','embed_q.','rnn_fx.']:
  ps=[p for name,p in net.named_parameters() if name.startswith(prefix)]
  assert ps and all(p.grad is not None and p.grad.count_nonzero()>0 for p in ps),prefix

def test_placebo_strict_checkpoint_roundtrip(net,tmp_path):
 x,y,n=packet();net.train();net.loss(*net(x,y,n))['loss'].backward()
 opt=torch.optim.Adam(net.parameters(),lr=.0005);opt.step()
 path=tmp_path/'placebo.pth';torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),path)
 clone=SocialVAEPlacebo(12,2,256);saved=torch.load(path,weights_only=True);clone.load_state_dict(saved['model'],strict=True)
 opt2=torch.optim.Adam(clone.parameters(),lr=.0005);opt2.load_state_dict(saved['optimizer'])
 net.eval();clone.eval();assert torch.equal(net(x,n),clone(x,n))
