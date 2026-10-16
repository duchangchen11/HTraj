"""Seed extension equivalence, paired initialization, snapshot and cluster semantics."""
import io
import copy
import numpy as np
import pytest
import torch
from src.socialvae_stability import evaluation as new
from src.socialvae_baseline import evaluation as old
from src.socialvae_stability.model import new_model
from src.socialvae_stability.training import checkpoint_bytes,write_checkpoint_bytes_new
from src.socialvae_causal.training import seed_all
from src.socialvae_baseline.artifacts import state_info
from src.socialvae_stability.bootstrap import paired_bootstrap,draws_to_differences
from test_socialvae_baseline import examples

@pytest.mark.parametrize('seed',[17,42,2026])
def test_initial_common_parameters_and_rng_exact(seed):
 seed_all(seed);a=new_model('Full','cpu');rng_a=torch.get_rng_state().clone()
 seed_all(seed);b=new_model('PlaceboSocial','cpu');rng_b=torch.get_rng_state().clone()
 assert state_info(a)==state_info(b) and torch.equal(rng_a,rng_b)
 assert sum(p.numel() for p in a.parameters())==2144002

@pytest.mark.parametrize('seed',[17,42,2026])
def test_common_noise_identity_order_partition_future(seed,tmp_path):
 _,items=examples(tmp_path)
 for posterior in [False,True]:
  for item in items:
   changed=copy.deepcopy(item);changed['y']+=30
   assert torch.equal(new.noise(item,posterior,seed)[0],new.noise(changed,posterior,seed)[0])
   if seed==17:assert torch.equal(new.noise(item,posterior,seed)[0],old.noise(item,posterior)[0])
 eps=new.batch_noise(items,False,'cpu',seed).reshape(12,20,2,32)
 rev=new.batch_noise(items[::-1],False,'cpu',seed).reshape(12,20,2,32)
 assert torch.equal(eps,rev.flip(2))
 assert torch.equal(eps[:,:,0],new.batch_noise(items[:1],False,'cpu',seed).reshape(12,20,1,32)[:,:,0])
 assert new.noise(items[0],False,42)[1]!=new.noise(items[0],False,2026)[1]

@pytest.mark.parametrize('kind',['Full','PlaceboSocial'])
def test_seed17_evaluator_bitwise_compatible_and_rng_preserved(kind,tmp_path):
 _,items=examples(tmp_path);seed_all(17);net=new_model(kind,'cpu').train()
 rng=torch.get_rng_state().clone();a,ar=old.evaluate(net,items,128,'cpu')
 b,br=new.evaluate(net,items,128,'cpu',run_seed=17)
 assert a==b and ar==br and torch.equal(torch.get_rng_state(),rng) and net.training
 assert not net.p_z._forward_hooks and not net.q_z._forward_hooks

def test_best_snapshot_bytes_immutable_and_no_rng_consumption(tmp_path):
 seed_all(42);net=new_model('Full','cpu');opt=torch.optim.Adam(net.parameters(),lr=.0005)
 rng=torch.get_rng_state().clone();before=state_info(net)
 payload=checkpoint_bytes(net,opt,epoch=3);assert torch.equal(rng,torch.get_rng_state())
 with torch.no_grad():next(net.parameters()).add_(1)
 saved=torch.load(io.BytesIO(payload),weights_only=True);clone=new_model('Full','cpu');clone.load_state_dict(saved['model'],strict=True)
 assert state_info(clone)==before and state_info(net)!=before
 path=tmp_path/'best.pth';write_checkpoint_bytes_new(path,payload)
 with pytest.raises(FileExistsError):write_checkpoint_bytes_new(path,payload)

def test_bootstrap_preserves_entire_unequal_person_clusters():
 # Two clusters: person0 has one +10 error, person1 nine -2 errors.
 sums=np.array([[10.,20.],[-18.,-36.]]);counts=np.array([1,9])
 draws=np.array([[0,0],[1,1],[0,1],[1,0]])
 window,person=draws_to_differences(sums,counts,draws)
 np.testing.assert_allclose(window,[[10,20],[-2,-4],[-.8,-1.6],[-.8,-1.6]])
 np.testing.assert_allclose(person,[[10,20],[-2,-4],[4,8],[4,8]])

def test_bootstrap_pairing_constant_difference_and_reproducibility():
 ids=['a','a','b','c','c','c'];a=np.full(6,-.01);f=np.full(6,.02)
 first=paired_bootstrap(ids,a,f,'hotel',42)
 second=paired_bootstrap(ids,a,f,'hotel',42)
 assert first['draws'].shape==(2000,3) and np.array_equal(first['draws'],second['draws'])
 for weighting in ['window','person']:np.testing.assert_allclose(first[weighting],np.tile([-.01,.02],(2000,1)))
 assert first['seed']!=paired_bootstrap(ids,a,f,'hotel',2026)['seed']

def test_bootstrap_rejects_unpaired_or_nonfinite_data():
 with pytest.raises(ValueError):paired_bootstrap(['a'],[],[],'hotel',17)
 with pytest.raises(ValueError):paired_bootstrap(['a'],[float('nan')],[1.],'hotel',17)
