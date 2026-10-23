import sys
import torch
import pytest
from scripts.stage11c1r.observer import KMeansObserver
from scripts.stage11c1r.common import Budget,rng_digest
from scripts.stage11b0.runtime import original_modules
from scripts.stage11b1r.sampler import isolated_kmeans_rng

def test_observer_cpu_counts_and_bitwise_rng_isolation():
    km=original_modules().model.evaluate.__globals__['kmeans']
    points=torch.tensor([[0.,0.],[.1,0.],[10.,0.],[10.1,0.],[0.,10.],[0.,10.1]])
    sid='runtime-unit-test';before=rng_digest()
    with isolated_kmeans_rng(sid,torch.device('cpu')):
        a,b=km(points,3,device=torch.device('cpu'),tqdm_flag=False,tol=.001,iter_limit=1000)
    assert rng_digest()==before
    with KMeansObserver(km) as monitor:
        with isolated_kmeans_rng(sid,torch.device('cpu')):
            c,d=km(points,3,device=torch.device('cpu'),tqdm_flag=False,tol=.001,iter_limit=1000)
    assert sys.getprofile() is None
    assert torch.equal(a,c) and torch.equal(b,d) and rng_digest()==before
    row=monitor.result();assert 0<row['iterations']<=1000 and row['tol_converged']
    assert row['output_device']=='cpu' and row['choice_cluster_device']=='cpu'

def test_profile_is_removed_after_failure():
    km=original_modules().model.evaluate.__globals__['kmeans']
    with pytest.raises(RuntimeError):
        with KMeansObserver(km):raise RuntimeError('synthetic test failure')
    assert sys.getprofile() is None

def test_new_safety_budget_boundary_and_save_reserve():
    clock=[0.];b=Budget(prior=2000,clock=lambda:clock[0]);b.guard(189)
    with pytest.raises(TimeoutError):b.guard(190)
    with pytest.raises(ValueError):b.guard(-1)
