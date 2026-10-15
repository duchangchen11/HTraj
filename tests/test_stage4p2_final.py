import pytest
import torch
from src.evaluation.graph_final_audit import cv_prediction,errors,jitter_history,motion_group,earliest_best,observed_batch,mean_positions


@pytest.mark.parametrize('kind',['CV-Last2','CV-OLS4','CV-OLS8'])
def test_exact_linear_motion_and_physical_time(kind):
    v=torch.tensor([[1.2,-.7],[0.,0.]],dtype=torch.float64);origin=torch.tensor([[12.,-3.],[-4.,9.]],dtype=torch.float64)
    x=origin+torch.arange(8,dtype=torch.float64)[:,None,None]*.4*v
    expected=x[-1]+torch.arange(1,13,dtype=torch.float64)[:,None,None]*.4*v
    torch.testing.assert_close(cv_prediction(x,kind),expected,atol=1e-12,rtol=1e-12)
    a,f=errors(expected,expected);assert torch.equal(a,torch.zeros_like(a)) and torch.equal(f,torch.zeros_like(f))


def test_estimators_differ_for_acceleration_and_only_use_history():
    x=torch.zeros(8,2,2,dtype=torch.float64);x[:,0,0]=torch.arange(8,dtype=torch.float64).square()
    predictions=[cv_prediction(x,k) for k in ('CV-Last2','CV-OLS4','CV-OLS8')]
    assert predictions[0][-1,0,0]>predictions[1][-1,0,0]>predictions[2][-1,0,0]
    bad=x.clone();bad[0,0,0]=float('nan')
    with pytest.raises(ValueError):cv_prediction(bad,'CV-OLS8')
    with pytest.raises(ValueError):cv_prediction(x[:7],'CV-Last2')


def test_geometry_id_permutation_and_jitter_amplitude():
    x=torch.randn(8,5,2,dtype=torch.float64);p=torch.tensor([2,4,1,0,3]);q=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64);shift=torch.tensor([13.,-8.])
    for k in ('CV-Last2','CV-OLS4','CV-OLS8'):
        torch.testing.assert_close(cv_prediction(x[:,p]@q+shift,k),cv_prediction(x,k)[:,p]@q+shift)
    ids=[10,11,12,13,14];j=jitter_history(x,'unit:10',ids,1e-3,0)
    torch.testing.assert_close((j-x).norm(dim=-1),torch.full((8,5),1e-3,dtype=torch.float64),atol=1e-14,rtol=1e-10)
    jj=jitter_history(x[:,p],'unit:10',[ids[i] for i in p],1e-3,0)
    assert torch.equal(jj,j[:,p]);assert torch.equal(j,jitter_history(x,'unit:10',ids,1e-3,0))


def test_groups_only_from_history_and_selection_tie():
    speeds=torch.tensor([0.,.4,1.,2.]);x=torch.zeros(8,4,2);x[:,:,0]=torch.arange(8)[:,None]*.4*speeds
    groups,extra=motion_group(x);assert groups==['stationary','low_speed','normal_walking','fast']
    assert extra['strict_stationary'].tolist()==[True,False,False,False]
    rows=[dict(epoch=31,K1_distribution_mean_ADE=.5),dict(epoch=30,K1_distribution_mean_ADE=.5),dict(epoch=100,K1_distribution_mean_ADE=.7)]
    assert earliest_best(rows)['epoch']==30


def test_observable_only_batch_matches_original_and_mean_integrator():
    from src.graph_attribution.data import pack,real_relation
    from src.graph_attribution.model import gaussian_predictions
    from copy import deepcopy
    samples=[]
    for n in (2,5):
        x=torch.randn(8,n,2)
        samples.append(dict(observed=x,relation=real_relation(x),future=torch.randn(12,n,2),agent_ids=list(range(n))))
    a=observed_batch(samples,'cpu');b=pack(samples,'G1','cpu')
    for k in a:assert torch.equal(a[k],b[k])
    unlabeled=[{k:v for k,v in s.items() if k!='future'} for s in samples]
    for k,value in observed_batch(unlabeled,'cpu').items():assert torch.equal(value,a[k])
    p=torch.randn(2,12,5,5)*.1
    expected,_=gaussian_predictions(p,a['last'],torch.zeros(2,20,12,5,2))
    assert torch.equal(mean_positions(p,a['last']),expected)
