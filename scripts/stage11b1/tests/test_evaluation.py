import inspect
import pytest
import torch
from scripts.stage11b1.evaluation import metrics,sample_seed,restored_pixels,history_record,Predictor

def test_independent_oracles_and_goal_metrics():
    gt=torch.zeros(1,12,2);p=torch.zeros(20,1,12,2);g=torch.zeros(20,1,2)
    p[:,:,0:11,0]=4; p[:,:,-1,0]=6
    p[0,:,0:11,0]=0; p[0,:,-1,0]=9
    p[1,:,-1,0]=0; g[:,:,0]=20;g[0,:,0]=30
    r=metrics(p,gt,g)
    assert r['best_ADE_head'].item()==0 and r['best_FDE_head'].item()==1
    assert r['ADE1'].item()==.75 and r['FDE1'].item()==9
    assert r['GoalFDE1'].item()==30 and r['minGoalFDE_K'].item()==20

def test_raw_pixel_restoration():
    a=torch.arange(480.).reshape(20,1,12,2)
    torch.testing.assert_close(restored_pixels(a*.25,.25),a,rtol=0,atol=0)
    with pytest.raises(ValueError):restored_pixels(a,0)

def test_seed_identity_order_and_no_GT():
    ids=['SDD:a:track1:frame0','SDD:b:track7:frame24']
    assert {x:sample_seed(x) for x in ids}=={x:sample_seed(x) for x in ids[::-1]}
    assert sample_seed(ids[0]) != sample_seed(ids[1])
    a=dict(sample_id=ids[0],xy=torch.arange(40.).reshape(20,2))
    b=dict(sample_id=ids[0],xy=a['xy'].clone());b['xy'][8:]=100000
    aa=history_record(a);bb=history_record(b)
    assert aa[0]==bb[0] and torch.equal(aa[1],bb[1])
    args=set(inspect.signature(Predictor.predict).parameters)
    assert args=={'self','sample_id','history','image','semantic'}

def test_nonfinite_and_shapes():
    gt=torch.zeros(1,12,2);p=torch.ones(20,1,12,2);g=torch.zeros(20,1,2)
    r=metrics(p,gt,g);assert torch.allclose(r['ADE1'],torch.tensor([2.**.5]))
    p[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):metrics(p,gt,g)
