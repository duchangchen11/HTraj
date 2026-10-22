import json
import numpy as np
import torch
import pytest
from scripts.stage11c0.interventions import intervene
from scripts.stage11c0.common import ROOT,OUT,CONF,checked,digest,Budget

@pytest.mark.parametrize('shape,valid', [((1,6,8,12),(5,9)),((1,6,3,4),(3,4))])
def test_all_four_definitions_and_input_immutability(shape,valid):
    s=torch.arange(np.prod(shape),dtype=torch.float32).reshape(shape)/100-2;original=s.clone();h,w=valid
    a=intervene(s,'S0',h,w);z=intervene(s,'S1',h,w);m=intervene(s,'S2',h,w);f=intervene(s,'S3',h,w)
    assert torch.equal(s,original) and torch.equal(a,s) and a.data_ptr()!=s.data_ptr()
    assert torch.count_nonzero(z)==0 and z.shape==s.shape and z.dtype==s.dtype
    assert torch.equal(m[:,:,:h,:w],s[:,:,:h,:w].mean((-2,-1),keepdim=True).expand(-1,-1,h,w))
    assert torch.equal(f[:,:,:h,:w],s[:,:,:h,:w].flip(-1))
    assert torch.equal(intervene(f,'S3',h,w),s)
    for v in [m,f]:
        assert torch.equal(v[:,:,h:,:],s[:,:,h:,:]) and torch.equal(v[:,:,:,w:],s[:,:,:,w:])

def test_history_channels_never_modified_by_semantic_only_function():
    s=torch.rand(1,6,4,5);h=torch.rand(1,8,4,5)
    for c in ['S0','S1','S2','S3']:
        v=intervene(s,c,3,4);combined=torch.cat([v,h],1)
        assert torch.equal(combined[:,6:],h) and combined.shape[1]==14
        assert torch.isfinite(v).all()

def test_reject_invalid_shape_dtype_and_valid_extent():
    for x in [torch.zeros(1,5,4,4),torch.zeros(1,6,4,4,dtype=torch.float64),torch.full((1,6,4,4),float('nan'))]:
        with pytest.raises(AssertionError):intervene(x,'S0',4,4)
    with pytest.raises(AssertionError):intervene(torch.zeros(1,6,4,4),'S3',5,4)

def test_actual_five_cache_interventions_padding_and_SHA():
    cfg=json.loads((CONF/'generation_config.json').read_text())
    assert len(cfg['semantics']['S0'])==5
    for scene,rec in cfg['semantics']['S0'].items():
        s=checked(rec);assert digest(s)==rec['tensor_SHA']
        for c in ['S0','S1','S2','S3']:
            r=cfg['semantics'][c][scene];v=checked(r)
            assert digest(v)==r['tensor_SHA'] and torch.equal(v,intervene(s,c,r['valid_height'],r['valid_width']))

def test_budget_accounts_all_previous_workers_and_startup():
    b=Budget(prior=500,clock=lambda:100,started=50);assert b.elapsed()==550
    b=Budget(prior=6500,clock=lambda:100,started=50)
    with pytest.raises(TimeoutError):b.guard(50)

def test_future_input_projection_and_unmodified_sampler_source():
    from scripts.stage11b1.evaluation import history_record
    xy=np.arange(40,dtype=np.float32).reshape(20,2);raw=dict(sample_id='synthetic',xy=xy.copy())
    a,h=history_record(raw);raw['xy'][8:]=-1e6;b,k=history_record(raw)
    assert a==b and torch.equal(h,k)
    text=(ROOT/'scripts/stage11c0/worker.py').read_text()
    assert 'AuditPredictor' in text and 'cluster_proposals(' not in text and 'torch.multinomial(' not in text
    assert 'optimizer' not in text and '.backward(' not in text
