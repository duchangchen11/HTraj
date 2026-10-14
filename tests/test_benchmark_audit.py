import copy
import pytest
import torch
from src.data.benchmark_audit import apply_protocol,PROTOCOLS,assert_recording_isolation
from src.evaluation.benchmark_audit import history_features,target_errors
from src.models.relation_routing import HistoricalCV


def sample():
    x=torch.zeros(8,4,2);x[:,1,0]=torch.arange(8)*.1;x[:,2,0]=2.-torch.arange(8)*.1;x[:,3,1]=torch.arange(8)*.1
    obs=torch.ones(8,4,dtype=torch.bool);obs[:6,0]=False
    future=torch.zeros(12,4,2);mask=torch.ones(12,4,dtype=torch.bool);mask[-1,1]=False
    return dict(observed=x,observed_valid=obs,future=future,future_valid=mask)


def test_complete_targets_but_context_never_uses_future_availability():
    s=sample();a=apply_protocol(s,PROTOCOLS[0]);z=copy.deepcopy(s);z['future'].fill_(999.);z['future_valid'].zero_();b=apply_protocol(z,PROTOCOLS[0])
    assert torch.equal(a['observed'],b['observed']) and torch.equal(a['observed_valid'],b['observed_valid'])
    assert a['observed_valid'][:,1].all() and not a['future_valid'][:,1].any()
    assert not a['observed_valid'][:,0].any() and not a['future_valid'][:,0].any()
    assert a['future_valid'][:,2:].all()


def test_B_view_preserves_archived_masks_exactly():
    s=sample();b=apply_protocol(s,PROTOCOLS[1]);assert all(torch.equal(s[k],b[k]) for k in s)


def test_groups_and_cv_invariant_to_future_gt_and_masks():
    s=sample();batch={k:v[None] for k,v in s.items()};batch['timestamps']=torch.arange(8,dtype=torch.float64)[None]*.4
    g,f=history_features(batch);z=copy.deepcopy(batch);z['future'].fill_(-1e6);z['future_valid'].zero_();h,v=history_features(z)
    assert all(torch.equal(g[k],h[k]) for k in g);assert all(torch.equal(f[k],v[k]) for k in f)
    model=HistoricalCV();assert torch.equal(model(batch['observed'],batch['observed_valid'],batch['timestamps']),model(z['observed'],z['observed_valid'],z['timestamps']))


def test_motion_groups_use_observed_direction_and_allow_overlap():
    s=sample();b={k:v[None] for k,v in s.items()};b['timestamps']=torch.arange(8,dtype=torch.float64)[None]*.4
    g,_=history_features(b);assert g['opposing'][0,1] and g['crossing'][0,1]
    assert g['history_full8'][0,1] and g['history_partial'][0,0]
    assert torch.equal(g['relationship_changing'].long()+g['relationship_stable'].long()+g['relationship_insufficient'].long(),g['all'].long())


def test_nonETH_recording_split_and_full_temporal_support():
    w=lambda r:dict(recording=r,scene='UNIV',start_frame=100,end_frame=290)
    m=dict(splits=dict(train=[w('students001')],validation=[w('students003')]))
    assert assert_recording_isolation(m)
    m['splits']['validation']=[w('students001')]
    with pytest.raises(ValueError):assert_recording_isolation(m)
    m['splits']['validation']=[dict(w('students003'),scene='ETH')]
    with pytest.raises(ValueError):assert_recording_isolation(m)


def test_track_mean_ADE_and_fixed_last_FDE_masks():
    p=torch.ones(1,12,2,2);y=torch.zeros_like(p);m=torch.ones(1,12,2,dtype=torch.bool);m[:,-1,1]=False
    ade,fde,a,f,count=target_errors(p,y,m);torch.testing.assert_close(ade,torch.full_like(ade,2**.5));assert a.all() and f.tolist()==[[True,False]]
    assert count.tolist()==[[12,11]]
