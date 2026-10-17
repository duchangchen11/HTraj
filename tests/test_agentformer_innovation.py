import numpy as np
import pytest
from src.agentformer_innovation.metrics import fixed_subsets, history_features, motion_metrics, candidate_metrics, social_features


def test_subsets_are_nested_label_free_and_repeatable():
    assert fixed_subsets()==fixed_subsets()
    assert len(fixed_subsets())==21
    for a in fixed_subsets():
        assert sorted(a['heads'])==list(range(20))


def test_motion_speed_units_and_bias():
    x=np.stack([np.arange(8)*.4,np.zeros(8)],-1)
    y=x[-1]+np.stack([np.arange(1,13)*.4,np.zeros(12)],-1)
    m=motion_metrics(x,y,y)
    assert m['ADE']==0
    assert m['GT_speed_mps']==pytest.approx(1.)
    assert m['speed_bias_mps']==0
    assert history_features(x)['speed_group']=='normal'
    p=x[-1]+.5*(y-x[-1])
    assert motion_metrics(x,y,p)['speed_bias_mps']==pytest.approx(-.5)


def test_scale_control_and_static_fallback():
    x=np.zeros((8,2));y=np.zeros((12,2));p=np.zeros((12,2))
    m=motion_metrics(x,y,p)
    assert m['heading_error_rad'] is None
    assert history_features(x)['turn_group']=='unreliable'
    assert all(v is None or np.isfinite(v) for v in m.values())
    x=np.stack([np.arange(8)*.4,np.zeros(8)],-1);y=x[-1]+np.stack([np.arange(1,13)*.4,np.zeros(12)],-1);p=x[-1]+.8*(y-x[-1])
    assert motion_metrics(x,y,p)['ADE_over_GT_path_floor0.5']==pytest.approx(motion_metrics(x*2,y*2,p*2)['ADE_over_GT_path_floor0.5'])


def test_candidates_oracles_ties_and_translation_rotation():
    x=np.zeros((8,2));y=np.ones((12,2));p=np.repeat(y[None],20,0)
    a,f,ac,fc,s=candidate_metrics(x,y,p)
    assert np.all(a==0) and np.all(f==0)
    assert np.allclose(ac,.05) and np.allclose(fc,.05)
    assert s['near_pair_fraction']==1
    rng=np.random.default_rng(3);p=rng.normal(size=(20,12,2))
    r=np.array([[0,-1],[1,0]])
    one=candidate_metrics(x,y,p);two=candidate_metrics(x@r+8,y@r+8,p@r+8)
    assert np.allclose(one[0],two[0])
    assert one[-1]['path_pair_mean_m']==pytest.approx(two[-1]['path_pair_mean_m'])


def test_social_grouping_cannot_read_future():
    raw=np.array([[t,i,t*.4+i,0] for t in range(8) for i in (1,2)])
    before=social_features(raw,7,1)
    after=social_features(np.concatenate([raw,[[8,2,999,999],[9,8,999,999]]]),7,1)
    assert before==after
    assert before['history_neighbors_2m']==1
    assert before['local_relative_speed_mps']==pytest.approx(0)
    assert social_features(raw[raw[:,1]==1],7,1)['nearest_history_neighbor_m'] is None


def test_nonfinite_or_incomplete_rejected():
    with pytest.raises(ValueError): motion_metrics(np.zeros((7,2)),np.zeros((12,2)),np.zeros((12,2)))
    with pytest.raises(ValueError): motion_metrics(np.zeros((8,2)),np.full((12,2),np.nan),np.zeros((12,2)))
