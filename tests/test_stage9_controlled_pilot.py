"""Stage9 data/metric/budget contracts; GPU structure results are tested separately."""
import csv,gzip,json,hashlib,math
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tables/stage9'

def load(name):return json.loads((OUT/name).read_text())

def test_complete_cohorts_and_no_test_recording():
    d=load('data_manifest.json')
    assert d['splits']['train']['targets']==29676
    assert d['splits']['val']['targets']==5203
    assert d['splits']['train']['max_context']==57
    assert len(d['files'])==14
    assert all('biwi_hotel' not in n and n.endswith(('_train','_val')) for n in d['files'])
    for split in ['train','val']:
        p=OUT/(split+'_targets.csv.gz')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==d['splits'][split]['manifest_SHA']
        with gzip.open(p,'rt') as f:rows=list(csv.DictReader(f))
        keys=[(r['sequence'],r['frame'],r['person_id']) for r in rows]
        assert len(keys)==len(set(keys))==d['splits'][split]['targets']
        assert all(len(r['GT_SHA'])==len(r['history_SHA'])==64 for r in rows)

def test_source_support_isolation():
    d=load('source_isolation.json');assert d['passed']
    assert len(d['sources'])==7
    for r in d['sources']:
        assert r['train_max']<r['val_min']
        assert r['shared_frames']==r['shared_frame_actor']==0

def test_aggregation_structure_admission():
    d=load('structure_tests.json');assert d['passed'] and d['shared_initial_tensors_identical']
    assert [r['parameters'] for r in d['cases']]==[3149192,3157384,3157384]
    for r in d['cases']:
        assert r['strict_restore_exact'] and r['empty_neighbor_finite']
        assert [c['N'] for c in r['cases']]==[1,2,5,57]
        assert all(c['shape']==[2,20,12,2] for c in r['cases'])
    assert d['pairing_witness'][0]['aggregation_max_difference']<1e-5
    assert all(r['aggregation_max_difference']>1e-5 for r in d['pairing_witness'][1:])

def test_frozen_budget_before_validation_accuracy():
    t=load('timing.json');b=load('budget_estimate.json')
    assert t['validation_accuracy_seen'] is False
    assert len(t['models'])==3 and all(r['updates']==50 for r in t['models'])
    passing=[r['epochs'] for r in b['alternatives'] if r['fits']]
    assert b['chosen_epochs']==max(passing)
    assert all(r['safety_factor']>=1.2 for r in b['alternatives'])
    cfg=json.loads((ROOT/'configs/stage9/training.json').read_text())
    assert cfg['epochs']==b['chosen_epochs'] and cfg['batch_size']==128
    assert cfg['seed']==42 and cfg['new_parameter_count_B']==cfg['new_parameter_count_C']==8192

def test_paired_metric_not_per_timestep_oracle():
    # Each trajectory must be assessed as a whole; ADE and FDE winners can differ.
    distance=np.array([[0.,0.,3.],[1.5,1.5,0.]])
    assert distance.mean(-1).min()==1.
    assert distance[:,-1].min()==0.
    assert distance.min(0).mean()==0.  # invalid time-wise mixing would yield this

def test_low_rank_cross_moment_and_permutation():
    g=np.array([[1.,2.],[3.,4.],[2.,1.]])
    h=np.array([[2.,0.],[0.,1.],[1.,3.]])
    u=np.array([[.4,1.],[1.,-.3]]);v=np.array([[.2,1.],[.7,-.2]])
    projected=np.diag(u.T@((g-g.mean(0)).T@(h-h.mean(0))/3)@v)
    actual=((g@u)*(h@v)).mean(0)-(g@u).mean(0)*(h@v).mean(0)
    np.testing.assert_allclose(projected,actual,atol=1e-12)
    order=[2,0,1]
    np.testing.assert_allclose(actual,((g[order]@u)*(h[order]@v)).mean(0)-(g[order]@u).mean(0)*(h[order]@v).mean(0),atol=1e-12)
