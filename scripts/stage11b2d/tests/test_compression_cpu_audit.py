import ast
import json
import numpy as np
import pandas as pd
import pytest
import torch
from scripts.stage11b2d.common import ROOT,OLD,OUT,CONF,sha,tensor_sha
from scripts.stage11b2d.geometry import assign_final,theory

def test_hand_computed_voronoi_ties_and_duplicate_draws():
    P=np.array([[0.,0.],[1.,0.],[1.,0.],[5.,0.],[2.,3.]])
    C=np.array([[0.,0.],[2.,0.],[6.,0.]])
    ids,d=assign_final(P,C,2)
    assert ids.tolist()==[0,0,0,2,1]
    np.testing.assert_allclose(d,[0,1,1,1,3],rtol=0,atol=0)
    assert np.bincount(ids,minlength=3).tolist()==[3,1,1]
    assert len(np.unique(P,axis=0))==4

@pytest.mark.parametrize('chunk',[1,7,512,20000])
def test_chunking_matches_independent_coordinate_distance(chunk):
    p=np.stack([np.arange(100)%13,np.arange(100)//13],axis=1)
    c=np.array([[1,2],[8,1],[2,6]])
    ids,d=assign_final(p,c,chunk)
    expected=np.array([[np.hypot(x[0]-z[0],x[1]-z[1]) for z in c] for x in p])
    np.testing.assert_array_equal(ids,expected.argmin(axis=1))
    np.testing.assert_allclose(d,expected.min(axis=1),rtol=1e-15,atol=1e-15)

def test_empty_and_invalid_geometry():
    ids,d=assign_final(np.empty((0,2)),np.array([[0.,0.]]))
    assert len(ids)==len(d)==0
    with pytest.raises(AssertionError):assign_final([[np.nan,0]],[[0,0]])

def test_theory_label_not_actual_experiment():
    assert theory(0)==0 and theory(1)==1
    assert abs(theory(1e-4)-(1-(1-1e-4)**10000))<1e-12

def test_frozen_cache_identity_and_original_k20_untouched():
    a=json.loads((OUT/'cache_audit.json').read_text())
    assert a['passed'] and a['targets']==998 and a['original20goals_paths_bitwise']
    assert not a['CUDA_initialized'] and a['GPU_inference']==a['training_steps']==0
    f=pd.read_csv(OUT/'target_diagnostics.csv');s=pd.read_csv(OLD/'sample_level_mechanism.csv')
    assert len(f)==f.sample_id.nunique()==998
    assert set(f.sample_id)==set(s.sample_id)
    assert f.category25.value_counts().to_dict()==s.category25.value_counts().to_dict()

def test_real_cluster_population_and_local_draw_conservation():
    f=pd.read_csv(OUT/'target_diagnostics.csv').set_index('sample_id')
    c=pd.read_csv(OUT/'cluster_diagnostics.csv')
    assert len(c)==998*19
    assert (c.groupby('sample_id').Population.sum()==10000).all()
    q=c.groupby('sample_id').Local25Count.sum()
    np.testing.assert_array_equal(q.to_numpy(),f.loc[q.index,'LocalProposalCount25'].to_numpy())
    local=pd.read_csv(OUT/'local_proposal_assignments.csv.gz')
    assert set(local.sample_id)==set(f.index[f.category25=='proposal_hit_final_miss'])
    assert (local.distance_GT<=25).all()
    assert local.groupby('sample_id').proposal_index.nunique().sum()==len(local)

def test_independent_real_frozen_case_assignment():
    sid=json.loads((CONF/'analysis_protocol.json').read_text())['case_ids'][1]
    row=pd.read_csv(OLD/'tensor_manifest.csv').set_index('sample_id').loc[sid]
    assert sha(ROOT/row.path)==row.sha256
    x=torch.load(ROOT/row.path,map_location='cpu',weights_only=True)['tensors']
    assert tensor_sha(x['centers_network'])==row.centers_SHA
    p=x['proposals_raw'][0,0].double();c=x['centers_raw'].double()
    # Independent Torch cdist, not the NumPy helper used in analysis.
    d=torch.cdist(p,c);labels=d.argmin(1).numpy()
    expected=pd.read_csv(OUT/'cluster_diagnostics.csv').query('sample_id == @sid').sort_values('center_index')
    np.testing.assert_array_equal(np.bincount(labels,minlength=19),expected.Population)

def test_zero_support_review_is_static_and_no_model_imports():
    z=pd.read_csv(OUT/'threshold_support_review.csv')
    assert len(z)==z.sample_id.nunique()==77
    assert (z.unthresholded_normalized_mass>0).all() and (z.actual_sampling_mass25==0).all()
    assert (z.positive_neighborhood_pixels==z.removed_positive_neighborhood_pixels).all()
    assert z.counterfactual_status.eq('ANALYTIC_COUNTERFACTUAL_ONLY').all()
    for name in ['analyze.py','geometry.py','prepare.py']:
        tree=ast.parse((ROOT/'scripts/stage11b2d'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                call=ast.unparse(node.func)
                assert not any(k in call for k in ['multinomial','cluster_proposals','sample_cpu','build_model','backward','optimizer'])
