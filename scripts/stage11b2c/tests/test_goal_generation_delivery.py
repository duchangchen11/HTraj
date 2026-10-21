"""Acceptance against actual exports and independently recomputed geometry."""
import json
import numpy as np
import pandas as pd
import torch
from scripts.stage11b2c.common import ROOT,OUT,CACHE
from scripts.stage11b1c.data import truths

def test_all998_original_and_reverse_regression_complete():
    m=pd.read_csv(OUT/'tensor_manifest.csv');r=pd.read_csv(OUT/'all998_reverse_regression.csv')
    assert len(m)==len(r)==998 and m.sample_id.nunique()==r.sample_id.nunique()==998
    assert set(m.sample_id)==set(r.sample_id) and m.all_five_original_SHA.all() and r.all_tensors_bitwise.all()
    assert r.sample_id.tolist()==list(reversed(m.sample_id.tolist()))

def test_final_covers_are_exact_original_epoch30():
    f=pd.read_csv(OUT/'final_goal_coverage.csv').set_index('sample_id');old=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv').set_index('sample_id')
    assert len(f)==998 and set(f.index)==set(old.index)
    assert np.allclose(f.FinalMinDist20,old.loc[f.index,'minGoalFDE20'],rtol=0,atol=1e-8)
    for r,n in [(10,489),(25,164),(50,99)]:assert (~f[f'FinalCover{r}']).sum()==n

def test_independent_torch_mass_and_point_distance():
    m=pd.read_csv(OUT/'tensor_manifest.csv');mass=pd.read_csv(OUT/'probability_mass.csv').set_index('sample_id');prop=pd.read_csv(OUT/'proposal_coverage.csv').set_index('sample_id');gt=truths('validation')
    for i in [0,499,997]:
        row=m.iloc[i];pack=torch.load(ROOT/row.path,map_location='cpu',weights_only=True);t=pack['tensors'];H,W=t['sigmoid'].shape[-2:];p=t['normalized_weights'].double().reshape(H,W);p=p/p.sum();y=torch.tensor(gt[row.sample_id][-1],dtype=torch.float64)
        yy,xx=torch.meshgrid(torch.arange(H,dtype=torch.float64),torch.arange(W,dtype=torch.float64),indexing='ij');mask=((xx/.25-y[0])**2+(yy/.25-y[1])**2)<=25**2
        assert abs(float(p[mask].sum())-mass.loc[row.sample_id,'P_mass25'])<1e-12
        d=torch.linalg.vector_norm(t['proposals_raw'][0,0].double()-y,dim=-1)
        assert abs(float(d.min())-prop.loc[row.sample_id,'ProposalMinDist'])<1e-8
        assert int((d<=25).sum())==prop.loc[row.sample_id,'proposal_draws_within25']

def test_all_binary_combinations_and_primary_partition_counts():
    e=pd.read_csv(OUT/'coverage_transition.csv');c=pd.read_csv(OUT/'all_event_combinations.csv');s=pd.read_csv(OUT/'scene_mechanism_summary.csv')
    for r in [10,25,50]:
        q=e[e.radius==r];assert len(q)==998
        combos=c[(c.scope=='ALL_998')&(c.radius==r)];assert len(combos)==8 and combos.targets.sum()==998
        assert not ((~q.A)&q.B).any()
        fail=q[~q.C];assert len(fail)==int(s[(s.scope=='ALL_998')&(s.radius==r)].final_fail_count.iloc[0])
        assert len(fail)==int((~fail.A).sum()+(fail.A&~fail.B).sum()+fail.B.sum())

def test_cases_follow_frozen_rank_rule_and_no_new_GPU_training():
    f=pd.read_csv(OUT/'sample_level_mechanism.csv');cases=pd.read_csv(OUT/'mechanism_case_manifest.csv')
    expected=[]
    for cat,s in f.groupby('category25'):
        s=s.sort_values('sample_id',kind='stable');ranks=sorted({int(np.floor(q*(len(s)-1))) for q in [.25,.5,.75]})
        expected.extend(s.iloc[ranks].sample_id.tolist())
    assert sorted(cases.sample_id)==sorted(expected)
    g=json.loads((OUT/'generation_execution.json').read_text());assert g['training_steps']==0 and g['parameters_unchanged'] and not g['test_loaded']
