"""Real-artifact scientific acceptance checks; no GPU/model call or old writes."""
import json
import numpy as np
import pandas as pd
import torch
from scripts.stage11b2b.common import ROOT,OUT,CACHE,OLD
from scripts.stage11b1c.data import truths,manifest

def test_paired_identity_and_GT_for_all_new_validations():
    ref=pd.DataFrame(manifest('validation')).set_index('sample_id')
    for ep in [35,40,45,50]:
        rows=pd.read_csv(OUT/f'paired_epoch{ep:03d}.csv').set_index('sample_id')
        assert len(rows)==998 and set(rows.index)==set(ref.index)
        for col in ['GT_SHA','history_SHA','person_id','frame_id']:assert (rows[col].astype(str)==ref.loc[rows.index,col].astype(str)).all()

def test_fixed_cohort_preserves_epoch30_identities():
    old=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv');ids=set(old.loc[old.minGoalFDE20>25,'sample_id']);fixed=pd.read_csv(OUT/'a25_fixed_cohort.csv')
    assert len(ids)==164
    for ep,rows in fixed.groupby('epoch'):assert len(rows)==164 and set(rows.sample_id)==ids and set(rows.fixed_cohort)=={'FIXED_A25_EPOCH30_COHORT'}

def test_best_selection_matches_all_complete_minADE_validations():
    candidates=[json.loads(p.read_text()) for base in [OLD,OUT] for p in base.glob('validation_epoch*.json')]
    valid=[r for r in candidates if r['status']=='COMPLETE'];best=min(valid,key=lambda r:(r['window']['minADE20'],r['epoch']))
    final=json.loads((OUT/'final_decision.json').read_text());assert final['global_best_epoch']==best['epoch']

def test_independent_torch_metric_recalculation_from_saved_coordinates():
    truth=truths('validation');summary=pd.read_csv(OUT/'validation_metrics.csv').set_index('epoch')
    for ep in [35,40,45,50]:
        pack=torch.load(CACHE/f'predictions/epoch{ep:03d}.pt',weights_only=True,map_location='cpu')
        y=torch.tensor(np.stack([truth[s] for s in pack['sample_ids']]),dtype=torch.float64)
        dist=torch.linalg.vector_norm(pack['paths'].double()-y[None],dim=-1);goal=torch.linalg.vector_norm(pack['goals'].double()-y[None,:,-1],dim=-1)
        values=dict(ADE1=dist[0].mean(),FDE1=dist[0,:,-1].mean(),minADE20=dist.mean(-1).min(0).values.mean(),minFDE20=dist[:,:,-1].min(0).values.mean(),GoalFDE1=goal[0].mean(),minGoalFDE20=goal.min(0).values.mean())
        for key,val in values.items():assert abs(float(val)-float(summary.loc[ep,key]))<1e-8

def test_case_coordinates_are_the_actual_cached_predictions():
    rows=pd.read_csv(OUT/'fixed_case_prediction_coordinates.csv.gz');cases=pd.read_csv(ROOT/'results/tables/stage11b2a/failure_case_manifest.csv');assert set(rows['case'])==set(cases['case'])
    for ep,r in rows.groupby('epoch'):
        cache=ROOT/f'experiments/{"stage11b1c" if ep==30 else "stage11b2b"}/cache/predictions/epoch{ep:03d}.pt'
        p=torch.load(cache,weights_only=True,map_location='cpu');index={sid:i for i,sid in enumerate(p['sample_ids'])}
        ix=np.array([index[sid] for sid in r.sample_id]);head=r['head'].to_numpy();step=r.future_step.to_numpy()-1
        assert np.allclose(r[['predicted_x','predicted_y']],p['paths'].numpy()[head,ix,step],rtol=0,atol=1e-8)
        assert np.allclose(r[['goal_x','goal_y']],p['goals'].numpy()[head,ix],rtol=0,atol=1e-8)
