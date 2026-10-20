"""History-only K1 baselines for all998 fixed development targets.

Generate every baseline path before accessing future coordinates for scoring.
OLS slopes use the real .4s interval and anchor predictions at last observation.
No Y-Net validation forward or training is performed by this script.
"""
from .common import *
from .data import load,manifest
from scripts.stage11b0.data_audit import canonical_sha
import numpy as np

def predict(history, method):
    h=np.asarray(history,dtype=np.float64)
    if h.shape!=(8,2) or not np.isfinite(h).all():raise ValueError('Complete finite history8 required')
    if method=='CV-Last2':v=(h[-1]-h[-2])/.4
    elif method in {'OLS4','OLS8'}:
        n=int(method[-1]);t=np.arange(n,dtype=float)*.4;t-=t.mean()
        v=(t[:,None]*(h[-n:]-h[-n:].mean(0))).sum(0)/(t*t).sum()
    else:raise ValueError(method)
    return h[-1]+np.arange(1,13)[:,None]*.4*v

def main():
    start=time.perf_counter();df=load('validation');rows=manifest('validation')
    groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
    predictions=[];arrays={}
    methods=['CV-Last2','OLS4','OLS8']
    # Phase1: no access to any future coordinates, no error calculation.
    for r in rows:
        history=groups[int(r['meta_id'])][['x','y']].iloc[:8].to_numpy(dtype=float)
        assert canonical_sha(history)==r['history_SHA']
        for method in methods:
            path=predict(history,method);arrays[(r['sample_id'],method)]=path
            for t,(x,y) in enumerate(path):
                predictions.append(dict(sample_id=r['sample_id'],model=method,step=t+1,x=x,y=y))
    cnew(OUT/'baseline_predictions.csv.gz',predictions)
    assert len(arrays)==998*3
    # Phase2: all predictions exist; now read GT and score paired identities.
    paired=[]
    for r in rows:
        truth=groups[int(r['meta_id'])][['x','y']].iloc[8:].to_numpy(dtype=float)
        assert canonical_sha(truth)==r['GT_SHA']
        for method in methods:
            d=np.linalg.norm(arrays[(r['sample_id'],method)]-truth,axis=-1)
            paired.append(dict(sample_id=r['sample_id'],scene=r['scene'],person_id=r['person_id'],
                frame_id=r['frame_id'],history_SHA=r['history_SHA'],GT_SHA=r['GT_SHA'],model=method,
                ADE1=float(d.mean()),FDE1=float(d[-1]),unit='original reference pixels'))
    cnew(OUT/'baseline_paired_errors.csv',paired)
    aggregate=[];scene_rows=[]
    for m in methods:
        a=[r for r in paired if r['model']==m]
        for s in sorted({r['scene'] for r in a}):
            b=[r for r in a if r['scene']==s]
            scene_rows.append(dict(model=m,scene=s,targets=len(b),ADE1=float(np.mean([r['ADE1'] for r in b])),
                                   FDE1=float(np.mean([r['FDE1'] for r in b])),unit='original reference pixels'))
        for weighting in ['target_window','recording_macro']:
            b=a if weighting=='target_window' else [r for r in scene_rows if r['model']==m]
            aggregate.append(dict(model=m,status='COMPLETE',weighting=weighting,targets=998,K=1,
                ADE1=float(np.mean([r['ADE1'] for r in b])),FDE1=float(np.mean([r['FDE1'] for r in b])),
                minADE20='',minFDE20='',GoalFDE1='',minGoalFDE20='',unit='original reference pixels'))
    for weighting in ['target_window','recording_macro']:
        aggregate.append(dict(model='Y-Net',status='NOT_RUN_EVALUATOR_GATE_FAILED',weighting=weighting,targets=998,K=20,
                              ADE1='',FDE1='',minADE20='',minFDE20='',GoalFDE1='',minGoalFDE20='',unit='original reference pixels'))
    cnew(OUT/'baseline_comparison.csv',aggregate)
    cnew(OUT/'baseline_scene_metrics.csv',scene_rows)
    jnew(OUT/'baseline_execution.json',dict(status='COMPLETE',targets=998,methods=methods,
         CPU_wall_seconds=time.perf_counter()-start,validation_manifest_SHA=sha(OUT/'validation_target_manifest.csv'),
         prediction_SHA=sha(OUT/'baseline_predictions.csv.gz'),prediction_completed_before_scoring=True,
         test_loaded=False,YNet_validation_prediction_run=False,
         formulas={'CV-Last2':'v=(x7-x6)/0.4; y[k]=x7+v*k*0.4',
                   'OLS4/8':'centered least-squares slope over last n history times; y[k]=x7+v*k*0.4'},
         GT_used_only_for='Identity check and offline metric after all paths completed'))
    print('Three K1 baselines complete on same998 validation identities; no Y-Net metric',flush=True)

if __name__=='__main__':main()
