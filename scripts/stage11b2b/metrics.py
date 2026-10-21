"""Frozen descriptive criteria; no model input or selection by goal coverage."""
import numpy as np

def verdict(fraction,share,new_complete):
    if not new_complete:return 'NOT_EVALUATED'
    if fraction>=.10 and share>=.50:return 'H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET'
    if fraction<.05 and share<.25:return 'H1_WEAKENED_WITHIN_BUDGET'
    return 'INCONCLUSIVE'

def candidate_metrics(paths,y,goals):
    p=np.asarray(paths,dtype=np.float64);y=np.asarray(y,dtype=np.float64);g=np.asarray(goals,dtype=np.float64)
    assert p.shape==(20,len(y),12,2) and g.shape==(20,len(y),2)
    assert all(np.isfinite(a).all() for a in [p,y,g])
    d=np.linalg.norm(p-y[None],axis=-1);ade=d.mean(-1);fde=d[:,:,-1];ge=np.linalg.norm(g-y[None,:,-1],axis=-1)
    gap=np.linalg.norm(g-p[:,:,-1],axis=-1);i,j=np.triu_indices(20,1);cd=np.linalg.norm(p[i]-p[j],axis=-1)
    m=dict(ADE1=ade[0],FDE1=fde[0],minADE20=ade.min(0),minFDE20=fde.min(0),GoalFDE1=ge[0],minGoalFDE20=ge.min(0),
        best_ADE_head=ade.argmin(0),best_FDE_head=fde.argmin(0),goal_path_gap_mean20=gap.mean(0),
        endpoint_diversity=cd[:,:,-1].mean(0),trajectory_diversity=cd.mean(-1).mean(0),
        GOAL_PATH_MISMATCH=((ge<=10)&(ade>=25)).any(0),SelectionGap_ADE=ade[0]-ade.min(0),SelectionGap_FDE=fde[0]-fde.min(0))
    return d,ade,fde,ge,gap,m

def aggregate(f):
    out=dict(targets=len(f))
    for k in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','endpoint_diversity','trajectory_diversity','goal_path_gap_mean20']:
        out[k]=float(f[k].mean()) if len(f) else None
    for t in [10,25,50]:
        miss=f.minGoalFDE20>t
        out[f'A{t}_miss_count']=int(miss.sum());out[f'A{t}_miss_fraction']=float(miss.mean()) if len(f) else None
    out['A25_error_share']=float(f.loc[f.minGoalFDE20>25,'minADE20'].sum()/f.minADE20.sum()) if len(f) and f.minADE20.sum()>0 else None
    for k in ['ADE','FDE']:
        x=f['SelectionGap_'+k]
        for stat,fn in [('mean',np.mean),('median',np.median),('P90',lambda x:np.percentile(x,90))]:out['SelectionGap_'+k+'_'+stat]=float(fn(x)) if len(f) else None
    out['GoalPathMismatch_count']=int(f.GOAL_PATH_MISMATCH.sum())
    out['first_ADE_best_fraction']=float((f.best_ADE_head==0).mean()) if len(f) else None
    out['ADE_FDE_same_head_fraction']=float((f.best_ADE_head==f.best_FDE_head).mean()) if len(f) else None
    return out
