"""Read-only float64 metrics; generation never imports this module."""
import numpy as np

def min_d2(points,centers,chunk=512):
    p=np.asarray(points,dtype=float);c=np.asarray(centers,dtype=float);d=np.empty(len(p))
    for i in range(0,len(p),chunk):d[i:i+chunk]=((p[i:i+chunk,None]-c[None])**2).sum(axis=-1).min(axis=1)
    return d

def dispersion(x):
    x=np.asarray(x,dtype=float)
    d=np.linalg.norm(x[:,None]-x[None],axis=-1)
    if x.ndim==3:d=d.mean(axis=-1)
    return float(d[np.triu_indices(len(x),1)].mean())

def path_metrics(paths,gt):
    x=np.asarray(paths,dtype=float);y=np.asarray(gt,dtype=float);d=np.linalg.norm(x-y[None],axis=-1)
    ade=d.mean(axis=1);fde=d[:,-1]
    return dict(ADE1=float(ade[0]),FDE1=float(fde[0]),minADE20=float(ade.min()),minFDE20=float(fde.min()),
        ADE_best_head=int(ade.argmin()),FDE_best_head=int(fde.argmin()),PathPairwiseDispersion=dispersion(x),
        UniquePaths=len(np.unique(np.asarray(paths).reshape(20,-1),axis=0))),ade,fde

def goal_metrics(goals,endpoint,proposals,weights,radii=(10,25,50)):
    g=np.asarray(goals,dtype=float);p=np.asarray(proposals,dtype=float);d=np.linalg.norm(g-np.asarray(endpoint),axis=-1)
    original=np.asarray(weights,dtype=float);H,W=original.shape;prob=original.ravel()/original.sum()
    support=np.flatnonzero(prob>0);xy=np.stack([support%W,support//W],axis=-1)/.25
    min_support=min_d2(xy,g)
    out=dict(minGoalFDE20=float(d.min()),GoalFDE1=float(d[0]),Goal_best_head=int(d.argmin()),
        GoalPairwiseDispersion=dispersion(g),UniqueGoals=len(np.unique(np.asarray(goals),axis=0)),
        Distortion19=float(min_d2(p,g[1:]).mean()),Distortion20=float(min_d2(p,g).mean()))
    for r in radii:
        out[f'GoalCover{r}']=bool(d.min()<=r)
        out[f'ProbabilityUnion{r}']=float(prob[support[min_support<=r*r]].sum())
    return out,d

def gate(original_macro,alternative_macro,original_other4,alternative_other4):
    gain=alternative_macro['GoalCover25']-original_macro['GoalCover25']
    ade=(original_macro['minADE20']-alternative_macro['minADE20'])/original_macro['minADE20']
    fde=(alternative_macro['minFDE20']-original_macro['minFDE20'])/original_macro['minFDE20']
    other=alternative_other4['GoalCover25']-original_other4['GoalCover25']
    tests=dict(macro_GoalCover25_gain_ge1pp=bool(gain>=.01),macro_minADE20_drop_ge1pct=bool(ade>=.01),
        macro_minFDE20_degradation_le1pct=bool(fde<=.01),other4_GoalCover25_strict_gain=bool(other>0))
    return dict(status='CONTROL_GO' if all(tests.values()) else 'CONTROL_NO_GO',conditions=tests,
        measurements=dict(macro_GoalCover25_gain=gain,macro_minADE20_relative_drop=ade,macro_minFDE20_relative_degradation=fde,other4_GoalCover25_gain=other))
