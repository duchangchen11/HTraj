"""Read-only offline comparisons; no functions used to generate predictions."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scripts.stage11b2e.metrics import path_metrics,dispersion

def distribution_stats(weights):
    p=np.asarray(weights,dtype=np.float64).ravel();assert np.isfinite(p).all() and (p>=0).all() and p.sum()>0
    p=p/p.sum();positive=p[p>0];entropy=float(-np.sum(positive*np.log(positive)))
    return p,dict(entropy_nats=entropy,support_pixels=int((p>0).sum()),support_area_raw_px2=int((p>0).sum())*16,
        exp_entropy_pixels=float(np.exp(entropy)))

def compare_distribution(original,alternative):
    p,os=distribution_stats(original['normalized_weights']);q,ns=distribution_stats(alternative['normalized_weights'])
    assert p.shape==q.shape;m=(p+q)/2;ip=p>0;iq=q>0
    js=float(.5*np.sum(p[ip]*np.log(p[ip]/m[ip]))+.5*np.sum(q[iq]*np.log(q[iq]/m[iq])))
    dx=np.asarray(alternative['logits'],dtype=float)-np.asarray(original['logits'],dtype=float)
    return dict(JS_nats=max(0.,js),total_variation=float(np.abs(p-q).sum()/2),
        logits_mean_abs_difference=float(np.abs(dx).mean()),logits_RMS_difference=float(np.sqrt(np.mean(dx*dx))),
        logits_max_abs_difference=float(np.abs(dx).max()),
        **{'Original_'+k:v for k,v in os.items()},**{'Alternative_'+k:v for k,v in ns.items()})

def compare_goals(original,alternative):
    a=np.asarray(original,dtype=float);b=np.asarray(alternative,dtype=float);assert a.shape==b.shape==(20,2)
    d=cdist(a,b);i,j=linear_sum_assignment(d)
    return dict(Hungarian_mean_distance=float(d[i,j].mean()),Hungarian_max_distance=float(d[i,j].max()),
        symmetric_Chamfer_distance=float((d.min(axis=0).mean()+d.min(axis=1).mean())/2),
        Hausdorff_distance=float(max(d.min(axis=0).max(),d.min(axis=1).max())),
        Softargmax_displacement=float(np.linalg.norm(a[0]-b[0])),
        all20_goal_coordinates_bitwise_unchanged=bool(np.array_equal(original,alternative)))

def prediction_metrics(paths,goals,gt):
    metrics,ade,fde=path_metrics(paths,gt);metrics.pop('ADE_best_head');metrics.pop('FDE_best_head')
    d=np.linalg.norm(np.asarray(goals,dtype=float)-np.asarray(gt[-1],dtype=float),axis=-1)
    metrics.update(GoalFDE1=float(d[0]),minGoalFDE20=float(d.min()),GoalPairwiseDispersion=dispersion(goals),
        GoalPathEndpointMeanDistance=float(np.linalg.norm(np.asarray(goals,dtype=float)-np.asarray(paths,dtype=float)[:,-1],axis=-1).mean()),
        UniqueGoals=len(np.unique(goals,axis=0)))
    for r in [10,25,50]:metrics[f'GoalCover{r}']=bool(d.min()<=r)
    return metrics,ade,fde,d

def compare_fixed_paths(original,alternative):
    d=np.linalg.norm(np.asarray(alternative,dtype=float)-np.asarray(original,dtype=float),axis=-1)
    changed=np.any(np.asarray(original)!=np.asarray(alternative),axis=(1,2))
    out=dict(Path_point_mean_displacement=float(d.mean()),Path_endpoint_mean_displacement=float(d[:,-1].mean()),
        Path_max_displacement=float(d.max()),Path_changed_heads=int(changed.sum()),Path_unchanged_heads=int((~changed).sum()),
        Head0_point_mean_displacement=float(d[0].mean()),Head0_endpoint_displacement=float(d[0,-1]),Head0_max_displacement=float(d[0].max()))
    for v,label in [(1e-4,'1e_4'),(1.,'1'),(5.,'5'),(25.,'25')]:out['Path_heads_mean_displacement_gt'+label]=int((d.mean(axis=1)>v).sum())
    return out
