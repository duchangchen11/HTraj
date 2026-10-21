"""Position-only descriptive features. Undefined directions stay NaN."""
import numpy as np
DT=.4
def motion(xy,minimum_direction_step=1.):
    x=np.asarray(xy,dtype='float64');d=np.diff(x,axis=0);length=np.linalg.norm(d,axis=-1);v=d/DT;speed=length/DT
    total=float(length.sum());net=float(np.linalg.norm(x[-1]-x[0]));n=len(speed)
    valid=(length[:-1]>=minimum_direction_step)&(length[1:]>=minimum_direction_step)
    denom=length[:-1]*length[1:];angles=np.full(max(0,n-1),np.nan)
    angles[valid]=np.degrees(np.arccos(np.clip(np.sum(d[:-1][valid]*d[1:][valid],axis=-1)/denom[valid],-1,1)))
    aa=angles[np.isfinite(angles)];mean=float(speed.mean());std=float(speed.std())
    cv=std/mean if mean>=minimum_direction_step/DT else np.nan
    return dict(net_displacement=net,path_length=total,mean_speed=mean,last2_speed=float(speed[-1]),speed_variance=float(speed.var()),
        speed_std=std,speed_CV=cv,mean_speed_change=float(np.abs(np.diff(speed)).mean()) if n>1 else np.nan,
        mean_acceleration=float(np.linalg.norm(np.diff(v,axis=0)/DT,axis=-1).mean()) if n>1 else np.nan,
        mean_turn_degrees=float(aa.mean()) if len(aa) else np.nan,max_turn_degrees=float(aa.max()) if len(aa) else np.nan,
        total_turn_degrees=float(aa.sum()) if len(aa) else np.nan,
        curvature_radians_per_pixel=float(np.radians(aa).sum()/total) if len(aa) and total>1e-12 else np.nan,valid_turn_pairs=int(valid.sum()),
        reliable_motion_steps=int((length>=minimum_direction_step).sum()),
        straightness=net/total if total>1e-12 else np.nan,stationary_step_fraction=float((length<=minimum_direction_step).mean()),
        all_steps_stationary=bool((length<=minimum_direction_step).all()),observed_positions=len(x))
def turn_group(f):
    if not np.isfinite(f['max_turn_degrees']):return 'undefined_direction'
    if f['max_turn_degrees']>=30:return 'turning_ge30deg'
    if f['max_turn_degrees']<=15 and f['straightness']>=.95:return 'straight'
    return 'intermediate'
def speed_change_group(f):
    if not np.isfinite(f['speed_CV']):return 'undefined_low_speed'
    if f['speed_CV']<=.15:return 'stable_CVle0.15'
    if f['speed_CV']>=.4:return 'variable_CVge0.4'
    return 'intermediate'
def tercile(x,cuts):return 'low' if x<cuts[0] else ('medium' if x<cuts[1] else 'high')
def stats(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return dict(valid_n=0,mean=None,median=None,p10=None,p25=None,p75=None,p90=None,p95=None)
    return dict(valid_n=len(a),mean=float(a.mean()),median=float(np.median(a)),**{f'p{q}':float(np.percentile(a,q,method='linear')) for q in [10,25,75,90,95]})
def spearman(x,y):
    import pandas as pd
    a=np.asarray(x,dtype=float);b=np.asarray(y,dtype=float);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.unique(a[ok]).size<2 or np.unique(b[ok]).size<2:return np.nan,int(ok.sum())
    return float(np.corrcoef(pd.Series(a[ok]).rank(method='average'),pd.Series(b[ok]).rank(method='average'))[0,1]),int(ok.sum())
