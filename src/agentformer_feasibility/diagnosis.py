"""Historical grouping and transparent metrics; no learned selector."""
import numpy as np


def cv_predictions(history,dt=.4):
    x=np.asarray(history,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(8,2) or not np.isfinite(x).all():raise ValueError('Complete N,8,2 history required')
    result={}
    for name,n in [('CV-Last2',2),('CV-OLS4',4),('CV-OLS8',8)]:
        t=np.arange(n)*dt;t=t-t.mean()
        v=(x[:,-n:]*t[None,:,None]).sum(1)/(t*t).sum()
        result[name]=x[:,-1,None,:]+v[:,None,:]*np.arange(1,13)[None,:,None]*dt
    return result


def history_labels(history):
    x=np.asarray(history,dtype=np.float64)
    speed=np.linalg.norm(np.diff(x,axis=1),axis=-1).mean(1)/.4
    bins=np.digitize(speed,[.05,.4,1.8])
    return speed,np.asarray(['stationary','low','normal','fast'])[bins]


def context_bin(other_people):
    if other_people==0:return 'none'
    if other_people<=3:return '1-3'
    if other_people<=9:return '4-9'
    return '10+'


def candidate_errors(prediction,truth):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(truth,dtype=np.float64)
    if p.shape!=(len(y),20,12,2) or y.shape[1:]!=(12,2) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid K20 trajectories')
    errors=np.linalg.norm(p-y[:,None],axis=-1)
    ade=errors.mean(-1);fde=errors[:,:,-1]
    ai,fi=ade.argmin(1),fde.argmin(1)
    endpoints=p[:,:,-1]
    rms=np.sqrt(np.square(endpoints-endpoints.mean(1,keepdims=True)).sum(-1).mean(1))
    ii,jj=np.triu_indices(20,1)
    pair=np.linalg.norm(endpoints[:,ii]-endpoints[:,jj],axis=-1).mean(1)
    return dict(minADE20=ade.min(1),minFDE20=fde.min(1),K1_ADE=ade[:,0],K1_FDE=fde[:,0],
        ADE_winner=ai,FDE_winner=fi,ADE_best_step=errors[np.arange(len(y)),ai],
        FDE_best_step=errors[np.arange(len(y)),fi],first_step=errors[:,0],endpoint_RMS=rms,endpoint_pairwise_mean=pair)


def step_summary(rows,warmup=10):
    a=np.array([r['step_seconds'] for r in rows],dtype=float)
    steady=a[warmup:]
    return dict(updates=len(rows),all_step_mean_seconds=float(a.mean()),P50_seconds=float(np.quantile(a,.5)),
        P90_seconds=float(np.quantile(a,.9)),P99_seconds=float(np.quantile(a,.99)),
        steady_warmup_updates_excluded=warmup,steady_step_count=len(steady),
        steady_mean_seconds=float(steady.mean()) if len(steady) else None)


def history_neighbor_counts(raw,frame,ids,radius=2.):
    """Diagnostic counts only: full8 existence AND radius depend on past/current rows only."""
    by_frame={t:{} for t in range(frame-7,frame+1)}
    for t,i,x,y in np.asarray(raw):
        if int(t) in by_frame:by_frame[int(t)][int(i)]=np.array([x,y])
    complete=set.intersection(*(set(v) for v in by_frame.values()))
    current=by_frame[frame]
    if not set(map(int,ids))<=complete:raise ValueError('Target lacks full history')
    local=np.array([sum(j!=int(i) and np.linalg.norm(current[j]-current[int(i)])<=radius for j in complete) for i in ids])
    return local,len(complete)
