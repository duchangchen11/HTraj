"""Conventional fixed-cell squared medoid; input API has no future argument."""
import numpy as np
import time

def assign(points,centers,chunk=512):
    p=np.asarray(points,dtype=np.float64);c=np.asarray(centers,dtype=np.float64)
    assert p.ndim==c.ndim==2 and p.shape[1]==c.shape[1]==2 and len(c)>0
    assert np.isfinite(p).all() and np.isfinite(c).all()
    a=np.empty(len(p),dtype=np.int64)
    for first in range(0,len(p),chunk):
        a[first:first+chunk]=np.argmin(np.sum((p[first:first+chunk,None,:]-c[None,:,:])**2,axis=-1),axis=1)
    return a

def squared_medoid(points,indices):
    idx=np.asarray(indices,dtype=np.int64)
    if not len(idx):return -1,None
    x=np.asarray(points)[idx].astype(np.float64);mean=x.mean(axis=0,dtype=np.float64)
    d2=((x-mean)**2).sum(axis=-1)
    winner=int(idx[d2==d2.min()].min())
    return winner,mean

def select(points,centers,softargmax):
    """Keep draw multiplicities and choose the first original index on an exact tie."""
    start=time.perf_counter();P=np.asarray(points);C=np.asarray(centers);soft=np.asarray(softargmax)
    assert P.shape==(10000,2) and C.shape==(19,2) and soft.shape==(2,)
    assert P.dtype==C.dtype==soft.dtype==np.float32
    a=assign(P,C);assignment_seconds=time.perf_counter()-start;t=time.perf_counter()
    goals=np.concatenate([soft[None].copy(),C.copy()],axis=0);rows=[]
    for k in range(19):
        idx=np.flatnonzero(a==k);x=P[idx].astype(np.float64)
        if len(idx):
            j,mean=squared_medoid(P,idx)
            goals[k+1]=P[j]  # Copy existing original float32 coordinates, no rounding.
            value=float(np.sum((x-P[j].astype(np.float64))**2))
            original=float(np.sum((x-C[k].astype(np.float64))**2))
            mu=mean.tolist()
        else:j=-1;value=original=None;mu=[None,None]
        rows.append(dict(center_index=k,head=k+1,population=len(idx),proposal_index=j,
            old_x=float(C[k,0]),old_y=float(C[k,1]),new_x=float(goals[k+1,0]),new_y=float(goals[k+1,1]),
            mean_x=mu[0],mean_y=mu[1],changed=not np.array_equal(C[k],goals[k+1]),
            displacement=float(np.linalg.norm(goals[k+1].astype(float)-C[k].astype(float))),
            empty_fallback=not bool(len(idx)),original_fixed_cell_squared_loss=original,
            alternative_fixed_cell_squared_loss=value))
    assert np.array_equal(goals[0],soft) and np.isfinite(goals).all()
    return goals,a,rows,dict(assignment_seconds=assignment_seconds,representative_seconds=time.perf_counter()-t,total_seconds=time.perf_counter()-start)

def project_record(record):
    """Deliberate whitelist used for raw-record future counterfactual tests."""
    t=record['tensors']
    return record['sample_id'],t['proposals_raw'][0,0].numpy(),t['centers_raw'].numpy(),t['softargmax_raw'].numpy()
