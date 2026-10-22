"""Static float64 measurements only. Never change or generate any center."""
import numpy as np

def assign_final(points,centers,chunk=512):
    p=np.asarray(points,dtype=np.float64);c=np.asarray(centers,dtype=np.float64)
    assert p.ndim==c.ndim==2 and p.shape[1]==c.shape[1]==2 and len(c)>0 and chunk>0
    assert np.isfinite(p).all() and np.isfinite(c).all()
    ids=np.empty(len(p),dtype=np.int64);dist=np.empty(len(p),dtype=np.float64)
    for first in range(0,len(p),chunk):
        d2=((p[first:first+chunk,None,:]-c[None,:,:])**2).sum(axis=-1)
        a=np.argmin(d2,axis=1);ids[first:first+len(a)]=a
        dist[first:first+len(a)]=np.sqrt(d2[np.arange(len(a)),a])
    return ids,dist

def stats(a):
    a=np.asarray(a,dtype=float)
    if not len(a):return dict(mean=None,median=None,p90=None,max=None)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def rms(points):
    p=np.asarray(points,dtype=float)
    return float(np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2,axis=-1)))) if len(p) else None

def theory(m,n=10000):
    assert 0<=m<=1
    return 0. if m==0 else (1. if m==1 else float(-np.expm1(n*np.log1p(-m))))
