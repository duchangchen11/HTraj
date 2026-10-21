"""Offline float64 geometry/probability definitions; never used for sampling."""
import numpy as np

def probability(weights):
    w=np.asarray(weights,dtype=np.float64);assert np.isfinite(w).all() and (w>=0).all() and w.sum()>0
    return w/w.sum(dtype=np.float64)

def theoretical_hit(mass,n=10000):
    assert 0<=mass<=1
    return 0. if mass==0 else (1. if mass==1 else float(-np.expm1(n*np.log1p(-mass))))

def classify(A,B,C):
    flags=dict(T1=not A,T2=A and not B,T3=B and not C,T4=B and C,T5=not B and C)
    flags['T6']=not any(flags.values())
    if not A and B:cat='other'
    elif B and C:cat='both_hit'
    elif B and not C:cat='proposal_hit_final_miss'
    elif not B and C:cat='proposal_miss_final_hit'
    elif not A:cat='zero_support_final_miss'
    elif A and not B:cat='positive_support_sampling_miss_final_miss'
    else:cat='other'
    return flags,cat

def neighborhood(p,sigmoid,threshold,endpoint,resize,radii=(10,25,50)):
    H,W=p.shape;yy,xx=np.indices((H,W));xy=np.stack([xx,yy],axis=-1).astype(float)/resize
    dist2=np.sum((xy-np.asarray(endpoint))**2,axis=-1);result={}
    for r in radii:
        mask=dist2<=r*r;mass=float(p[mask].sum(dtype=np.float64));assert -1e-14<=mass<=1+1e-14
        mass=min(1.,max(0.,mass))
        result[r]=dict(P_mass=mass,positive_support_pixels=int(((p>0)&mask).sum()),sigmoid_weight_sum=float(np.asarray(sigmoid,dtype=float)[mask].sum()),threshold_weight_sum=float(np.asarray(threshold,dtype=float)[mask].sum()),theoretical_P_hit=theoretical_hit(mass))
    return result

def coverage(points,endpoint,radii=(10,25,50)):
    d=np.linalg.norm(np.asarray(points,dtype=np.float64)-np.asarray(endpoint,dtype=np.float64),axis=-1)
    return float(d.min()),{r:dict(cover=bool((d<=r).any()),draws_in_neighborhood=int((d<=r).sum())) for r in radii}
