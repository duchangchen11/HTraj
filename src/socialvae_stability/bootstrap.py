"""Paired person-cluster bootstrap; overlapping windows never resampled alone."""
import hashlib
import numpy as np


def bootstrap_seed(recording,training_seed,base_seed=20261016):
    key=f'stage5p5-cluster-bootstrap-v1|base{base_seed}|{recording}|{training_seed}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63)


def cluster_errors(person_ids,ade_deltas,fde_deltas):
    ids=np.asarray(person_ids);a=np.asarray(ade_deltas,dtype=float);f=np.asarray(fde_deltas,dtype=float)
    if len(ids)!=len(a) or len(ids)!=len(f) or len(ids)==0:raise ValueError('Nonempty paired target rows required')
    if not np.isfinite(a).all() or not np.isfinite(f).all():raise ValueError('Nonfinite error')
    people=sorted(set(ids.tolist()))
    sums=np.array([[a[ids==p].sum(),f[ids==p].sum()] for p in people])
    counts=np.array([(ids==p).sum() for p in people])
    return people,sums,counts


def draws_to_differences(sums,counts,draws):
    """Each draw is a list of selected person indices including multiplicities."""
    sums=np.asarray(sums,dtype=float);counts=np.asarray(counts,dtype=float);draws=np.asarray(draws,dtype=int)
    if (counts<=0).any() or draws.ndim!=2 or draws.shape[1]!=len(counts):raise ValueError('Invalid cluster draws')
    if (draws<0).any() or (draws>=len(counts)).any():raise ValueError('Invalid cluster index')
    window=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
    person=(sums/counts[:,None])[draws].mean(axis=1)
    return window,person


def paired_bootstrap(person_ids,ade_deltas,fde_deltas,recording,training_seed,replicates=2000):
    people,sums,counts=cluster_errors(person_ids,ade_deltas,fde_deltas)
    seed=bootstrap_seed(recording,training_seed)
    draws=np.random.default_rng(seed).integers(0,len(people),size=(replicates,len(people)))
    window,person=draws_to_differences(sums,counts,draws)
    return dict(people=people,sums=sums,counts=counts,seed=seed,draws=draws,window=window,person=person)
