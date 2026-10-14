"""Exact unweighted timed-walk identity counts, without enumerating time tuples."""
import numpy as np


def path_identity_counts(adjacency, continuity, mode='strict'):
    """Return per-cutoff-target statistics. Waiting is not another path event.

    q2[end, origin, middle] retains identity and counts distinct edge-time tuples.
    For a->b->c->d, repeats are a==c, b==d, or a==d (no adjacent self edges).
    q3 tracks these exact disjoint/union categories with O(N^3) memory.
    Learned alpha/carry magnitude is deliberately absent: this is topology only.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    continuity = np.asarray(continuity, dtype=bool)
    if mode not in ('strict','nondecreasing'):
        raise ValueError('Unknown temporal mode')
    steps,n,_ = adjacency.shape
    if continuity.shape != (steps,n) or np.diagonal(adjacency,axis1=1,axis2=2).any():
        raise ValueError('Invalid graph/continuity')
    first = np.zeros((n,n),dtype=np.int64)
    second = np.zeros((n,n,n),dtype=np.int64)
    third = np.zeros((4,n),dtype=np.int64)  # total, backtrack, end return, all distinct
    for t in range(steps):
        a = adjacency[t] & continuity[t][:,None] & continuity[t][None,:]
        old1 = first*continuity[t][:,None]
        old2 = second*continuity[t][:,None,None]
        old3 = third*continuity[t][None,:]
        new1 = a.astype(np.int64)
        source1 = old1 if mode=='strict' else old1+new1
        new2 = a[:,None,:]*source1.T[None,:,:]
        source2 = old2 if mode=='strict' else old2+new2
        new3 = np.zeros_like(third)
        for d,c in zip(*a.nonzero()):
            paths = source2[c]
            total = paths.sum()
            backtrack = paths[c,:].sum()+paths[:,d].sum()-paths[c,d]
            end_return = paths[d,:].sum()
            new3[:,d] += np.asarray([total,backtrack,end_return,total-backtrack-end_return])
        first,second,third = old1+new1,old2+new2,old3+new3
    counts2=second.sum((1,2))
    returns2=np.einsum('iij->i',second)
    mids=(second>0).sum(-1)
    different_endpoints=~np.eye(n,dtype=bool)
    pairs=mids>0
    nonreturn_pairs=pairs & different_endpoints
    return dict(q1_total=first.sum(-1),q2_total=counts2,q2_return=returns2,q2_three_distinct=counts2-returns2,
        q2_unique_identity_triples=(second>0).sum((1,2)),
        q2_endpoint_pairs=pairs.sum(-1),q2_pairs_multiple_middles=(mids>=2).sum(-1),
        q2_distinct_middle_sum=mids.sum(-1),
        q2_nonreturn_endpoint_pairs=nonreturn_pairs.sum(-1),
        q2_nonreturn_pairs_multiple_middles=((mids>=2)&different_endpoints).sum(-1),
        q2_nonreturn_distinct_middle_sum=(mids*different_endpoints).sum(-1),
        q3_total=third[0],q3_backtrack=third[1],q3_end_return=third[2],q3_four_distinct=third[3],
        q3_repeated=third[1]+third[2])
