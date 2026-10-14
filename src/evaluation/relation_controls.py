"""Explicitly constrained history-only inference relation interventions."""
from contextlib import contextmanager
import numpy as np
import torch
from ..models.temporal_graph import GraphFrame,relation_geometry,target_softmax

CONTROL_MODES=['full','radius_max_change','scene_random','scene_max_change','degree_swap_global',
               'cut_indirect_paths','matched_random_delete','remove_all_messages']


def rescore(model,encoded,t,pairs):
    b,_,n,d=encoded['hidden'].shape
    device=encoded['hidden'].device
    pairs=sorted(pairs,key=lambda p:(p[1],p[0]))
    source=torch.tensor([p[0] for p in pairs],dtype=torch.long,device=device)
    target=torch.tensor([p[1] for p in pairs],dtype=torch.long,device=device)
    r,w,distance,tau,closest=relation_geometry(encoded['positions'][:,t],encoded['velocity'][:,t],model.graph.tau_max)
    scene=target//n; i=target%n; j=source%n
    features=torch.cat([r[scene,i,j],w[scene,i,j],tau[scene,i,j,None],closest[scene,i,j,None],
                        torch.ones(len(source),1,device=device)],-1)
    previous=encoded['hidden'][:,t-1] if t else torch.zeros_like(encoded['hidden'][:,0])
    score=model.graph.relation_score(torch.cat([features,encoded['hidden'][scene,t,i],previous[scene,j]],-1)).squeeze(-1)
    weight=target_softmax(score,target,b*n)
    totals=features.new_zeros(b*n,7).index_add(0,target,features)
    degree=weight.new_zeros(b*n).index_add(0,target,torch.ones_like(weight))
    return GraphFrame(source,target,weight,features,(totals/degree[:,None].clamp_min(1)).reshape(b,n,7))


def _cut_pairs(encoded,frames):
    b,_,n,_=encoded['hidden'].shape
    seen=torch.zeros(b*n,dtype=torch.bool,device=encoded['hidden'].device)
    out=[]
    for t,frame in enumerate(frames):
        continuity=encoded['continuity'][:,t].reshape(-1)
        seen=seen & continuity
        keep=~seen[frame.source]
        pairs=list(zip(frame.source[keep].tolist(),frame.target[keep].tolist()))
        out.append(pairs)
        received=torch.zeros_like(seen)
        received[frame.target[keep]]=True
        seen=(seen | received) & continuity
    return out


def altered_frames(model,encoded,frames,mode,seed=1726):
    if mode not in CONTROL_MODES:
        raise ValueError('Unknown graph control')
    if mode=='full':
        return frames
    b,_,n,_=encoded['hidden'].shape
    cuts=_cut_pairs(encoded,frames) if mode in ('cut_indirect_paths','matched_random_delete') else None
    out=[]
    for t,frame in enumerate(frames):
        real=list(zip(frame.source.tolist(),frame.target.tolist()))
        if mode=='remove_all_messages':
            pairs=[]
        elif mode=='cut_indirect_paths':
            pairs=cuts[t]
        elif mode=='degree_swap_global':
            pairs=[]
            for scene in range(b):
                edges=[p for p in real if p[0]//n==scene]
                edge_set=set(edges); original=edge_set.copy()
                rng=np.random.default_rng(seed+t*10007+scene*7919)
                for _ in range(50*len(edges)):
                    if len(edges)<2:
                        break
                    a,c=rng.choice(len(edges),2,replace=False)
                    s1,t1=edges[a]; s2,t2=edges[c]
                    e1,e2=(s2,t1),(s1,t2)
                    if s1==s2 or t1==t2 or s2==t1 or s1==t2 or e1 in edge_set or e2 in edge_set:
                        continue
                    improvement=int((s1,t1) in original)+int((s2,t2) in original)-int(e1 in original)-int(e2 in original)
                    if improvement<0:
                        continue
                    edge_set.remove((s1,t1)); edge_set.remove((s2,t2)); edge_set.update([e1,e2])
                    edges[a]=e1; edges[c]=e2
                pairs.extend(edges)
        else:
            pairs=[]
            distance=(encoded['positions'][:,t,:,None]-encoded['positions'][:,t,None,:]).norm(dim=-1).cpu().numpy()
            for scene in range(b):
                eligible=encoded['continuity'][scene,t].nonzero().flatten().cpu().numpy()
                for target in range(n):
                    original=[s%n for s,i in real if i==scene*n+target]
                    degree=len(original)
                    if mode=='matched_random_delete':
                        degree=sum(i==scene*n+target for s,i in cuts[t])
                        choices=np.asarray(original,dtype=np.int64)
                    else:
                        choices=eligible[eligible!=target]
                        if mode=='radius_max_change':
                            choices=choices[distance[scene,target,choices]<=model.graph.radius]
                    if not degree:
                        continue
                    rng=np.random.default_rng(seed+t*10007+scene*7919+target)
                    if mode in ('radius_max_change','scene_max_change'):
                        outside=choices[~np.isin(choices,original)]
                        inside=choices[np.isin(choices,original)]
                        selected=list(rng.permutation(outside)[:degree])
                        selected.extend(rng.permutation(inside)[:degree-len(selected)].tolist())
                    else:
                        selected=rng.choice(choices,degree,replace=False).tolist()
                    assert len(selected)==degree
                    pairs.extend((scene*n+int(s),scene*n+target) for s in selected)
        out.append(rescore(model,encoded,t,pairs))
    return out


def control_counts(encoded,real,altered,radius=4.):
    n=encoded['hidden'].shape[2]
    output=dict(real_edges=0,control_edges=0,removed_edges=0,added_edges=0,
                control_outside_radius_edges=0,possible_local_changed_edges=0,
                incoming_degree_mismatch_nodes=0,outgoing_degree_mismatch_nodes=0)
    for t,(original,changed) in enumerate(zip(real,altered)):
        a=set(zip(original.source.tolist(),original.target.tolist()))
        z=set(zip(changed.source.tolist(),changed.target.tolist()))
        output['real_edges']+=len(a); output['control_edges']+=len(z)
        output['removed_edges']+=len(a-z); output['added_edges']+=len(z-a)
        b=encoded['hidden'].shape[0]
        distance=(encoded['positions'][:,t,:,None]-encoded['positions'][:,t,None,:]).norm(dim=-1)
        for s,i in z:
            assert s//n==i//n and s!=i and bool(encoded['continuity'][s//n,t,s%n]) and bool(encoded['continuity'][i//n,t,i%n])
            output['control_outside_radius_edges']+=int(distance[i//n,i%n,s%n]>radius)
        for node in range(b*n):
            scene,i=node//n,node%n
            orig_in={s for s,k in a if k==node}; new_in={s for s,k in z if k==node}
            orig_out={k for s,k in a if s==node}; new_out={k for s,k in z if s==node}
            output['incoming_degree_mismatch_nodes']+=int(len(orig_in)!=len(new_in))
            output['outgoing_degree_mismatch_nodes']+=int(len(orig_out)!=len(new_out))
            eligible=encoded['continuity'][scene,t] & (distance[scene,i]<=radius)
            eligible=eligible.clone(); eligible[i]=False
            original_degree=len(orig_in)
            output['possible_local_changed_edges']+=min(original_degree,int(eligible.sum())-original_degree)
    return output


@contextmanager
def graph_control(model,mode,seed=1726):
    """Temporarily substitute graph.sequence; original model code/weights stay intact."""
    original=model.graph.sequence
    trace={}
    def sequence(encoded):
        frames=original(encoded)
        altered=altered_frames(model,encoded,frames,mode,seed)
        trace.update(encoded=encoded,real=frames,altered=altered)
        return altered
    model.graph.sequence=sequence
    try:
        yield trace
    finally:
        # Restore class method resolution, without leaving a bound-method copy.
        del model.graph.sequence
