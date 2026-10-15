"""History-only relations, degree-matched observable control and padded batches."""
import hashlib
import torch
from src.public_training.data import displacement


def real_relation(observed,radius=4.,sigma=2.):
    n=observed.shape[1];distance=(observed[:,:,None]-observed[:,None,:]).square().sum(-1)
    legal=(distance<=radius**2)&~torch.eye(n,dtype=torch.bool,device=observed.device)[None]
    weight=torch.exp(-distance/(2*sigma**2))*legal
    return weight/weight.sum(-1,keepdim=True).clamp_min(1e-20)


def controlled_relation(real,ids,window_id,seed=271828):
    """Same receiver degree and weight multiset; same-window sources only.

    Deliberately relaxes real4m radius, a reported distance-distribution confound.
    ID-keyed candidate ordering gives reproducibility independent of storage order.
    """
    result=torch.zeros_like(real);n=len(ids)
    for i,pid in enumerate(ids):
        candidates=[j for j in range(n) if j!=i]
        candidates.sort(key=lambda j:hashlib.sha256(f'{seed}:{window_id}:{pid}:{ids[j]}'.encode()).digest())
        for t in range(8):
            weights=real[t,i][real[t,i]>0].sort().values
            if len(weights):result[t,i,candidates[:len(weights)]]=weights
    return result


def pack(samples,kind,device='cuda',with_noise=False):
    b=len(samples);n=max(len(s['agent_ids']) for s in samples)
    v=torch.zeros(b,2,8,n);r=torch.zeros(b,8,n,n);y=torch.zeros(b,12,n,2);velocity=torch.zeros_like(y)
    last=torch.zeros(b,n,2);valid=torch.zeros(b,n,dtype=torch.bool)
    noise=torch.zeros(b,20,12,n,2) if with_noise else None
    for k,s in enumerate(samples):
        count=len(s['agent_ids']);v[k,:,:,:count]=displacement(s['observed']).permute(2,0,1)
        r[k,:,:count,:count]=s['control'] if kind=='G2' else s['relation']
        y[k,:,:count]=s['future'];last[k,:count]=s['observed'][-1];valid[k,:count]=True
        velocity[k,0,:count]=s['future'][0]-s['observed'][-1];velocity[k,1:,:count]=s['future'][1:]-s['future'][:-1]
        if with_noise:noise[k,:,:,:count]=s['noise']
    result=dict(v=v.to(device),relation=r.to(device),future=y.to(device),velocity=velocity.to(device),last=last.to(device),valid=valid.to(device))
    if with_noise:result['noise']=noise.to(device)
    return result


def history_groups(observed,relation,visible_degree,complete8_degree):
    velocity=(observed[1:]-observed[:-1])/.4;speed=velocity.norm(dim=-1);mean=speed.mean(0)
    acceleration=((velocity[1:]-velocity[:-1])/.4).norm(dim=-1).mean(0)
    distance=(observed[1:]-observed[:-1]).norm(dim=-1).sum(0)
    ratio=(observed[-1]-observed[0]).norm(dim=-1)/(distance+1e-8)
    degree=(relation>0).sum(-1);n=observed.shape[1];result=[]
    for i in range(n):
        groups=['all','context_N2to4' if n<=4 else 'context_N5to8' if n<=8 else 'context_N9plus',
            'legal_neighbors_none' if int(degree[:,i].sum())==0 else 'legal_neighbors_present',
            'legal_degree_low' if float(degree[:,i].float().mean())<=1 else 'legal_degree_mid' if float(degree[:,i].float().mean())<=3 else 'legal_degree_high',
            'speed_stationary' if mean[i]<=.2 else 'speed_slow' if mean[i]<=.8 else 'speed_walking' if mean[i]<=1.6 else 'speed_fast',
            'motion_smooth' if acceleration[i]<.5 else 'motion_variable',
            'path_turning_or_return' if mean[i]>.2 and ratio[i]<.8 else 'path_straight_or_low_speed',
            'observed_neighbors_none' if float(visible_degree[:,i].sum())==0 else 'observed_neighbors_present']
        if int(degree[:,i].sum())==0 and float(visible_degree[:,i].sum())>0:groups.append('observed_neighbors_but_no_model_edge')
        last=relation[-1,i]>0
        moving=(speed[-1]>.2)&(speed[-1,i]>.2)
        cosine=(velocity[-1]*velocity[-1,i]).sum(-1)/(speed[-1]*speed[-1,i]+1e-8)
        for name,mask in [('same_direction',cosine>=.7),('opposing_direction',cosine<=-.7),('crossing_direction',(cosine>-.7)&(cosine<.7))]:
            if bool((last&moving&mask).any()):groups.append(name)
        changed=((relation[1:,i]>0)!=(relation[:-1,i]>0)).any()
        groups.append('relations_changing' if bool(changed) else 'relations_stable')
        result.append(dict(groups=groups,mean_speed_mps=float(mean[i]),mean_acceleration_mps2=float(acceleration[i]),path_ratio=float(ratio[i]),
            offline_context_people=n,model_mean_degree=float(degree[:,i].float().mean()),
            historical_visible_mean_degree=float(visible_degree[:,i].float().mean()),historical_complete8_mean_degree=float(complete8_degree[:,i].float().mean())))
    return result
