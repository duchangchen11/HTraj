"""Analytical baselines and fixed diagnostic utilities; no trainable modules."""
import hashlib
import numpy as np
import torch


def cv_prediction(observed,kind,dt=.4,pred_len=12):
    if observed.shape[0]!=8 or observed.shape[-1]!=2 or not torch.isfinite(observed).all():
        raise ValueError('This audit requires complete finite8-frame2D observations')
    if dt<=0:raise ValueError('Positive sampling interval required')
    x=observed.double()
    if kind=='CV-Last2':velocity=(x[-1]-x[-2])/dt
    else:
        if kind not in ('CV-OLS4','CV-OLS8'):raise ValueError(kind)
        k=4 if kind=='CV-OLS4' else 8
        times=torch.arange(k,dtype=x.dtype,device=x.device)*dt;times-=times.mean()
        centered=x[-k:]-x[-1:]
        velocity=(centered*times[:,None,None]).sum(0)/times.square().sum()
    future_times=torch.arange(1,pred_len+1,dtype=x.dtype,device=x.device)*dt
    return x[-1:]+future_times[:,None,None]*velocity


def errors(prediction,truth):
    distance=(prediction.double()-truth.double()).norm(dim=-1)
    return distance.mean(0),distance[-1]


def mean_positions(parameters,last):
    steps=parameters[...,:2].unbind(1);accumulated=[steps[0]]
    for step in steps[1:]:accumulated.append(accumulated[-1]+step)
    return last[:,None]+torch.stack(accumulated,1)


def observed_batch(samples,device='cuda'):
    """Model inputs only; deliberately never accesses a future or target label."""
    b=len(samples);n=max(s['observed'].shape[1] for s in samples)
    v=torch.zeros(b,2,8,n);r=torch.zeros(b,8,n,n);last=torch.zeros(b,n,2);valid=torch.zeros(b,n,dtype=torch.bool)
    for k,s in enumerate(samples):
        count=s['observed'].shape[1];observed=s['observed']
        v[k,:,:,:count]=torch.cat([torch.zeros_like(observed[:1]),observed[1:]-observed[:-1]],0).permute(2,0,1)
        r[k,:,:count,:count]=s['relation'];last[k,:count]=observed[-1];valid[k,:count]=True
    return dict(v=v.to(device),relation=r.to(device),last=last.to(device),valid=valid.to(device))


def motion_group(observed):
    speed=((observed[1:]-observed[:-1])/.4).norm(dim=-1)
    mean=speed.mean(0)
    names=['stationary' if v<=.2 else 'low_speed' if v<=.8 else 'normal_walking' if v<=1.6 else 'fast' for v in mean.tolist()]
    return names,dict(exact_stationary=(observed[1:]-observed[:-1]).abs().amax((0,2))<=1e-8,
        strict_stationary=(speed.max(0).values<=.05)&((observed[-1]-observed[0]).norm(dim=-1)<=.05))


def jitter_history(observed,window_id,ids,epsilon,replicate):
    """Stable-ID keyed directions, exactlyepsilon norm per historical2D point."""
    noise=torch.zeros_like(observed)
    for i,pid in enumerate(ids):
        seed=int.from_bytes(hashlib.sha256(f'final-ptn:271828:{window_id}:{pid}:{replicate}'.encode()).digest()[:8],'little')%(2**63-1)
        direction=torch.randn(8,2,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
        direction/=direction.norm(dim=-1,keepdim=True)
        noise[:,i]=(direction*epsilon).to(observed)
    return observed+noise


def diagnose_curve(rows):
    def avg(a,b,key):return float(np.mean([float(r[key]) for r in rows[a-1:b]]))
    train_before=avg(20,24,'train_NLL_window_mean');train_after=avg(26,30,'train_NLL_window_mean')
    ade_before=avg(20,24,'K1_distribution_mean_ADE');ade_after=avg(26,30,'K1_distribution_mean_ADE')
    best=min(rows,key=lambda r:(float(r['K1_distribution_mean_ADE']),int(r['epoch'])))
    finite=all(np.isfinite(float(r[key])) for r in rows for key in ('train_NLL_window_mean','NLL','K1_distribution_mean_ADE','K1_distribution_mean_FDE','gradient_norm_max'))
    needs=finite and train_before-train_after>=.01 and ade_before-ade_after>=max(.005,.01*ade_before) and int(best['epoch'])>=28
    dn=np.diff([float(r['NLL']) for r in rows]);da=np.diff([float(r['K1_distribution_mean_ADE']) for r in rows])
    return dict(train_NLL_20to24=train_before,train_NLL_26to30=train_after,train_NLL_drop=train_before-train_after,
        val_ADE_20to24=ade_before,val_ADE_26to30=ade_after,val_ADE_drop=ade_before-ade_after,
        val_NLL_20to24=avg(20,24,'NLL'),val_NLL_26to30=avg(26,30,'NLL'),best_epoch=int(best['epoch']),all_finite=finite,
        NLL_decrease_ADE_increase_steps=int(((dn<0)&(da>0)).sum()),NLL_ADE_epoch_correlation=float(np.corrcoef([float(r['NLL']) for r in rows],[float(r['K1_distribution_mean_ADE']) for r in rows])[0,1]),
        clearly_unfinished=bool(needs),median_train_eval_seconds=float(np.median([float(r['train_seconds'])+float(r['evaluation_seconds']) for r in rows[-5:]])))


def earliest_best(rows):
    return min(rows,key=lambda r:(float(r['K1_distribution_mean_ADE']),int(r['epoch'])))
