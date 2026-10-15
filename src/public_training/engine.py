"""Shared deterministic evaluator; no model selection or graph construction from GT."""
import time
import torch
from src.public_training.data import displacement,network_input
from src.public_training.model import forward,nll,gaussian_trajectories

MODES=('K1_distribution_mean','K1_first_sample','K20_independent_min')


@torch.no_grad()
def evaluate(model,samples,seed=314159,return_targets=False):
    model.eval();generator=torch.Generator(device='cuda').manual_seed(seed)
    sums={mode:torch.zeros(2,device='cuda',dtype=torch.float64) for mode in MODES};count=0;loss_sum=0.;targets=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for sample in samples:
        v,a=network_input(sample,'cuda');parameters=forward(model,v,a)
        truth=sample['future'].cuda();mask=sample['target_mask'].cuda();n=int(mask.sum());loss=nll(parameters,truth_velocity(sample).cuda(),mask)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite evaluation NLL')
        loss_sum+=float(loss)*n;count+=n
        noise=torch.randn(20,12,v.shape[-1],2,generator=generator,device='cuda')
        mean,draws=gaussian_trajectories(parameters,sample['observed'][-1].cuda(),noise)
        distance=(draws-truth[None]).norm(dim=-1);ade=distance.mean(1);fde=distance[:,-1]
        mean_error=(mean-truth).norm(dim=-1)
        scores=dict(K1_distribution_mean=(mean_error.mean(0),mean_error[-1]),K1_first_sample=(ade[0],fde[0]),
            K20_independent_min=(ade.min(0).values,fde.min(0).values))
        for mode,(aa,ff) in scores.items():
            aa,ff=aa[mask],ff[mask]
            if not torch.isfinite(aa).all() or not torch.isfinite(ff).all():raise FloatingPointError('Nonfinite metric')
            sums[mode]+=torch.stack([aa.double().sum(),ff.double().sum()])
            if return_targets:
                av,fv=aa.cpu().tolist(),ff.cpu().tolist()
                for pid,ae,fe in zip(sample['target_ids'],av,fv):targets.append(dict(recording=sample['recording'],window_id=sample['window_id'],agent_id=pid,
                    group_people=len(sample['agent_ids']),mode=mode,ADE_m=ae,FDE_m=fe))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    metrics=dict(NLL=loss_sum/count,target_windows=count,windows=len(samples),evaluation_seconds=elapsed)
    for mode,total in sums.items():
        pair=(total/count).cpu().tolist();metrics[mode+'_ADE']=pair[0];metrics[mode+'_FDE']=pair[1]
    return metrics,targets


def truth_velocity(sample):
    return torch.cat([sample['future'][:1]-sample['observed'][-1:],sample['future'][1:]-sample['future'][:-1]],0)
