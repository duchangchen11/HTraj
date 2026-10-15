"""Explicit latent-prior-mean, first draw, independent minADE/FDE and ELBO."""
import numpy as np
import torch
from .data import collate_examples


def trajectory_errors(prediction,truth):
    if prediction.ndim!=4 or prediction.shape[1:]!=truth.shape:
        raise ValueError('Expected [K,12,B,2] predictions and [12,B,2] truth')
    error = (prediction-truth[None]).norm(dim=-1)
    return error.mean(1),error[:,-1]


def cv_prediction(history,kind,dt=.4):
    x=np.asarray(history,dtype=np.float64)
    if x.shape!=(8,2) or not np.isfinite(x).all():raise ValueError('Complete8 positions required')
    if kind=='CV-Last2':velocity=(x[-1]-x[-2])/dt
    elif kind in ('CV-OLS4','CV-OLS8'):
        n=int(kind[-1]);t=np.arange(n,dtype=np.float64)*dt;t-=t.mean()
        velocity=(t[:,None]*x[-n:]).sum(0)/(t*t).sum()
    else:raise ValueError(kind)
    return x[-1]+np.arange(1,13)[:,None]*dt*velocity


def evaluate(model,dataset,batch_size=128,device='cuda',posterior=True):
    rows=[];probability=[]
    model.eval()
    with torch.no_grad():
        for b,start in enumerate(range(0,len(dataset),batch_size)):
            items=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
            x,y,n=collate_examples(items,device)
            mean=model(x,n)
            torch.manual_seed(171000+b)
            samples=model(x,n,20)
            a,f=trajectory_errors(samples,y)
            ma,mf=trajectory_errors(mean[None],y)
            if not torch.isfinite(samples).all() or not torch.isfinite(mean).all():raise FloatingPointError('Nonfinite validation prediction')
            for i,item in enumerate(items):
                rows.append(dict(recording=item['recording'],start_frame=item['start_frame'],target_id=item['actor'],
                    K1_prior_ADE_m=float(ma[0,i]),K1_prior_FDE_m=float(mf[0,i]),
                    K1_sample_ADE_m=float(a[0,i]),K1_sample_FDE_m=float(f[0,i]),
                    minADE20_m=float(a[:,i].min()),minFDE20_m=float(f[:,i].min()),
                    minADE_index=int(a[:,i].argmin()),minFDE_index=int(f[:,i].argmin()),sampling_seed=171000+b))
            if posterior:
                # Future targets enter the supervised posterior/loss only. Future
                # neighbor slots remain padding; never supplied neighbor GT.
                torch.manual_seed(172000+b)
                model.train()
                losses=model.loss(*model(x,y,n))
                if not all(torch.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite posterior diagnostic')
                probability.append((len(items),{k:float(v) for k,v in losses.items()}))
                model.eval()
    summary={k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k.endswith('_m')}
    summary.update(target_windows=len(rows),windows=len({(r['recording'],r['start_frame']) for r in rows}),
        people=len({(r['recording'],r['target_id']) for r in rows}))
    if posterior:
        summary.update({f'posterior_{k}':sum(n*v[k] for n,v in probability)/len(rows) for k in ['loss','rec','kl']})
    return summary,rows
