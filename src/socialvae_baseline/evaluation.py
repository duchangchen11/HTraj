"""Evaluation-only fixed standard-Normal draws; frozen official forward/loss.

Target-local generators make epsilon invariant to batch partition/order. Floating
point batched kernels can still cause rounding-level prediction differences.
"""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from src.socialvae_causal.data import PROTOCOL, collate_examples
from src.socialvae_causal.evaluation import trajectory_errors

METRICS = ('K1_prior_ADE_m','K1_prior_FDE_m','K1_sample_ADE_m',
           'K1_sample_FDE_m','minADE20_m','minFDE20_m')


def identity(item):
    return f"{PROTOCOL}|seed17|{item['recording']}|{item['start_frame']}|{item['actor']}"


def stable_seed(item, namespace):
    return int.from_bytes(hashlib.sha256((namespace+'|'+identity(item)).encode()).digest()[:8], 'big') % (2**63)


def noise(item, posterior=False):
    namespace = 'stage5p2-posterior-noise-v1' if posterior else 'stage5p2-prior-noise-v1'
    seed = stable_seed(item, namespace)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    shape = (12,32) if posterior else (20,12,32)
    return torch.randn(shape,generator=generator),seed


class FixedNormal(torch.distributions.Normal):
    def __init__(self, original, epsilon):
        super().__init__(original.loc,original.scale)
        if epsilon.shape != self.loc.shape:raise ValueError('Noise layout differs from original Normal')
        self.epsilon=epsilon.to(device=self.loc.device,dtype=self.loc.dtype)

    def rsample(self,sample_shape=torch.Size()):
        if sample_shape:raise ValueError('Official calls use empty sample_shape')
        return self.loc+self.scale*self.epsilon

    def sample(self,sample_shape=torch.Size()):
        with torch.no_grad():return self.rsample(sample_shape)


@contextmanager
def fixed_draws(module, epsilon):
    count=0
    def hook(_module,_inputs,distribution):
        nonlocal count
        if count >= len(epsilon):raise AssertionError('Unexpected extra latent draw')
        result=FixedNormal(distribution,epsilon[count]);count+=1
        return result
    handle=module.register_forward_hook(hook)
    try:
        yield
        if count != len(epsilon):raise AssertionError('Missing latent draw')
    finally:handle.remove()


def batch_noise(items,posterior,device):
    values=[noise(item,posterior)[0] for item in items]
    if posterior:return torch.stack(values).permute(1,0,2).to(device)
    # Official h.repeat(1,K,1): flattened order is K groups of B targets.
    return torch.stack(values).permute(2,1,0,3).reshape(12,20*len(items),32).to(device)


def evaluate(model,dataset,batch_size=128,device='cuda',posterior=True):
    rows=[];probability=[]
    old_mode=model.training
    devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            model.eval()
            for start in range(0,len(dataset),batch_size):
                items=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
                x,y,n=collate_examples(items,device)
                mean=model(x,n)
                with fixed_draws(model.p_z,batch_noise(items,False,device)):
                    samples=model(x,n,20)
                if not torch.isfinite(samples).all() or not torch.isfinite(mean).all():
                    raise FloatingPointError('Nonfinite validation prediction')
                a,f=trajectory_errors(samples,y);ma,mf=trajectory_errors(mean[None],y)
                for i,item in enumerate(items):
                    rows.append(dict(recording=item['recording'],start_frame=item['start_frame'],target_id=item['actor'],
                        K1_prior_ADE_m=float(ma[0,i]),K1_prior_FDE_m=float(mf[0,i]),
                        K1_sample_ADE_m=float(a[0,i]),K1_sample_FDE_m=float(f[0,i]),
                        minADE20_m=float(a[:,i].min()),minFDE20_m=float(f[:,i].min()),
                        minADE_index=int(a[:,i].argmin()),minFDE_index=int(f[:,i].argmin()),
                        prior_seed=stable_seed(item,'stage5p2-prior-noise-v1'),
                        posterior_seed=stable_seed(item,'stage5p2-posterior-noise-v1')))
                if posterior:
                    model.train()
                    with fixed_draws(model.q_z,batch_noise(items,True,device)):
                        losses=model.loss(*model(x,y,n))
                    if not all(torch.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite posterior diagnostic')
                    probability.append((len(items),{k:float(v) for k,v in losses.items()}))
                    model.eval()
    finally:model.train(old_mode)
    summary={k:float(np.mean([r[k] for r in rows])) for k in METRICS}
    summary.update(target_windows=len(rows),windows=len({(r['recording'],r['start_frame']) for r in rows}),
        people=len({(r['recording'],r['target_id']) for r in rows}))
    if posterior:summary.update({f'posterior_{k}':sum(n*v[k] for n,v in probability)/len(rows) for k in ('loss','rec','kl')})
    return summary,rows


def evaluate_train_K1(model,dataset,batch_size=128,device='cuda'):
    old_mode=model.training;total_a=total_f=0.
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(dataset),batch_size):
                items=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
                x,y,n=collate_examples(items,device)
                prediction=model(x,n)
                if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite train deterministic prediction')
                a,f=trajectory_errors(prediction[None],y)
                total_a+=float(a.sum());total_f+=float(f.sum())
    finally:model.train(old_mode)
    return dict(train_K1_prior_ADE_m=total_a/len(dataset),train_K1_prior_FDE_m=total_f/len(dataset))
