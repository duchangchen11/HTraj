"""Engineering-only training utilities for the unchanged official VAE/loss."""
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import collate_examples
from .official import load_official


def seed_all(seed=17):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def new_model():return load_official().SocialVAE(12,2,256).cuda()


def train_step(model,optimizer,batch):
    model.train();optimizer.zero_grad(set_to_none=True)
    x,y,n=batch
    loss=model.loss(*model(x,y,n))
    if not torch.stack([torch.isfinite(v) for v in loss.values()]).all():raise FloatingPointError('Nonfinite loss')
    loss['loss'].backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():raise FloatingPointError('Nonfinite/missing gradient')
    norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    optimizer.step()
    return dict(loss=float(loss['loss'].detach()),rec=float(loss['rec'].detach()),kl=float(loss['kl'].detach()),gradient_L2=float(norm))


def timed_batch(dataset,indices):
    start=time.perf_counter()
    items=[dataset[int(i)] for i in indices]
    batch=collate_examples(items,'cuda')
    torch.cuda.synchronize()
    return batch,time.perf_counter()-start


def save_checkpoint_new(path,model,optimizer,**metadata):
    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),**metadata)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(payload,f)


def tree_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def verify_checkpoint(path,model,optimizer,batch):
    """Strict model+optimizer restore and same deterministic prediction.

    Clone initialization consumes no subsequent training RNG: fork_rng restores it.
    """
    with torch.random.fork_rng(devices=[0]):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=new_model();restored.load_state_dict(saved['model'],strict=True)
        opt=torch.optim.Adam(restored.parameters(),lr=.0005);opt.load_state_dict(saved['optimizer'])
        assert tree_equal(model.state_dict(),restored.state_dict())
        assert tree_equal(optimizer.state_dict(),opt.state_dict())
        model.eval();restored.eval()
        with torch.no_grad():
            a=model(batch[0],batch[2]);b=restored(batch[0],batch[2])
        difference=float((a-b).abs().max())
        if not torch.equal(a,b):raise AssertionError('Strict restore changed deterministic prediction')
        # Confirm both CPU/CUDA random states reproduce a supervised latent
        # draw as well, without performing an extra optimizer update.
        model.train();restored.train()
        with torch.no_grad():
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            original_loss=model.loss(*model(*batch))
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            restored_loss=restored.loss(*restored(*batch))
        stochastic_difference=max(float((original_loss[k]-restored_loss[k]).abs()) for k in original_loss)
        if stochastic_difference!=0.:raise AssertionError('RNG restore changed supervised stochastic loss')
        del restored,opt,saved
    return dict(strict_model_restore=True,optimizer_state_identical=True,deterministic_prediction_max_difference_m=difference,
        rng_resume_verified=True,supervised_stochastic_loss_max_difference=stochastic_difference)
