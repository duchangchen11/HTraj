"""Thin integration of immutable Stage11B-0 training mathematics."""
from scripts.stage11b0.training_adapter import training_loss,gradients
from scripts.stage11b0.runtime import state_digest,save_checkpoint,load_checkpoint
import torch

def train_batch(net,opt,batch,semantic,cfg,templates):
    opt.zero_grad(set_to_none=True)
    out=training_loss(net,batch,semantic,cfg,templates)
    if not all(torch.isfinite(v) for v in out.values()):raise FloatingPointError('Nonfinite training loss')
    out['loss'].backward();norm,nz,ng=gradients(net);opt.step()
    if not torch.stack([torch.isfinite(p).all() for p in net.parameters() if p.requires_grad]).all():
        raise FloatingPointError('Nonfinite predictor parameter')
    return dict(loss=float(out['loss']),goal_loss=float(out['goal_loss']),path_loss=float(out['path_loss']),gradient_norm=norm,
                nonzero_gradient_elements=nz,gradient_elements=ng)

def checkpoint(path,net,opt,epoch,step,order,settings):
    progress=dict(epoch=epoch,global_step=step,epoch_boundary=True,
        completed_scene_order=order,next_epoch=epoch+1,next_scene_seed=17+epoch)
    if settings.get('preflight'):progress.update(epoch_boundary=False,engineering_only=True)
    return save_checkpoint(path,net,opt,progress,settings)

def choose_best(results):
    valid=[r for r in results if r['status']=='COMPLETE']
    return min(valid,key=lambda r:(r['window']['minADE20'],r['epoch'])) if valid else None
