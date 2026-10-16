"""NoSocial checkpoint restore and actual lifetime gradient bookkeeping."""
import torch
from src.socialvae_causal.training import tree_equal
from .model import new_model,structural_parameter_masks


class GradientTracker:
    def __init__(self,model):
        self.ever={n:torch.zeros_like(p,dtype=torch.bool) for n,p in model.named_parameters()}
        self.maxima={n:torch.zeros((),device=p.device) for n,p in model.named_parameters()}
        self.connected=set();self.masks=structural_parameter_masks(model,'NoSocial')
        self.initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        self.steps=0

    def update(self,model):
        self.steps+=1
        for name,p in model.named_parameters():
            if p.grad is not None:
                g=p.grad.detach();self.connected.add(name)
                self.ever[name]|=(g!=0)
                self.maxima[name]=torch.maximum(self.maxima[name],g.abs().max())

    def snapshot(self,model,epoch):
        rows=[]
        for name,p in model.named_parameters():
            mask=self.masks[name];g=self.ever[name].cpu()
            assert not g[~mask].any(),name
            unchanged=bool(torch.equal(p.detach().cpu()[~mask],self.initial[name][~mask]))
            assert unchanged,('Structurally dead parameter changed',name)
            rows.append(dict(epoch=epoch,parameter=name,nominal=p.numel(),
                gradient_connected_ever=name in self.connected,ever_nonzero_gradient_scalars=int(g.sum()),
                never_nonzero_gradient_scalars=int((~g).sum()),max_abs_gradient=float(self.maxima[name]),
                structural_effective_upper_bound=int(mask.sum()),structurally_dead_scalars=int((~mask).sum()),
                dead_parameter_values_unchanged=unchanged,optimizer_steps_observed=self.steps))
        return rows


def verify_checkpoint(path,model,optimizer,batch):
    with torch.random.fork_rng(devices=[0]):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=new_model();restored.load_state_dict(saved['model'],strict=True)
        opt=torch.optim.Adam(restored.parameters(),lr=.0005);opt.load_state_dict(saved['optimizer'])
        assert tree_equal(model.state_dict(),restored.state_dict())
        assert tree_equal(optimizer.state_dict(),opt.state_dict())
        model.eval();restored.eval()
        with torch.no_grad():
            a=model(batch[0],batch[2]);b=restored(batch[0],batch[2])
        difference=float((a-b).abs().max());assert torch.equal(a,b)
        model.train();restored.train()
        with torch.no_grad():
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            original=model.loss(*model(*batch))
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            resumed=restored.loss(*restored(*batch))
        loss_difference=max(float((original[k]-resumed[k]).abs()) for k in original)
        assert loss_difference==0.
    return dict(strict_model_restore=True,optimizer_state_identical=True,
        deterministic_prediction_max_difference_m=difference,rng_resume_verified=True,
        supervised_stochastic_loss_max_difference=loss_difference)
