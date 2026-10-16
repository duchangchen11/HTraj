"""Parameter participation probes, no optimizer update, real frozen train data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from src.socialvae_attribution.model import OfficialSocialVAE,SocialVAENoSocial,structural_parameter_masks
from src.socialvae_causal.data import CausalDataset,collate_examples
from src.socialvae_causal.training import seed_all
from src.socialvae_baseline.artifacts import json_new,state_info
from src.socialvae_causal.io import csv_new

seed_all(17)
plan=json.load(open('configs/stage5p3/protocol.json'))
ds=CausalDataset(plan['data_root'],plan['manifests']['crowds_zara02']['path'],'train')
# Four batches spanning the fixed pool, no validation-driven choice.
indices=[np.arange(s,min(s+16,len(ds))) for s in [0,len(ds)//3,len(ds)//2,len(ds)-16]]
rows=[];summary=[]
for label,cls in [('Full',OfficialSocialVAE),('NoSocial',SocialVAENoSocial)]:
    seed_all(17);model=cls(12,2,256).cuda().train();initial=state_info(model)
    called=set();handles=[];seen={name:torch.zeros_like(p,dtype=torch.bool) for name,p in model.named_parameters()};connected=set()
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            def hook(mod,inputs,output):
                for p in mod.parameters(recurse=False):called.add(id(p))
            handles.append(module.register_forward_hook(hook))
    losses=[]
    for idx in indices:
        model.zero_grad(set_to_none=True)
        batch=collate_examples([ds[int(i)] for i in idx],'cuda')
        loss=model.loss(*model(*batch));loss['loss'].backward()
        losses.append({k:float(v.detach()) for k,v in loss.items()})
        for name,p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all();connected.add(name);seen[name]|=(p.grad!=0)
    masks=structural_parameter_masks(model,label)
    for name,p in model.named_parameters():
        actual=seen[name].cpu();mask=masks[name]
        assert not actual[~mask].any(),(label,name)
        rows.append(dict(model=label,parameter=name,nominal=p.numel(),forward_module_called=id(p) in called,
            loss_graph_connected=name in connected,observed_nonzero_gradient_scalars=int(actual.sum()),
            observed_zero_gradient_scalars=int((~actual).sum()),structurally_effective_upper_bound=int(mask.sum()),
            structurally_dead_scalars=int((~mask).sum())))
    rr=[r for r in rows if r['model']==label]
    summary.append(dict(model=label,total_parameters=sum(r['nominal'] for r in rr),
        module_forward_used_parameters=sum(r['nominal'] for r in rr if r['forward_module_called']),
        loss_graph_connected_parameters=sum(r['nominal'] for r in rr if r['loss_graph_connected']),
        observed_nonzero_gradient_scalars=sum(r['observed_nonzero_gradient_scalars'] for r in rr),
        structural_effective_upper_bound=sum(r['structurally_effective_upper_bound'] for r in rr),
        probe_batches=4,probe_targets=64,optimizer_updates=0,losses=losses,
        initial_parameter_state_sha256=initial['parameter_state_sha256']))
    assert state_info(model)==initial
    for h in handles:h.remove()
    del model
csv_new('results/tables/stage5p3/capacity_by_parameter.csv',rows)
json_new('results/tables/stage5p3/capacity_audit.json',summary)
print(json.dumps(summary,indent=2))
# Inspect pre-existing Full optimizer states, not an exact lifetime-gradient log.
old=[]
for rec in ['crowds_zara02','biwi_hotel']:
    selection=json.load(open(f'results/tables/stage5p2/{rec}/selection.json'))
    for which in ['best','last']:
        cp=torch.load(selection[which]['path'],map_location='cpu',weights_only=True)
        names=list(OfficialSocialVAE(12,2,256).named_parameters())
        ids=cp['optimizer']['param_groups'][0]['params']
        for (name,p),i in zip(names,ids):
            state=cp['optimizer']['state'].get(i,{})
            old.append(dict(recording=rec,selection=which,parameter=name,nominal=p.numel(),
                optimizer_state_exists=bool(state),second_moment_nonzero_scalars=int((state['exp_avg_sq']!=0).sum()) if state else 0,
                interpretation='nonzero second moment proxy, not exact lifetime gradient union'))
csv_new('results/tables/stage5p3/full_optimizer_capacity_proxy.csv',old)
