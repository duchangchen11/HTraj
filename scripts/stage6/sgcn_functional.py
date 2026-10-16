"""Frozen official SGCN synthetic forward audit; no scene metrics or updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import importlib.util,json,time
import torch
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new

def main():
 root=Path('results/tables/stage6/public');metadata=json.load(open(root/'SGCN_git_fallback.json'))
 folder=Path(metadata['source_directory']);spec=importlib.util.spec_from_file_location('stage6_frozen_sgcn',folder/'model.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 torch.set_num_threads(1);torch.manual_seed(61016);torch.cuda.manual_seed_all(61016)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 cases=[];loads=[];start=time.perf_counter()
 for fold in ['hotel','zara2']:
  path=folder/f'checkpoints/{fold}/val_best.pth';payload=torch.load(path,map_location='cpu',weights_only=True)
  model=module.TrajectoryModel().cuda();model.load_state_dict(payload,strict=True);model.eval()
  loads.append(dict(fold=fold,path=str(path),sha256=sha256(path),strict_load=True,parameters=sum(p.numel() for p in model.parameters()),state_keys=len(payload)))
  for n in [1,2,3,5,8,12]:
   gen=torch.Generator().manual_seed(61016+n);g=torch.randn((1,8,n,3),generator=gen).cuda()*.1
   g[0,:,:,0]=torch.arange(8,device='cuda')[:,None]
   identities=[torch.eye(n,device='cuda').repeat(8,1,1),torch.eye(8,device='cuda').repeat(n,1,1)]
   p=torch.roll(torch.arange(n,device='cuda'),1);inv=torch.argsort(p)
   entry=dict(fold=fold,people=n,artificial_input=True,optimizer_updates=0)
   try:
    observed=[]
    handle=model.sparse_weighted_adjacency_matrices.register_forward_hook(lambda m,i,o:observed.append(o[0].detach().clone()))
    with torch.no_grad():a=model(g,identities);b=model(g[:,:,p],identities)[:,inv]
    handle.remove()
    entry.update(forward_success=True,output_shape=list(a.shape),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
     raw_gaussian_parameter_max_difference=float((a-b).abs().max()),
     mean_trajectory_max_difference_m=float(torch.linalg.vector_norm((a[:,:,:2]-b[:,:,:2]).cumsum(0),dim=-1).max()),
     learned_spatial_adjacency_max_difference=float((observed[0]-observed[1][:,:,inv][:,:,:,inv]).abs().max()))
    if n==3:
     def own_only(m,i,o):
      return identities[0][:,None].repeat(1,4,1,1),identities[1][:,None].repeat(1,4,1,1),o[2],o[3]
     handle=model.sparse_weighted_adjacency_matrices.register_forward_hook(own_only)
     x=g.detach().clone().requires_grad_(True);out=model(x,identities)
     grad=torch.autograd.grad(out[:,0,:2].sum(),x)[0]
     handle.remove()
     entry['other_node_Jacobian_L2_with_identity_graph']=float(grad[:,:,1:,:].norm())
     entry['identity_intervention']='structure-only, untrained intervention; no performance inference'
   except Exception as e:
    entry.update(forward_success=False,error_type=type(e).__name__,error=str(e))
    # A failed forward may leave a hook; remove it before the next independent case.
    if 'handle' in locals():handle.remove()
   cases.append(entry)
  del model;torch.cuda.empty_cache()
 csv_new(root/'SGCN_synthetic_cases.csv',cases,fieldnames=sorted({k for r in cases for k in r}))
 json_new(root/'SGCN_functional_execution.json',dict(repository_commit=metadata['commit'],source_sha256=sha256(folder/'model.py'),loads=loads,cases=cases,
  torch=torch.__version__,GPU=torch.cuda.get_device_name(),elapsed_seconds=time.perf_counter()-start,new_training_runs=0,optimizer_updates=0,
  real_scene_evaluation=False,ETH_evaluated=False,official_source_modified=False,
  caveat='Artificial graphs only. Upstream nonETH weights may have trained on ETH. Strict load and finite forward do not reproduce paper scores.'))
 print('SGCN strict loading and synthetic structural audit saved')

if __name__=='__main__':main()
