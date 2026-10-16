"""Module-axis checks and a limited normalization probe, never model training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,importlib.util
import torch
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

root=Path('results/tables/stage6/public')
sg=json.load(open(root/'SGCN_git_fallback.json'));folder=Path(sg['source_directory'])
m=load('stage6_sgcn_axes',folder/'model.py');model=m.TrajectoryModel().cuda()
state=torch.load(folder/'checkpoints/zara2/val_best.pth',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model.eval()
torch.manual_seed(616);n=5;g=torch.randn(1,8,n,3,device='cuda')*.1;g[0,:,:,0]=torch.arange(8,device='cuda')[:,None]
ident=[torch.eye(n,device='cuda').repeat(8,1,1),torch.eye(8,device='cuda').repeat(n,1,1)]
shapes={}
def capture(name):
 def hook(mod,args):shapes[name]=list(args[0].shape)
 return hook
handles=[model.tcns[0].register_forward_pre_hook(capture('tcns.0.input')),model.sparse_weighted_adjacency_matrices.interaction_mask.spatial_asymmetric_convolutions[0].register_forward_pre_hook(capture('spatial_mask.conv0.input'))]
with torch.no_grad():model(g,ident)
for h in handles:h.remove()
def decode(x):
 f=model.tcns[0](x)
 for layer in model.tcns[1:]:f=layer(f)+f
 return model.output(f).mean(-2)
x=torch.randn(n,8,4,16,device='cuda',requires_grad=True);p=torch.tensor([2,4,0,3,1],device='cuda');inv=torch.argsort(p)
a=decode(x);b=decode(x[p])[inv];grad=torch.autograd.grad(a[0].sum(),x)[0]
sg_out=dict(shapes=shapes,TXP_node_axis='batch N, not spatial convolution axis',TXP_permutation_max_difference=float((a-b).abs().max()),TXP_other_node_Jacobian_L2=float(grad[1:].norm()),
 correction='Stage5 static suspicion that SGCN TXP directly convolves node indices was too broad. Spatial NN mask still does; do not overwrite old report.',
 full_model_permutation_issue='See original synthetic cases; full model failure is not disproved by isolated decoder equality',official_source_changed=False)
st=json.load(open(root/'SingularTrajectory_git_fallback.json'));normpath=Path(st['source_directory'])/'SingularTrajectory/normalizer.py'
norm=load('stage6_singular_norm',normpath);cases=[]
for speed in [0.,1e-6,1.]:
 h=torch.zeros(1,8,2);h[0,:,0]=torch.arange(8)*.4*speed
 moving=bool(torch.norm((h[:,-1]-h[:,-3])/2)> .3)
 raw=norm.TrajNorm(ori=True,rot=True,sca=True);raw.calculate_params(h)
 actual=norm.TrajNorm(ori=True,rot=True,sca=moving);actual.calculate_params(h);z=actual.normalize(h)
 cases.append(dict(history_speed_mps=speed,official_history_motion_mask=moving,raw_scaling_finite=bool(torch.isfinite(raw.traj_sca).all()),selected_branch_normalized_finite=bool(torch.isfinite(z).all()),
  selected_branch_inverse_max_error_m=float((actual.denormalize(z)-h).abs().max())))
json_new(root/'structural_supplement.json',dict(SGCN=sg_out,SingularTrajectory=dict(scope='Normalizer component only; no trained full-model forward or paper reproduction',normalizer_sha256=sha256(normpath),cases=cases),
 new_training_runs=0,optimizer_updates=0,ETH_evaluated=False))
print('Axis audit and limited normalization probe saved')
