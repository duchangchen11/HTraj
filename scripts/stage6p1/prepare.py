"""Freeze the only sampling experiment, protected history and source-only pools."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,math
import numpy as np
import torch
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new,state_info
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.io import csv_new,read_csv
from src.socialvae_causal.training import seed_all
from src.socialvae_attribution.model import new_model,structural_parameter_masks
from src.socialvae_baseline.evaluation import noise
from src.motion_sampling.policy import classify_history,probabilities,NAMES,epoch_order,exposure

ROOT=Path('results/tables/stage6p1')
BASE='af7da3ffda0d65ec8ff1e05a5d0235d62cd53dad'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/stage6p1-motion-sampling-pilot'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage6/protected_artifacts.json'))['files']}
for extension in ['*.pth','*.pt','*.ckpt']:
 paths|={str(p) for p in Path('experiments').glob('**/checkpoints/**/'+extension) if p.is_file()}
for name in ['SGCN','SingularTrajectory']:
 paths|={r['local_path'] for r in json.load(open(f'results/tables/stage6/public/{name}_git_fallback.json'))['files']}
old_exec=json.load(open('results/tables/stage6/diagnosis/execution.json'))
paths|={old_exec['trajectory_packet_path'],'results/tables/stage6/diagnosis/per_step_motion_diagnostics.csv.gz'}
json_new('configs/stage6p1/protected_artifacts.json',dict(base_commit=BASE,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
old=json.load(open('configs/stage5p3/protocol.json'))
plan=dict(stage='6.1',base_commit=BASE,branch='feat/stage6p1-motion-sampling-pilot',
 model='Frozen Stage5.3 SocialVAE-NoSocial',model_source='src/socialvae_attribution/model.py',
 official_commit=old['official_commit'],official_source_sha256=old['official_source_sha256'],
 protocol=old['protocol'],manifests=old['manifests'],data_root=old['data_root'],
 seed=17,epochs=30,batch_size=128,optimizer='Adam',learning_rate=.0005,
 weight_decay=0.,scheduler=None,gradient_clip=None,hidden_dim=256,ob_radius_m=2.,
 loss=old['loss'],model_selection=old['model_selection'],evaluation=old['evaluation'],
 training_order=dict(Uniform='numpy default_rng(17+epoch).permutation(N), no replacement',
  Stratified='numpy default_rng(17+epoch).choice(N,size=N,replace=True,p=P)',
  probability='P(i)=0.5/N+0.5/(G*N_g); G counts only nonempty history-speed classes',
  speed='mean norm of seven consecutive raw historical displacements / 0.4 seconds; float64',
  thresholds_mps=[.05,.4,1.8],classes=list(NAMES),
  changes='Same N visits/ceil(N/128) updates per epoch; repeated target windows and unique coverage differ; no target-future input'),
 formal_runs=[dict(name=f'Run{i+1}',recording=r,condition=c,seed=17) for i,(r,c) in enumerate([
  ('biwi_hotel','Uniform'),('biwi_hotel','Stratified'),('crowds_zara02','Uniform'),('crowds_zara02','Stratified')])],
 initialization='seed_all(17), same frozen constructor; random weights, no old checkpoint initialization; exact state/RNG audit',
 smoke='Epochs1..5 within same run; if all diagnostics/strict restore finite, continue unchanged to30; never restart from smoke',
 checkpoint_restore_epochs=[1,5,10,20,30],
 checkpoint_saving='Immutable periodic epochs1/5/10/20/30 plus best snapshot serialized in RAM and written once at completion; last=30; all epoch curves retained; no overwrite',
 train_K1_epochs=[1,5,10,20,30],
 stopping=dict(ADE_gain_abs_m=.005,ADE_gain_relative=.01,
  FDE_max_regression_abs_m=.005,FDE_max_regression_relative=.01,
  speed_bias_reduction_relative=.10,speed_gate='HOTEL speed_normal window-weighted abs(mean signed bias), also report mean(abs(per-target bias)) and ID weights',
  stationary_FDE_max_regression_abs_m=.01,stationary_FDE_max_regression_relative=.05,
  weightings=['target-window','equal-person-ID'],all_conditions_required=True,
  best_epoch_30='budget truncation; no extension',extra_seeds=False,extra_epochs=False,
  failure='Stop current hypothesis on any failed gate; no ratio/threshold/model/seed/budget search'),
 scheduling=dict(max_concurrent=4,rule='exactly four independent processes, shared GPU; wall time includes contention, no settings depend on results'),
 ETH_evaluated=False,independent_confirmation=False,FPC=False,new_network=False,automatic_push=False,
 failure_policy='First nonfinite stops affected run, save failure model/optimizer/RNG/batch and traceback; no retry. Other preregistered runs may complete.')
plan['evaluation']['train_K1_epochs']=plan['train_K1_epochs']
json_new('configs/stage6p1/protocol.json',plan)
checks=[];distributions=[];sampling_seeds=[]
for rec,manifest in plan['manifests'].items():
 assert sha256(manifest['path'])==manifest['sha256']
 for split in ['train','validation']:
  ds=CausalDataset(plan['data_root'],manifest['path'],split)
  reference=read_csv(f'results/tables/stage5p1/manifests/{rec}/B_{split}_targets.csv.gz')
  assert len(reference)==len(ds)
  rows=[]
  for idx,ref in enumerate(reference):
   wi,ci=ds.indices[idx];context=ds.contexts[wi];window=ds.windows[wi]
   h=context.state[:,ci,:2]
   category,speed=classify_history(h)  # No future or neighbor argument.
   item=ds[idx]  # Separate integrity audit only: labels do not determine category/probability.
   assert (item['recording'],item['start_frame'],item['actor'])==(ref['recording'],int(ref['start_frame']),int(ref['target_id']))
   assert array_sha(h)==ref['history_sha256'] and array_sha(item['y'])==ref['future_sha256']
   rows.append(dict(target_index=idx,recording=item['recording'],start_frame=item['start_frame'],
    target_id=item['actor'],history_speed_mps=speed,category_index=category,category=NAMES[category],
    history_sha256=ref['history_sha256'],future_sha256=ref['future_sha256']))
   if split=='validation':
    prior,ps=noise(item);post,qs=noise(item,True)
    sampling_seeds.append(dict(recording=rec,start_frame=item['start_frame'],target_id=item['actor'],
     prior_seed=ps,posterior_seed=qs,prior_noise_sha256=array_sha(prior.numpy()),posterior_noise_sha256=array_sha(post.numpy())))
  labels=np.array([r['category_index'] for r in rows]);p=probabilities(labels)
  for i,row in enumerate(rows):row['Stratified_probability']=float(p[i]) if split=='train' else None
  csv_new(ROOT/f'pools/{rec}_{split}.csv.gz',rows)
  for g,name in enumerate(NAMES):
   n=int((labels==g).sum())
   distributions.append(dict(fold=rec,split=split,category=name,targets=n,total=len(ds),
    original_fraction=n/len(ds),expected_Stratified_fraction=float(p[labels==g].sum()) if split=='train' else None))
  checks.append(dict(recording=rec,split=split,target_windows=len(ds),
   person_IDs=len({(r['recording'],r['target_id']) for r in rows}),records=sorted(ds.records),
   manifest_sha256=manifest['sha256'],all_target_IDs_history_GT_SHA_equal_Stage5p1=True,
   pool_file_sha256=sha256(ROOT/f'pools/{rec}_{split}.csv.gz'),optimizer_steps_per_epoch=math.ceil(len(ds)/128) if split=='train' else None))
  if split=='train':assert rec not in ds.records
  else:assert set(ds.records)=={rec}
csv_new(ROOT/'source_class_distribution.csv',distributions)
csv_new(ROOT/'evaluation_sampling_seeds.csv.gz',sampling_seeds)
json_new(ROOT/'cohort_verification.json',checks)
seed_all(17);a=new_model(device='cpu');sa=state_info(a);rng=torch.get_rng_state().clone()
seed_all(17);b=new_model(device='cpu');sb=state_info(b)
assert sa==sb and torch.equal(rng,torch.get_rng_state())
masks=structural_parameter_masks(a,'NoSocial')
json_new(ROOT/'initialization_check.json',dict(**sa,conditions_same_initial_state=True,post_init_torch_RNG_equal=True,
 structurally_effective_scalar_upper_bound=sum(int(m.sum()) for m in masks.values()),optimizer_updates=0,
 frozen_model_source_sha256=sha256('src/socialvae_attribution/model.py')))
for name in ['history_features.csv.gz','cv_paired_targets.csv.gz','cv_metrics.json']:
 src=Path('results/tables/stage5p3')/name;dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(src.read_bytes())
 assert sha256(src)==sha256(dest)
print('Frozen unique sampling policy; all source/val IDs/history/GT verified; history files protected:',len(paths),flush=True)
