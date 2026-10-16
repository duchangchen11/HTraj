"""No formal training: exact initialization, cohort, references and fixed noise."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,shutil
import torch
from src.socialvae_stability.model import new_model
from src.socialvae_stability.evaluation import noise
from src.socialvae_causal.training import seed_all
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.io import csv_new
from src.socialvae_baseline.artifacts import json_new,state_info
from src.data.preprocessing import sha256
plan=json.load(open('configs/stage5p5/protocol.json'));root=Path('results/tables/stage5p5');checks=[]
for seed in [17,42,2026]:
 seed_all(seed);a=new_model('Full','cpu');rng_a=torch.get_rng_state().clone();state_a=state_info(a)
 seed_all(seed);b=new_model('PlaceboSocial','cpu');rng_b=torch.get_rng_state().clone();state_b=state_info(b)
 assert state_a==state_b and torch.equal(rng_a,rng_b)
 checks.append(dict(seed=seed,initial_parameter_state_sha256=state_a['parameter_state_sha256'],nominal_parameters=state_a['parameter_count'],
  Full_Placebo_exactly_equal=True,post_initialization_torch_RNG_equal=True,optimizer_updates=0))
json_new(root/'initialization_checks.json',checks)
for name in ['history_features.csv.gz','cv_paired_targets.csv.gz','cv_metrics.json','context_distribution.csv','context_mask_summary.csv']:
 src=Path('results/tables/stage5p4')/name;dest=root/name
 with dest.open('xb') as f:f.write(src.read_bytes())
 assert sha256(src)==sha256(dest)
seeds=[];index=[]
for rec,manifest in plan['manifests'].items():
 assert sha256(manifest['path'])==manifest['sha256']
 ds=CausalDataset(plan['data_root'],manifest['path'],'validation')
 for i in range(len(ds)):
  item=ds[i]
  for seed in [17,42,2026]:
   prior,ps=noise(item,False,seed);posterior,qs=noise(item,True,seed)
   seeds.append(dict(recording=rec,start_frame=item['start_frame'],target_id=item['actor'],training_seed=seed,
    prior_seed=ps,posterior_seed=qs,prior_noise_sha256=array_sha(prior.numpy()),posterior_noise_sha256=array_sha(posterior.numpy())))
 for kind,source in [('Full',Path('results/tables/stage5p2')),('PlaceboSocial',Path('results/tables/stage5p4/placebo_social'))]:
  source=source/rec;s=json.load(open(source/'selection.json'));assert s['completed_epochs']==100 and s['training_completed']
  for which in ['best','last']:assert sha256(s[which]['path'])==s[which]['sha256']
  index.append(dict(seed=17,model=kind,recording=rec,status='frozen completed reference',directory=str(source),
   best_epoch=s['best']['epoch'],best_checkpoint=s['best']['path'],best_checkpoint_sha256=s['best']['sha256'],
   code_commit=s['code_commit'],manifest_sha256=manifest['sha256']))
for run in plan['formal_runs']:
 index.append(dict(**run,status='preregistered not started',directory=f"results/tables/stage5p5/runs/seed{run['seed']}/{run['model']}/{run['recording']}",
  best_epoch=None,best_checkpoint=None,best_checkpoint_sha256=None,code_commit=None,manifest_sha256=plan['manifests'][run['recording']]['sha256']))
csv_new(root/'evaluation_sampling_seeds.csv.gz',seeds)
csv_new(root/'preregistered_12_run_index.csv',index)
json_new(root/'frozen_reference_metadata_sha.json',{str(Path('results/tables/stage5p4')/n):sha256(Path('results/tables/stage5p4')/n) for n in ['history_features.csv.gz','context_distribution.csv','context_mask_summary.csv']})
print('Preflight passed: common initial states/RNG for 3 seeds; exact frozen inputs; 12-run index; all per-target sampling seeds')
