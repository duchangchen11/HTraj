"""Restore frozen K1 paths, verify paired scalars, then describe motion failures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gzip,json,time,subprocess
from collections import defaultdict
import numpy as np
import torch
from src.data.preprocessing import sha256
from src.socialvae_causal.data import CausalDataset,collate_examples,array_sha
from src.socialvae_causal.evaluation import cv_prediction,trajectory_errors
from src.socialvae_stability.model import new_model
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from src.evaluation.stage6_motion import diagnose,history_features

ROOT=Path('results/tables/stage6');OUT=ROOT/'diagnosis'

def key(r):return r['recording'],int(r['start_frame']),int(r.get('target_id',r.get('actor')))
def write(name,rows):csv_new(OUT/name,rows,fieldnames=sorted({k for r in rows for k in r}))

def main():
 begin=time.perf_counter();OUT.mkdir(parents=True)
 plan=json.load(open('configs/stage6/diagnostic_protocol.json'))
 index=read_csv('results/tables/stage5p5/analysis/all_12_run_index.csv')
 saved={(int(r['seed']),r['model'],key(r)):r for r in read_csv('results/tables/stage5p5/analysis/all_target_errors.csv.gz') if r['selection']=='best' and r['metric_family']=='K1-prior'}
 features={key(r):r for r in read_csv('results/tables/stage5p5/history_features.csv.gz')}
 datasets={};history_rows=[];runs=[];rows=[];steps=[];max_difference=0.
 cache=Path('experiments/stage6/cache');cache.mkdir(parents=True,exist_ok=True)
 packet_path=cache/'frozen_k1_trajectory_packets.jsonl.gz'
 packet=gzip.GzipFile(filename='',mode='wb',fileobj=packet_path.open('xb'),mtime=0)
 def consume(item,prediction,model,seed,registered=None):
  nonlocal max_difference
  k=key(item);h=item['x'][:,:2];y=item['y'];f=features[k]
  assert array_sha(y)==f['future_position_sha256']
  d,ss=diagnose(h,y,prediction)
  if registered is not None:
   diff=max(abs(d['ADE_m']-float(registered['ADE_m'])),abs(d['FDE_m']-float(registered['FDE_m'])))
   max_difference=max(max_difference,diff);assert diff<=plan['inference_match_tolerance_m'],(k,model,seed,diff)
  common=dict(recording=k[0],start_frame=k[1],target_id=k[2],model=model,seed=seed)
  groups=f['groups']+'|heading_'+d['history_heading_bin']
  rows.append(dict(**common,groups=groups,future_position_sha256=f['future_position_sha256'],**d))
  for s in ss:steps.append(dict(**common,groups=groups,**s))
  packet.write((json.dumps(dict(**common,history=h.tolist(),truth=y.tolist(),prediction=np.asarray(prediction).tolist()),allow_nan=False)+'\n').encode())
 for rec,manifest in plan['manifests'].items():
  assert sha256(manifest['path'])==manifest['sha256']
  val=CausalDataset(plan['data_root'],manifest['path'],'validation');datasets[rec]=val
  for split,dataset in [('validation',val),('train',CausalDataset(plan['data_root'],manifest['path'],'train'))]:
   for i in range(len(dataset)):
    item=dataset[i];history_rows.append(dict(fold=rec,split=split,recording=item['recording'],target_id=item['actor'],start_frame=item['start_frame'],**history_features(item['x'][:,:2])))
  for i in range(len(val)):
   item=val[i]
   for model in ['CV-Last2','CV-OLS4','CV-OLS8']:consume(item,cv_prediction(item['x'][:,:2],model),model,'analytic')
 for r in index:
  tick=time.perf_counter();rec=r['recording'];seed=int(r['seed']);kind=r['model'];dataset=datasets[rec]
  assert sha256(r['best_checkpoint'])==r['best_checkpoint_sha256']
  checkpoint=torch.load(r['best_checkpoint'],map_location='cpu',weights_only=True)
  model=new_model(kind);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
  torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():
   for start in range(0,len(dataset),128):
    items=[dataset[i] for i in range(start,min(start+128,len(dataset)))];x,y,n=collate_examples(items,'cuda')
    prediction=model(x,n);assert torch.isfinite(prediction).all()
    a,f=trajectory_errors(prediction[None],y)
    for j,item in enumerate(items):
     ref=saved[(seed,kind,key(item))]
     assert abs(float(a[0,j])-float(ref['ADE_m']))<=2e-6 and abs(float(f[0,j])-float(ref['FDE_m']))<=2e-6
     consume(item,prediction[:,j].cpu().numpy(),kind,seed,ref)
  runs.append(dict(seed=seed,model=kind,recording=rec,best_epoch=int(r['best_epoch']),checkpoint=r['best_checkpoint'],checkpoint_sha256=r['best_checkpoint_sha256'],
   strict_load=True,targets=len(dataset),elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0))
  del model,checkpoint;torch.cuda.empty_cache()
  print('Restored',rec,seed,kind,len(dataset),'targets',flush=True)
 packet.close()
 cvs={(r['model'],key(r)):r for r in rows if r['seed']=='analytic'}
 for r in rows:
  for cv in ['CV-Last2','CV-OLS4','CV-OLS8']:
   r['delta_ADE_vs_'+cv+'_m']=r['ADE_m']-cvs[(cv,key(r))]['ADE_m'];r['delta_FDE_vs_'+cv+'_m']=r['FDE_m']-cvs[(cv,key(r))]['FDE_m']
 write('per_target_motion_diagnostics.csv.gz',rows);write('per_step_motion_diagnostics.csv.gz',steps);write('history_only_train_validation.csv.gz',history_rows)
 grouped=defaultdict(list)
 for r in rows:
  for group in r['groups'].split('|'):grouped[(r['recording'],r['seed'],r['model'],group)].append(r)
 metrics=[];tags=[k for k in rows[0] if k.startswith('tag_')]
 for k,rr in grouped.items():
  people=defaultdict(list)
  for r in rr:people[r['target_id']].append(r)
  fields=['ADE_m','FDE_m','predicted_mean_speed_mps','truth_mean_speed_mps','speed_bias_mps','predicted_early_speed_mps','predicted_late_speed_mps','truth_early_speed_mps','truth_late_speed_mps']+[p for p in rr[0] if p.startswith('delta_')]
  for weight in ['target-window','equal-person-ID']:
   def agg(f):return float(np.mean([r[f] for r in rr])) if weight=='target-window' else float(np.mean([np.mean([r[f] for r in pp]) for pp in people.values()]))
   metrics.append(dict(recording=k[0],seed=k[1],model=k[2],group=k[3],weighting=weight,targets=len(rr),people=len(people),small_group=len(rr)<30 or len(people)<10,
    **{f:agg(f) for f in fields},**{t+'_rate':agg(t) for t in tags}))
 write('all_group_diagnostics.csv',metrics)
 profiles=[]
 bystep=defaultdict(list)
 for r in steps:bystep[(r['recording'],r['seed'],r['model'],r['step'])].append(r)
 for k,rr in bystep.items():
  pp=defaultdict(list)
  for r in rr:pp[r['target_id']].append(r)
  for weight in ['target-window','equal-person-ID']:
   fields=['error_m','predicted_speed_mps','truth_speed_mps','speed_bias_mps','velocity_vector_error_mps']
   v={f:float(np.mean([r[f] for r in rr])) if weight=='target-window' else float(np.mean([np.mean([r[f] for r in p]) for p in pp.values()])) for f in fields}
   profiles.append(dict(recording=k[0],seed=k[1],model=k[2],step=k[3],horizon_seconds=k[3]*.4,weighting=weight,targets=len(rr),people=len(pp),**v))
 write('per_horizon_aggregates.csv',profiles)
 ids=defaultdict(list);idrows=[]
 for r in rows:ids[(r['recording'],r['seed'],r['model'],r['target_id'])].append(r)
 for k,rr in ids.items():
  idrows.append(dict(recording=k[0],seed=k[1],model=k[2],target_id=k[3],windows=len(rr),ADE_m=float(np.mean([r['ADE_m'] for r in rr])),FDE_m=float(np.mean([r['FDE_m'] for r in rr])),
   total_ADE_error_m=sum(r['ADE_m'] for r in rr),signed_excess_OLS4_m=sum(r['delta_ADE_vs_CV-OLS4_m'] for r in rr),positive_excess_OLS4_m=sum(max(r['delta_ADE_vs_CV-OLS4_m'],0) for r in rr),
   **{f:float(np.mean([r[f] for r in rr])) for f in ['history_mean_speed_mps','predicted_mean_speed_mps','truth_mean_speed_mps','speed_bias_mps']}))
 write('per_person_contributions.csv',idrows)
 concentrations=[]
 for rec,seed,model,_ in ids:
  if any(c['recording']==rec and c['seed']==seed and c['model']==model for c in concentrations):continue
  rr=[r for r in idrows if (r['recording'],r['seed'],r['model'])==(rec,seed,model)]
  for n in [1,3,5]:
   total=sum(r['total_ADE_error_m'] for r in rr);pos=sum(r['positive_excess_OLS4_m'] for r in rr)
   top=sorted(rr,key=lambda r:r['total_ADE_error_m'],reverse=True)[:n];excess=sorted(rr,key=lambda r:r['positive_excess_OLS4_m'],reverse=True)[:n]
   concentrations.append(dict(recording=rec,seed=seed,model=model,top_n=n,people=len(rr),top_ADE_IDs='|'.join(str(r['target_id']) for r in top),
    ADE_share=sum(r['total_ADE_error_m'] for r in top)/total if total else None,positive_excess_OLS4_IDs='|'.join(str(r['target_id']) for r in excess),
    positive_excess_OLS4_share=sum(r['positive_excess_OLS4_m'] for r in excess)/pos if pos else None))
 write('error_concentration.csv',concentrations)
 distributions=[]
 hh=defaultdict(list)
 for r in history_rows:hh[(r['fold'],r['split'])].append(r)
 for k,rr in hh.items():
  for group in ['all']+sorted({r['recording'] for r in rr}):
   sub=rr if group=='all' else [r for r in rr if r['recording']==group]
   v=np.array([r['history_mean_speed_mps'] for r in sub]);ds=np.array([r['history_directness'] for r in sub])
   distributions.append(dict(fold=k[0],split=k[1],source=group,targets=len(sub),people=len({(r['recording'],r['target_id']) for r in sub}),
    speed_mean=float(v.mean()),speed_P50=float(np.median(v)),speed_P95=float(np.percentile(v,95)),stationary_fraction=float((v<.05).mean()),slow_fraction=float(((v>=.05)&(v<.4)).mean()),normal_fraction=float(((v>=.4)&(v<1.8)).mean()),fast_fraction=float((v>=1.8).mean()),
    directness_mean=float(ds.mean()),**{'heading_'+d+'_fraction':sum(r['history_heading_bin']==d for r in sub)/len(sub) for d in ['E','NE','N','NW','W','SW','S','SE','unreliable']}))
 write('history_distribution_comparison.csv',distributions)
 inputs=['configs/stage6/diagnostic_protocol.json','results/tables/stage5p5/analysis/all_12_run_index.csv','results/tables/stage5p5/analysis/all_target_errors.csv.gz','results/tables/stage5p5/history_features.csv.gz']
 source=['scripts/stage6/diagnose.py','src/evaluation/stage6_motion.py','src/socialvae_stability/model.py','src/socialvae_stability/evaluation.py','src/socialvae_causal/data.py']
 json_new(OUT/'execution.json',dict(inference_runs=runs,restored_combinations=len(runs),new_training_runs=0,optimizer_updates=0,ETH_evaluated=False,
  maximum_error_match_difference_m=max_difference,trajectory_packet_path=str(packet_path),trajectory_packet_sha256=sha256(packet_path),
  code_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA={p:sha256(p) for p in source},input_SHA={p:sha256(p) for p in inputs},elapsed_seconds=time.perf_counter()-begin,
  tags='Post-hoc outcome diagnostics, overlapping; no causal proof or predictive gate',original_prediction_coordinates_saved_in_stage5p5=False))
 print('All frozen predictions verified; actual diagnostics saved; no training')

if __name__=='__main__':main()
