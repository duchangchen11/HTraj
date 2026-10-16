"""Twelve frozen-or-new combinations, all histories and paired seed statistics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from collections import defaultdict
import numpy as np
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_baseline.groups import GROUPS
from src.data.preprocessing import sha256
from scripts.stage5p3.analyze import key,means,FAMILIES
ROOT=Path('results/tables/stage5p5');OUT=ROOT/'analysis'
RECS=['biwi_hotel','crowds_zara02'];SEEDS=[17,42,2026];MODELS=['Full','PlaceboSocial']


def directory(seed,model,rec):
 if seed==17:return (Path('results/tables/stage5p2') if model=='Full' else Path('results/tables/stage5p4/placebo_social'))/rec
 return ROOT/'runs'/f'seed{seed}'/model/rec


def main():
 plan=json.load(open('configs/stage5p5/protocol.json'))
 for run in plan['formal_runs']:
  path=directory(run['seed'],run['model'],run['recording'])/'selection.json'
  assert path.exists() and json.load(open(path))['training_completed'],f'Not completed: {path}'
 OUT.mkdir()
 features={key(r):r for r in read_csv(ROOT/'history_features.csv.gz')}
 sampling={(int(r['training_seed']),key(r)):r for r in read_csv(ROOT/'evaluation_sampling_seeds.csv.gz')}
 initial={r['seed']:r['initial_parameter_state_sha256'] for r in json.load(open(ROOT/'initialization_checks.json'))}
 index=[];targets=[];convergence=[];curves_all=[]
 for seed in SEEDS:
  for rec in RECS:
   for model in MODELS:
    d=directory(seed,model,rec);s=json.load(open(d/'selection.json'));p=json.load(open(d/'provenance.json'))
    assert s['training_completed'] and s['completed_epochs']==100
    assert p['seed']==seed and p['initial_parameter_state']['parameter_state_sha256']==initial[seed]
    assert p['manifest']['sha256']==plan['manifests'][rec]['sha256']
    assert rec not in p['train_recordings'] and p['validation_recordings']==[rec]
    assert 'biwi_eth' not in p['train_recordings']+p['validation_recordings']
    curve=[json.load(open(d/f'epochs/epoch{e:03d}.json')) for e in range(1,101)]
    assert min(curve,key=lambda c:(c['K1_prior_ADE_m'],c['epoch']))['epoch']==s['best']['epoch']
    assert all(c['targets_seen']==p['training_targets'] and c['parameters_finite'] for c in curve)
    for c in curve:curves_all.append(dict(seed=seed,model=model,recording=rec,source_directory=str(d),**c))
    for which in ['best','last']:assert sha256(s[which]['path'])==s[which]['sha256']
    index.append(dict(seed=seed,model=model,recording=rec,reference_or_new='frozen seed17' if seed==17 else 'new formal run',
     completed_epochs=100,directory=str(d),configuration_sha256=p['plan_sha256'],manifest_sha256=p['manifest']['sha256'],
     data_source_files=json.dumps(p['source_files'],sort_keys=True),initial_state_sha256=initial[seed],
     training_targets=p['training_targets'],validation_targets=p['validation_targets'],steps_per_epoch=p['steps_per_epoch'],
     optimizer_steps=s['total_optimizer_steps'],best_epoch=s['best']['epoch'],last_epoch=s['last']['epoch'],
     best_checkpoint=s['best']['path'],best_checkpoint_sha256=s['best']['sha256'],last_checkpoint=s['last']['path'],last_checkpoint_sha256=s['last']['sha256'],
     training_curve_path=str(d/'training_curve.csv'),training_curve_sha256=sha256(d/'training_curve.csv'),
     code_commit=p['code_commit'],selection_source_sha256=sha256(d/'selection.json'),actual_run_wall_seconds=s['actual_run_wall_seconds']))
    for which in ['best','last']:
     rr=read_csv(d/f"epochs/epoch{s[which]['epoch']:03d}_targets.csv.gz")
     assert len(rr)==len({key(r) for r in rr})==sum(k[0]==rec for k in features)
     assert {key(r) for r in rr}=={k for k in features if k[0]==rec}
     for r in rr:
      k=key(r);eps=sampling[(seed,k)]
      assert r['prior_seed']==eps['prior_seed'] and r['posterior_seed']==eps['posterior_seed']
      for family,(ak,fk) in FAMILIES.items():
       targets.append(dict(seed=seed,model=model,selection=which,metric_family=family,recording=rec,start_frame=k[1],target_id=k[2],
        ADE_m=float(r[ak]),FDE_m=float(r[fk]),future_position_sha256=features[k]['future_position_sha256'],groups=features[k]['groups']))
    fields=['train_loss','train_rec','train_kl','K1_prior_ADE_m','K1_prior_FDE_m','posterior_loss','minADE20_m','minFDE20_m']
    convergence.append(dict(seed=seed,model=model,recording=rec,best_epoch=s['best']['epoch'],best_within_first5=s['best']['epoch']<=5,
     best_within_last10=s['best']['epoch']>=91,best_validation=s['best_validation'],best_train_K1=s['best_train_K1'],
     best_epoch_curve=curve[s['best']['epoch']-1],last_epoch_curve=curve[-1],
     last20={f:dict(epoch81_90=float(np.mean([c[f] for c in curve[80:90]])),epoch91_100=float(np.mean([c[f] for c in curve[90:]])),
      change=float(np.mean([c[f] for c in curve[90:]])-np.mean([c[f] for c in curve[80:90]])),
      slope_per_epoch=float(np.polyfit(np.arange(81,101),[c[f] for c in curve[80:]],1)[0])) for f in fields},
     actual_run_wall_seconds=s['actual_run_wall_seconds'],peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in curve),
     peak_reserved_bytes=max(c['peak_reserved_bytes'] for c in curve),gradient_L2_max=max(c['gradient_L2_max'] for c in curve),
     numerical_failure=False,full_budget_completed=True,extra_training=False))
 csv_new(OUT/'all_12_run_index.csv',index)
 csv_new(OUT/'all_12_training_curves.csv.gz',curves_all,fieldnames=sorted({k for c in curves_all for k in c}))
 csv_new(OUT/'all_target_errors.csv.gz',targets)
 json_new(OUT/'convergence_audit.json',convergence)
 full={(r['seed'],r['selection'],r['metric_family'],key(r)):r for r in targets if r['model']=='Full'}
 pairs=[]
 for p in targets:
  if p['model']!='PlaceboSocial':continue
  f=full[(p['seed'],p['selection'],p['metric_family'],key(p))]
  assert f['future_position_sha256']==p['future_position_sha256'] and f['groups']==p['groups']
  pairs.append(dict(seed=p['seed'],selection=p['selection'],metric_family=p['metric_family'],recording=p['recording'],start_frame=p['start_frame'],target_id=p['target_id'],
   Full_ADE_m=f['ADE_m'],Full_FDE_m=f['FDE_m'],Placebo_ADE_m=p['ADE_m'],Placebo_FDE_m=p['FDE_m'],
   delta_ADE_m=f['ADE_m']-p['ADE_m'],delta_FDE_m=f['FDE_m']-p['FDE_m'],future_position_sha256=p['future_position_sha256'],groups=p['groups']))
 csv_new(OUT/'paired_target_errors.csv.gz',pairs)
 metrics=[];effects=[]
 def small(n,people):return people<plan['small_group']['person_IDs_less_than'] or n<plan['small_group']['target_windows_less_than']
 for seed in SEEDS:
  for rec in RECS:
   for which in ['best','last']:
    for family in FAMILIES:
     cohort=[r for r in pairs if (r['seed'],r['recording'],r['selection'],r['metric_family'])==(seed,rec,which,family)]
     for group in GROUPS:
      ss=[r for r in cohort if group in r['groups'].split('|')];people=len({r['target_id'] for r in ss})
      for weight in ['target-window','equal-person-ID']:
       vals={}
       for model,prefix in [('Full','Full'),('PlaceboSocial','Placebo')]:
        rr=[dict(recording=rec,target_id=r['target_id'],ADE_m=r[f'{prefix}_ADE_m'],FDE_m=r[f'{prefix}_FDE_m']) for r in ss]
        a,f=means(rr,weight);vals[model]=(a,f)
        metrics.append(dict(seed=seed,model=model,recording=rec,selection=which,metric_family=family,group=group,weighting=weight,
         target_windows=len(ss),people=people,small_group=small(len(ss),people),ADE_m=a,FDE_m=f))
       fa,ff=vals['Full'];pa,pf=vals['PlaceboSocial']
       effects.append(dict(seed=seed,recording=rec,selection=which,metric_family=family,group=group,weighting=weight,target_windows=len(ss),people=people,small_group=small(len(ss),people),
        Full_ADE_m=fa,Full_FDE_m=ff,Placebo_ADE_m=pa,Placebo_FDE_m=pf,delta_ADE_m=None if fa is None else fa-pa,delta_FDE_m=None if ff is None else ff-pf))
 csv_new(OUT/'all_group_model_metrics.csv',metrics)
 csv_new(OUT/'model_metrics.csv',[r for r in metrics if r['group']=='all'])
 csv_new(OUT/'all_group_paired_effects.csv',effects)
 csv_new(OUT/'paired_overall_metrics.csv',[r for r in effects if r['group']=='all'])
 macros=[]
 for seed in SEEDS:
  for which in ['best','last']:
   for family in FAMILIES:
    for group in GROUPS:
     for weight in ['target-window','equal-person-ID']:
      rr=[r for r in effects if (r['seed'],r['selection'],r['metric_family'],r['group'],r['weighting'])==(seed,which,family,group,weight) and r['delta_ADE_m'] is not None]
      macros.append(dict(seed=seed,recording='two-recording-macro',selection=which,metric_family=family,group=group,weighting=weight,
       recordings_with_targets=len(rr),two_recording_macro_available=len(rr)==2,target_windows=sum(r['target_windows'] for r in rr),people=sum(r['people'] for r in rr),small_group=not rr or any(r['small_group'] for r in rr),
       **{field:float(np.mean([r[field] for r in rr])) if len(rr)==2 else None for field in ['Full_ADE_m','Full_FDE_m','Placebo_ADE_m','Placebo_FDE_m','delta_ADE_m','delta_FDE_m']}))
 csv_new(OUT/'recording_macro_paired_metrics.csv',macros)
 stats=[]
 for rec in RECS+['two-recording-macro']:
  source=macros if rec=='two-recording-macro' else effects
  for which in ['best','last']:
   for family in FAMILIES:
    for group in GROUPS:
     for weight in ['target-window','equal-person-ID']:
      selected=[r for r in source if (r['recording'],r['selection'],r['metric_family'],r['group'],r['weighting'])==(rec,which,family,group,weight)]
      assert len(selected)==3
      rr=[r for r in selected if r['delta_ADE_m'] is not None]
      assert len(rr) in [0,3]
      assert len({r['target_windows'] for r in selected})==1 and len({r['people'] for r in selected})==1
      item=dict(recording=rec,selection=which,metric_family=family,group=group,weighting=weight,available_seeds=len(rr),
       target_windows_per_seed=selected[0]['target_windows'],people_per_seed=selected[0]['people'],
       recordings_with_targets_per_seed=selected[0].get('recordings_with_targets',int(selected[0]['target_windows']>0)),
       small_group=not rr or any(r['small_group'] for r in selected))
      for field in ['Full_ADE_m','Full_FDE_m','Placebo_ADE_m','Placebo_FDE_m','delta_ADE_m','delta_FDE_m']:
       v=[r[field] for r in rr]
       item.update({f'{field}_mean':float(np.mean(v)) if v else None,f'{field}_sample_SD':float(np.std(v,ddof=1)) if v else None,
        f'{field}_min':min(v) if v else None,f'{field}_max':max(v) if v else None})
      for field in ['ADE','FDE']:
       v=[r[f'delta_{field}_m'] for r in rr]
       item.update({f'{field}_Full_better_seeds':sum(d<0 for d in v),f'{field}_Placebo_better_seeds':sum(d>0 for d in v),f'{field}_tie_seeds':sum(d==0 for d in v)})
      stats.append(item)
 csv_new(OUT/'multiseed_all_group_statistics.csv',stats)
 csv_new(OUT/'multiseed_overall_statistics.csv',[r for r in stats if r['group']=='all'])
 person=defaultdict(list)
 for r in targets:person[(r['seed'],r['model'],r['recording'],r['selection'],r['metric_family'],r['target_id'])].append(r)
 csv_new(OUT/'person_aggregated_errors.csv',[dict(seed=k[0],model=k[1],recording=k[2],selection=k[3],metric_family=k[4],target_id=k[5],target_windows=len(rr),
  ADE_m=float(np.mean([r['ADE_m'] for r in rr])),FDE_m=float(np.mean([r['FDE_m'] for r in rr]))) for k,rr in sorted(person.items())])
 reference=[]
 for r in read_csv('results/tables/stage5p4/analysis_final/all_group_metrics.csv'):
  if r['model'] in ['NoSocial','EgoCapacity','CV-Last2','CV-OLS4','CV-OLS8'] and r['selection'] in ['analytic','best']:
   reference.append(dict(training_seed='not applicable' if r['model'].startswith('CV-') else '17',scope='historical single-seed or analytic reference only; excluded from 3-seed aggregates',**r))
 csv_new(OUT/'historical_reference_only.csv',reference)
 print('Analyzed all12 complete combinations; three seeds kept separate; historical single-seed controls excluded from aggregates')

if __name__=='__main__':main()
