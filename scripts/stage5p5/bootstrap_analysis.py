"""Executed paired 2000-draw person-cluster bootstrap, all fixed seeds/families."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from src.socialvae_stability.bootstrap import cluster_errors,bootstrap_seed,draws_to_differences
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from scripts.stage5p3.analyze import FAMILIES
ROOT=Path('results/tables/stage5p5');OUT=ROOT/'bootstrap'
plan=json.load(open('configs/stage5p5/protocol.json'))
rows=[r for r in read_csv(ROOT/'analysis/paired_target_errors.csv.gz') if r['selection']=='best']
OUT.mkdir();summaries=[];metadata=[]
points={(int(r['seed']),r['recording'],r['metric_family'],r['weighting']):r for r in read_csv(ROOT/'analysis/paired_overall_metrics.csv') if r['selection']=='best'}
for seed in [17,42,2026]:
 for rec in ['biwi_hotel','crowds_zara02']:
  common=None;distribution=[]
  for family in FAMILIES:
   rr=[r for r in rows if int(r['seed'])==seed and r['recording']==rec and r['metric_family']==family]
   people,sums,counts=cluster_errors([int(r['target_id']) for r in rr],[float(r['delta_ADE_m']) for r in rr],[float(r['delta_FDE_m']) for r in rr])
   assert len(people)==(28 if rec=='biwi_hotel' else 35)
   if common is None:
    bs=bootstrap_seed(rec,seed,plan['bootstrap']['base_seed'])
    draws=np.random.default_rng(bs).integers(0,len(people),size=(plan['bootstrap']['replicates'],len(people)))
    common=(people,counts.copy(),draws)
    csv_new(OUT/f'seed{seed}_{rec}_cluster_draws.csv.gz',[
     dict(draw=i,bootstrap_seed=bs,sampled_person_IDs='|'.join(str(people[j]) for j in draw)) for i,draw in enumerate(draws)])
    metadata.append(dict(training_seed=seed,recording=rec,bootstrap_seed=bs,person_IDs=people,
     windows_per_person=counts.tolist(),replicates=len(draws),source='best paired target errors; all overlapping windows retained by cluster multiplicity'))
   else:assert people==common[0] and np.array_equal(counts,common[1])
   window,person=draws_to_differences(sums,counts,common[2])
   for weight,dd in [('target-window',window),('equal-person-ID',person)]:
    for i,(a,f) in enumerate(dd):distribution.append(dict(training_seed=seed,recording=rec,metric_family=family,weighting=weight,
     draw=i,bootstrap_seed=bs,delta_ADE_m=float(a),delta_FDE_m=float(f)))
    point=sums.sum(axis=0)/counts.sum() if weight=='target-window' else (sums/counts[:,None]).mean(axis=0)
    ref=points[(seed,rec,family,weight)]
    assert abs(float(ref['delta_ADE_m'])-point[0])<1e-12 and abs(float(ref['delta_FDE_m'])-point[1])<1e-12
    for j,metric in enumerate(['ADE','FDE']):
     lower,upper=np.percentile(dd[:,j],[2.5,97.5])
     summaries.append(dict(training_seed=seed,recording=rec,metric_family=family,weighting=weight,metric=metric,
      target_windows=len(rr),people=len(people),replicates=2000,bootstrap_seed=bs,paired_point_delta_m=float(point[j]),
      bootstrap_mean_delta_m=float(dd[:,j].mean()),bootstrap_SD_m=float(dd[:,j].std(ddof=1)),
      percentile_95_lower_m=float(lower),percentile_95_upper_m=float(upper),
      interval_contains_zero=bool(lower<=0<=upper),fraction_negative_draws=float((dd[:,j]<0).mean()),
      interpretation='descriptive conditional on exposed recording and fixed trained models; not a p-value, scene confirmation, or seed-generalization interval'))
  csv_new(OUT/f'seed{seed}_{rec}_paired_bootstrap_distribution.csv.gz',distribution)
csv_new(OUT/'all_bootstrap_intervals.csv',summaries)
json_new(OUT/'cluster_sampling_metadata.json',metadata)
print('2000 actual paired ID-cluster draws per seed/recording; all families/weightings share their draw matrix')
