"""Descriptive paired comparisons; no iid-window significance claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import numpy as np
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.groups import GROUPS
ROOT=Path('results/tables/stage5p4')
OUT=ROOT/('analysis_final' if (ROOT/'analysis_final').exists() else 'analysis_phase1')
rows=read_csv(OUT/'paired_differences_vs_Full.csv.gz');summaries=[]
variants=sorted({(r['model'],r['selection'],r['metric_family']) for r in rows})
for rec in ['biwi_hotel','crowds_zara02']:
 for model,selection,family in variants:
  cohort=[r for r in rows if (r['recording'],r['model'],r['selection'],r['metric_family'])==(rec,model,selection,family)]
  for group in GROUPS:
   subset=[r for r in cohort if group in r['groups'].split('|')]
   for weighting in ['target-window','equal-person-ID']:
    deltas=[(float(r['model_minus_Full_ADE_m']),float(r['model_minus_Full_FDE_m'])) for r in subset]
    if weighting=='equal-person-ID':
     pp=defaultdict(list)
     for r,d in zip(subset,deltas):pp[r['target_id']].append(d)
     deltas=[np.mean(d,axis=0).tolist() for d in pp.values()]
    summaries.append(dict(recording=rec,model=model,selection=selection,metric_family=family,group=group,weighting=weighting,
     target_windows=len(subset),people=len({r['target_id'] for r in subset}),
     mean_ADE_delta_m=float(np.mean([d[0] for d in deltas])) if deltas else None,
     mean_FDE_delta_m=float(np.mean([d[1] for d in deltas])) if deltas else None,
     median_ADE_delta_m=float(np.median([d[0] for d in deltas])) if deltas else None,
     ADE_lower_than_Full_fraction=float(np.mean([d[0]<0 for d in deltas])) if deltas else None,
     FDE_lower_than_Full_fraction=float(np.mean([d[1]<0 for d in deltas])) if deltas else None))
csv_new(OUT/'all_group_paired_differences.csv',summaries)
