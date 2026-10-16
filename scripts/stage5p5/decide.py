"""Apply precommitted sign-consistency criteria, never authorize more training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
root=Path('results/tables/stage5p5');plan=json.load(open('configs/stage5p5/protocol.json'))
rr=[r for r in read_csv(root/'analysis/paired_overall_metrics.csv') if r['selection']=='best' and r['metric_family']=='K1-prior']
assert len(rr)==12 and {int(r['seed']) for r in rr}=={17,42,2026}
ade_all=all(float(r['delta_ADE_m'])<0 for r in rr);fde_all=all(float(r['delta_FDE_m'])<0 for r in rr)
cells=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for weight in ['target-window','equal-person-ID']:
  rows=sorted([r for r in rr if r['recording']==rec and r['weighting']==weight],key=lambda r:int(r['seed']))
  a=[float(r['delta_ADE_m']) for r in rows];f=[float(r['delta_FDE_m']) for r in rows]
  cells.append(dict(recording=rec,weighting=weight,seeds=[int(r['seed']) for r in rows],ADE_deltas_m=a,FDE_deltas_m=f,
   ADE_direction_reversal=min(a)<0<max(a),FDE_direction_reversal=min(f)<0<max(f),
   all_seeds_Full_better_ADE=all(d<0 for d in a),all_seeds_Full_better_FDE=all(d<0 for d in f)))
verdict='stronger_cross_initialization_development_evidence' if ade_all and fde_all else 'ADE_only_evidence_with_FDE_conflict' if ade_all else 'benefit_not_stable_or_not_present_stop_complex_social_modules'
json_new(root/'stability_decision.json',dict(frozen_rule=plan['stability'],cells=cells,all_12_ADE_deltas_negative=ade_all,
 all_12_FDE_deltas_negative=fde_all,verdict=verdict,strict_causal_attribution=False,input_distribution_matched=False,
 independent_recording_confirmation=False,bootstrap_changes_decision_rule=False,
 seed_count=3,new_formal_runs=8,additional_training=False,ETH_evaluated=False))
print(json.dumps(dict(verdict=verdict,cells=cells),indent=2))
