"""Conditional phase2 final interpretation, using separately frozen thresholds."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
root=Path('results/tables/stage5p4');plan=json.load(open('configs/stage5p4/placebo_protocol.json'));rule=plan['phase2_interpretation']
rows=read_csv(root/'analysis_final/model_metrics.csv');cells=[]
for rec in plan['manifests']:
 for weight in ['target-window','equal-person-ID']:
  def one(model):return next(r for r in rows if (r['model'],r['recording'],r['weighting'],r['selection'],r['metric_family'])==(model,rec,weight,'best','K1-prior'))
  f=one('Full');p=one('PlaceboSocial');fa,ff,pa,pf=map(float,[f['ADE_m'],f['FDE_m'],p['ADE_m'],p['FDE_m']])
  cells.append(dict(recording=rec,weighting=weight,Full_ADE_m=fa,Full_FDE_m=ff,Placebo_ADE_m=pa,Placebo_FDE_m=pf,
   Full_relative_ADE_gain=(pa-fa)/pa,placebo_sufficient=pa<=fa and pf<=ff*(1+rule['FDE_substantial_regression_relative']),
   Full_qualifies=(pa-fa)/pa>=rule['Full_meaningful_relative_ADE_gain'] and ff<=pf*(1+rule['FDE_substantial_regression_relative'])))
sufficient=all(c['placebo_sufficient'] for c in cells);support=all(c['Full_qualifies'] for c in cells)
verdict='target_only_Full_path_sufficient_stop' if sufficient else 'provisional_content_signal_with_distribution_confound' if support else 'conditional_or_conflicting_content_evidence_stop'
json_new(root/'phase2_decision.json',dict(frozen_rule=rule,cells=cells,placebo_sufficient_all_four=sufficient,Full_qualifies_all_four=support,
 verdict=verdict,Placebo_executed=True,extra_training_authorized=False,single_seed=True,independent_confirmation=False,
 exact_function_class_control=False,input_distribution_matched=False,ETH_evaluated=False))
print(json.dumps(dict(verdict=verdict,cells=cells),indent=2))
