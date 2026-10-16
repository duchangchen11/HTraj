"""Combine frozen and new participation audits with exact provenance labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import csv_new,read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
root=Path('results/tables/stage5p4');rows=[]
final=(root/'placebo_social/biwi_hotel/selection.json').exists()
suffix='_final' if final else ''
old=json.load(open('results/tables/stage5p3/capacity_audit.json'))
for r in old+json.load(open(root/'capacity_audit.json'))+(json.load(open(root/'placebo_capacity_audit.json')) if final else []):
 rows.append(dict(model=r['model'],recording='64_target_fixed_training_probe',nominal=r['total_parameters'],
  module_forward_used=r['module_forward_used_parameters'],module_loss_connected=r['loss_graph_connected_parameters'],
  structural_upper_bound=r['structural_effective_upper_bound'],observed_nonzero=r['observed_nonzero_gradient_scalars'],
  observed_zero=r['total_parameters']-r['observed_nonzero_gradient_scalars'],
  added_ego_nonzero=None,interpretation='four no-update probe batches; not lifetime, rank, or exact function capacity'))
for model,path in [('NoSocial',Path('results/tables/stage5p3')),('EgoCapacity',root/'ego_capacity'),('PlaceboSocial',root/'placebo_social')]:
 for rec in ['biwi_hotel','crowds_zara02']:
  p=path/rec/'lifetime_parameter_gradients.csv'
  if not p.exists():continue
  rr=read_csv(p)
  rows.append(dict(model=model,recording=rec,nominal=sum(int(r['nominal']) for r in rr),
   module_forward_used=None,module_loss_connected=sum(int(r['nominal']) for r in rr if r['gradient_connected_ever']=='True'),
   structural_upper_bound=sum(int(r['structural_effective_upper_bound']) for r in rr),
   observed_nonzero=sum(int(r['ever_nonzero_gradient_scalars']) for r in rr),
   observed_zero=sum(int(r['never_nonzero_gradient_scalars']) for r in rr),
   added_ego_nonzero=sum(int(r['ever_nonzero_gradient_scalars']) for r in rr if r['parameter'].startswith('ego_capacity.')),
   interpretation='100 complete epochs: exact union of observed nonzero gradients; not permanent theoretical inactivity'))
csv_new(root/f'combined_capacity_audit{suffix}.csv',rows)
json_new(root/f'capacity_audit_sources{suffix}.json',dict(
 Full_NoSocial_probe_source='results/tables/stage5p3/capacity_audit.json',
 Full_NoSocial_probe_sha256=sha256('results/tables/stage5p3/capacity_audit.json'),
 Full_lifetime_exact_gradient_union_available=False,
 Full_optimizer_moments_proxy_source='results/tables/stage5p3/full_optimizer_capacity_proxy.csv',
 Full_proxy_is_not_exact_gradient_union=True,
 EgoCapacity_probe_source='results/tables/stage5p4/capacity_audit.json',
 parameter_masks_applied_to_weights=False,function_class_equality_claim=False))
print('Combined capacity observations saved, Full lifetime unknown explicitly retained')
