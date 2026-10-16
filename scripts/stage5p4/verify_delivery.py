"""History preservation, complete epochs and source/target/checkpoint acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,math,subprocess
import xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
root=Path('results/tables/stage5p4');plan=json.load(open('configs/stage5p4/ego_protocol.json'))
protected=json.load(open('configs/stage5p4/protected_artifacts.json'))
for row in protected['files']:assert sha256(row['path'])==row['sha256'],row['path']
for p,h in json.load(open(root/'ego_frozen_source_sha.json')).items():assert sha256(p)==h,p
for p,h in plan['official_source_sha256'].items():
 source=Path('experiments/stage5/cache/socialvae')/plan['official_commit']/p
 assert sha256(source)==h
initial=[];code=[];checkpoints=0;run_stats=[]
for variant in ['ego_capacity','placebo_social']:
 if variant=='placebo_social' and not (root/variant/'biwi_hotel/selection.json').exists():continue
 for rec in plan['manifests']:
  d=root/variant/rec;s=json.load(open(d/'selection.json'));p=json.load(open(d/'provenance.json'))
  assert s['training_completed'] and s['completed_epochs']==100
  assert rec not in p['train_recordings'] and p['validation_recordings']==[rec]
  assert 'biwi_eth' not in p['train_recordings']+p['validation_recordings']
  assert p['fresh_random_initialization'] and not p['checkpoint_loaded_before_training']
  assert p['manifest']['sha256']==plan['manifests'][rec]['sha256']==sha256(plan['manifests'][rec]['path'])
  curve=read_csv(d/'training_curve.csv');inventory=read_csv(d/'checkpoint_inventory.csv');saved={int(c['epoch']) for c in inventory}
  assert [int(c['epoch']) for c in curve]==list(range(1,101))
  expected_saved={1,*range(10,101,10)};best=float('inf')
  for c in curve:
   if float(c['K1_prior_ADE_m'])<best:expected_saved.add(int(c['epoch']));best=float(c['K1_prior_ADE_m'])
  assert saved==expected_saved
  assert all(int(c['targets_seen'])==p['training_targets'] and int(c['steps'])==p['steps_per_epoch'] and c['parameters_finite']=='True' for c in curve)
  for checkpoint in inventory:assert sha256(checkpoint['path'])==checkpoint['sha256'];checkpoints+=1
  for epoch in range(1,101):
   steps=read_csv(d/f'epochs/epoch{epoch:03d}_steps.csv.gz')
   assert len(steps)==p['steps_per_epoch'] and sum(int(t['targets']) for t in steps)==p['training_targets']
   assert all(math.isfinite(float(t[k])) for t in steps for k in ['loss','rec','kl','gradient_L2'])
   assert all(abs(float(t['loss'])-float(t['rec'])-float(t['kl']))<1e-5 for t in steps)
   targets=read_csv(d/f'epochs/epoch{epoch:03d}_targets.csv.gz');assert len(targets)==p['validation_targets']
   assert all(math.isfinite(float(t[k])) for t in targets for k in ['K1_prior_ADE_m','K1_prior_FDE_m','K1_sample_ADE_m','K1_sample_FDE_m','minADE20_m','minFDE20_m'])
  expected=min(curve,key=lambda c:(float(c['K1_prior_ADE_m']),int(c['epoch'])))
  assert int(expected['epoch'])==s['best']['epoch'] and float(expected['K1_prior_FDE_m'])==s['best']['K1_prior_FDE_m']
  assert all(c['strict_model_restore'] and c['rng_resume_verified'] for c in json.load(open(d/'checkpoint_restore_checks.json')))
  lifetime=read_csv(d/'lifetime_parameter_gradients.csv')
  assert all(r['dead_parameter_values_unchanged']=='True' for r in lifetime)
  assert sum(int(r['structural_effective_upper_bound']) for r in lifetime)==(1985474 if variant=='ego_capacity' else 2045698)
  assert all(int(r['optimizer_steps_observed'])==s['total_optimizer_steps'] for r in lifetime)
  if variant=='ego_capacity':
   assert all(int(r['ever_nonzero_gradient_scalars'])>0 for r in lifetime if r['parameter'].startswith('ego_capacity.'))
  initial.append((variant,p['initial_parameter_state']['parameter_state_sha256']));code.append((variant,p['code_commit']))
  run_stats.append(dict(variant=variant,recording=rec,steps=s['total_optimizer_steps'],saved_checkpoints=len(inventory),
   nominal=sum(int(r['nominal']) for r in lifetime),structural_upper_bound=sum(int(r['structural_effective_upper_bound']) for r in lifetime),
   module_loss_connected_ever=sum(int(r['nominal']) for r in lifetime if r['gradient_connected_ever']=='True'),
   lifetime_ever_nonzero_gradient=sum(int(r['ever_nonzero_gradient_scalars']) for r in lifetime),
   lifetime_never_nonzero_gradient=sum(int(r['never_nonzero_gradient_scalars']) for r in lifetime)))
for variant in {v for v,c in initial}:
 assert len({c for v,c in initial if v==variant})==1
 assert len({c for v,c in code if v==variant})==1
xml=ET.parse(root/'final_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
trained=json.load(open(root/'trained_isolation_audit.json'));assert all(r['strictly_equal'] for r in trained)
subprocess.run(['git','diff','--check'],check=True)
json_new(root/'delivery_verification.json',dict(protected_files_verified=len(protected['files']),run_statistics=run_stats,
 checkpoints_verified=checkpoints,complete_epochs_per_run=100,ETH_evaluated=False,official_source_changed=False,
 frozen_ego_source_sha_verified=True,unit_tests=sum(int(s.attrib['tests']) for s in suites),unit_test_failures=0,
 Full_results_frozen=True,NoSocial_results_frozen=True,trained_EgoCapacity_isolation_verified=True,
 scalar_upper_bound_matched_within_5percent=True,function_classes_equal=False,local_commit_only=True))
print('PASS: historical artifacts, frozen sources, complete budgets, checkpoints and tests')
