"""Real complete-budget, paired-target, frozen-source, history and bootstrap checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,math,subprocess
import xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
ROOT=Path('results/tables/stage5p5');plan=json.load(open('configs/stage5p5/protocol.json'))
protected=json.load(open('configs/stage5p5/protected_artifacts.json'))
for r in protected['files']:assert sha256(r['path'])==r['sha256'],r['path']
for p,h in json.load(open(ROOT/'frozen_source_sha.json')).items():assert sha256(p)==h,p
for name,h in plan['official_source_sha256'].items():
 p=Path('experiments/stage5/cache/socialvae')/plan['official_commit']/name;assert sha256(p)==h
initial={r['seed']:r['initial_parameter_state_sha256'] for r in json.load(open(ROOT/'initialization_checks.json'))}
seeds={(int(r['training_seed']),r['recording'],int(r['start_frame']),int(r['target_id'])):r for r in read_csv(ROOT/'evaluation_sampling_seeds.csv.gz')}
runs=[];checkpoints=0;commits=[]
for r in plan['formal_runs']:
 seed,model,rec=r['seed'],r['model'],r['recording'];d=ROOT/'runs'/f'seed{seed}'/model/rec
 s=json.load(open(d/'selection.json'));p=json.load(open(d/'provenance.json'));cfg=json.load(open(d/'run_config.json'))
 assert cfg['seed']==seed and cfg['model']==model and cfg['epochs']==100 and cfg['batch_size']==128
 assert s['training_completed'] and s['completed_epochs']==100 and s['seed']==seed and s['model_kind']==model
 assert p['fresh_random_initialization'] and not p['checkpoint_loaded_before_training']
 assert p['initial_parameter_state']['parameter_state_sha256']==initial[seed]
 assert rec not in p['train_recordings'] and p['validation_recordings']==[rec]
 assert 'biwi_eth' not in p['train_recordings']+p['validation_recordings']
 assert p['manifest']['sha256']==plan['manifests'][rec]['sha256']==sha256(plan['manifests'][rec]['path'])
 assert p['plan_sha256']==sha256('configs/stage5p5/protocol.json')
 for recording,source in p['source_files'].items():
  assert recording!='biwi_eth'
  assert sha256(Path(plan['data_root'])/source['file'])==source['sha256']
 curve=read_csv(d/'training_curve.csv');inventory=read_csv(d/'checkpoint_inventory.csv')
 assert [int(c['epoch']) for c in curve]==list(range(1,101))
 expected=min(curve,key=lambda c:(float(c['K1_prior_ADE_m']),int(c['epoch'])))
 assert s['best']['epoch']==int(expected['epoch']) and s['last']['epoch']==100
 assert s['best']['K1_prior_FDE_m']==float(expected['K1_prior_FDE_m'])
 assert {int(c['epoch']) for c in inventory}=={1,25,50,75,100,s['best']['epoch']}
 for c in inventory:assert sha256(c['path'])==c['sha256'];checkpoints+=1
 assert all(int(c['targets_seen'])==p['training_targets'] and int(c['steps'])==p['steps_per_epoch'] and c['parameters_finite']=='True' for c in curve)
 for epoch in range(1,101):
  steps=read_csv(d/f'epochs/epoch{epoch:03d}_steps.csv.gz')
  assert len(steps)==p['steps_per_epoch'] and sum(int(x['targets']) for x in steps)==p['training_targets']
  assert all(math.isfinite(float(x[k])) for x in steps for k in ['loss','rec','kl','gradient_L2'])
  assert all(abs(float(x['loss'])-float(x['rec'])-float(x['kl']))<1e-5 for x in steps)
  targets=read_csv(d/f'epochs/epoch{epoch:03d}_targets.csv.gz');assert len(targets)==p['validation_targets']
  assert len({(t['recording'],t['start_frame'],t['target_id']) for t in targets})==len(targets)
  for t in targets:
   assert t['recording']==rec
   ref=seeds[(seed,rec,int(t['start_frame']),int(t['target_id']))]
   assert t['prior_seed']==ref['prior_seed'] and t['posterior_seed']==ref['posterior_seed']
   assert all(math.isfinite(float(t[k])) for k in ['K1_prior_ADE_m','K1_prior_FDE_m','K1_sample_ADE_m','K1_sample_FDE_m','minADE20_m','minFDE20_m'])
 lifetime=read_csv(d/'lifetime_parameter_gradients.csv')
 assert all(t['dead_parameter_values_unchanged']=='True' for t in lifetime)
 assert sum(int(t['nominal']) for t in lifetime)==2144002 and sum(int(t['structural_effective_upper_bound']) for t in lifetime)==2045698
 assert all(int(t['optimizer_steps_observed'])==s['total_optimizer_steps'] for t in lifetime)
 restores=json.load(open(d/'checkpoint_restore_checks.json'))
 assert [c['epoch'] for c in restores]==[1,25,50,75,100]
 assert all(c['strict_model_restore'] and c['rng_resume_verified'] for c in restores)
 assert s['selected_best_restore']['strict_model_restore'] and s['selected_best_restore']['rng_resume_verified']
 commits.append(p['code_commit']);runs.append(dict(**r,epochs=100,steps=s['total_optimizer_steps'],checkpoints=len(inventory),
  initial_parameter_SHA=initial[seed],actual_run_wall_seconds=s['actual_run_wall_seconds'],code_commit=p['code_commit']))
assert len(set(commits))==1
scheduler=json.load(open(ROOT/'scheduler_completion.json'));assert scheduler['all_succeeded'] and scheduler['launched_runs']==8
assert len([e for e in scheduler['events'] if e['event']=='finish' and e['returncode']==0])==8
index=read_csv(ROOT/'analysis/all_12_run_index.csv');assert len(index)==12
assert {(int(r['seed']),r['model'],r['recording']) for r in index}=={(s,m,r) for s in [17,42,2026] for m in plan['models'] for r in plan['manifests']}
ci=read_csv(ROOT/'bootstrap/all_bootstrap_intervals.csv');assert len(ci)==72 and all(int(r['replicates'])==2000 for r in ci)
for seed in [17,42,2026]:
 for rec in plan['manifests']:
  draws=read_csv(ROOT/f'bootstrap/seed{seed}_{rec}_cluster_draws.csv.gz');assert len(draws)==2000
  assert all(len(r['sampled_person_IDs'].split('|'))==(28 if rec=='biwi_hotel' else 35) for r in draws)
  dist=read_csv(ROOT/f'bootstrap/seed{seed}_{rec}_paired_bootstrap_distribution.csv.gz');assert len(dist)==12000
  assert all(math.isfinite(float(r[k])) for r in dist for k in ['delta_ADE_m','delta_FDE_m'])
xml=ET.parse(ROOT/'final_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
assert sum(int(s.attrib['tests']) for s in suites)>=349
subprocess.run(['git','diff','--check'],check=True)
json_new(ROOT/'delivery_verification.json',dict(protected_files_verified=len(protected['files']),frozen_source_SHA_verified=True,
 official_source_changed=False,Placebo_rule_changed=False,ETH_evaluated=False,model_structure_changed=False,
 raw_nonETH_source_SHA_verified=True,
 completed_new_runs=len(runs),complete_epochs_per_new_run=100,new_run_statistics=runs,new_checkpoints_verified=checkpoints,
 all_12_combinations_indexed=True,same_seed_initial_state_verified=True,paired_noise_verified_every_epoch=True,
 bootstrap_replicates_per_recording_seed=2000,bootstrap_intervals=72,small_cluster_limitation=True,
 single_seed_historical_controls_excluded_from_multiseed=True,unit_tests=sum(int(s.attrib['tests']) for s in suites),unit_test_failures=0,
 additional_training=False,automatic_push=False))
print('PASS: complete8 runs, all12 index, paired seeds/targets, cluster bootstrap, historical SHA and tests')
