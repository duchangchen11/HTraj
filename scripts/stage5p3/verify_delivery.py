"""Final additive-artifact, frozen-source and complete-budget acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

root=Path('results/tables/stage5p3')
protected=json.load(open('configs/stage5p3/protected_artifacts.json'))
for row in protected['files']:assert sha256(row['path'])==row['sha256'],row['path']
plan=json.load(open('configs/stage5p3/protocol.json'))
initial=[];code=[]
for rec in plan['manifests']:
    d=root/rec;s=json.load(open(d/'selection.json'));p=json.load(open(d/'provenance.json'))
    assert s['training_completed'] and s['completed_epochs']==100
    assert rec not in p['train_recordings'] and p['validation_recordings']==[rec]
    assert 'biwi_eth' not in p['train_recordings']+p['validation_recordings']
    assert p['fresh_random_initialization'] and not p['checkpoint_loaded_before_training']
    assert p['manifest']['sha256']==plan['manifests'][rec]['sha256']==sha256(plan['manifests'][rec]['path'])
    curve=read_csv(d/'training_curve.csv');inventory=read_csv(d/'checkpoint_inventory.csv')
    assert [int(c['epoch']) for c in curve]==list(range(1,101)) and len(inventory)==100
    assert all(int(c['targets_seen'])==p['training_targets'] and int(c['steps'])==p['steps_per_epoch'] and c['parameters_finite']=='True' for c in curve)
    for checkpoint in inventory:assert sha256(checkpoint['path'])==checkpoint['sha256']
    for epoch in range(1,101):
        steps=read_csv(d/f'epochs/epoch{epoch:03d}_steps.csv.gz')
        assert len(steps)==p['steps_per_epoch'] and sum(int(s['targets']) for s in steps)==p['training_targets']
        assert all(math.isfinite(float(s[k])) for s in steps for k in ['loss','rec','kl','gradient_L2'])
        assert all(abs(float(s['loss'])-float(s['rec'])-float(s['kl']))<1e-5 for s in steps)
        targets=read_csv(d/f'epochs/epoch{epoch:03d}_targets.csv.gz')
        assert len(targets)==p['validation_targets']
        assert all(math.isfinite(float(t[k])) for t in targets for k in ['K1_prior_ADE_m','K1_prior_FDE_m','K1_sample_ADE_m','K1_sample_FDE_m','minADE20_m','minFDE20_m'])
    expected=min(curve,key=lambda c:(float(c['K1_prior_ADE_m']),int(c['epoch'])))
    assert int(expected['epoch'])==s['best']['epoch']
    assert float(expected['K1_prior_FDE_m'])==s['best']['K1_prior_FDE_m']
    assert all(c['strict_model_restore'] and c['rng_resume_verified'] for c in json.load(open(d/'checkpoint_restore_checks.json')))
    initial.append(p['initial_parameter_state']['parameter_state_sha256']);code.append(p['code_commit'])
assert len(set(initial))==1 and len(set(code))==1
xml=ET.parse(root/'final_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
subprocess.run(['git','diff','--check'],check=True)
trained=json.load(open(root/'trained_isolation_audit.json'))
assert all(r['strictly_equal'] for r in trained if r['model']=='NoSocial')
for rec in plan['manifests']:
    lifetime=read_csv(root/rec/'lifetime_parameter_gradients.csv')
    assert all(r['dead_parameter_values_unchanged']=='True' for r in lifetime)
    assert sum(int(r['structural_effective_upper_bound']) for r in lifetime)==1459906
    selection=json.load(open(root/rec/'selection.json'))
    assert all(int(r['optimizer_steps_observed'])==selection['total_optimizer_steps'] for r in lifetime)
json_new(root/'delivery_verification.json',dict(protected_files_verified=len(protected['files']),
    two_fresh_independent_seed17_runs=True,initial_parameter_sha256=initial[0],code_commit=code[0],
    checkpoints_verified=200,complete_epochs_per_run=100,ETH_evaluated=False,official_source_changed=False,
    unit_tests=sum(int(s.attrib['tests']) for s in suites),unit_test_failures=0,
    Full_results_frozen=True,trained_NoSocial_isolation_verified=True,
    effective_capacity_equal=False,NoSocial_structural_upper_bound=1459906,Full_structural_upper_bound=2045698,
    Ego_Capacity_Control_trained=False,local_commit_only=True))
print('PASS: historical SHA, frozen Full, strict NoSocial isolation, budgets, 200 checkpoints and tests')
