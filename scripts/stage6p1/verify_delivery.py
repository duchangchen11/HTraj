"""Audit frozen history, training budgets, paired targets, weights and test results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,time,hashlib,xml.etree.ElementTree as ET
import numpy as np
from src.data.preprocessing import sha256
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

ROOT=Path('results/tables/stage6p1')


def main():
    started=time.perf_counter();plan=json.load(open('configs/stage6p1/protocol.json'))
    protection=json.load(open('configs/stage6p1/protected_artifacts.json'))
    failures=[r['path'] for r in protection['files'] if not Path(r['path']).is_file() or sha256(r['path'])!=r['sha256']]
    assert not failures,failures
    print('Historical SHA unchanged:',len(protection['files']),flush=True)
    initialization=json.load(open(ROOT/'initialization_check.json'));runs=[];checkpoints=[];data={}
    for row in plan['formal_runs']:
        rec=row['recording'];condition=row['condition'];root=ROOT/'runs'/rec/condition
        provenance=json.load(open(root/'provenance.json'));selected=json.load(open(root/'selection.json'))
        assert selected['training_completed'] and selected['completed_epochs']==30
        assert selected['initial_parameter_state']['parameter_state_sha256']==initialization['parameter_state_sha256']
        assert provenance['protocol_sha256']==sha256('configs/stage6p1/protocol.json')
        assert provenance['fresh_random_initialization'] and not provenance['checkpoint_loaded_before_training']
        commit=provenance['code_commit']
        assert commit=='74e441a065d96979c1cdc60497ad1f4178b1bcd6'
        for file,expected in provenance['source_hashes'].items():
            assert sha256(file)==expected
            frozen=subprocess.check_output(['git','show',f'{commit}:{file}'])
            assert hashlib.sha256(frozen).hexdigest()==expected
        assert sha256(plan['manifests'][rec]['path'])==plan['manifests'][rec]['sha256']
        manifest=json.load(open(plan['manifests'][rec]['path']))
        assert not manifest['ETH_evaluated'] and not manifest['original_validation_promoted_to_train']
        for name,source in manifest['source_files'].items():
            assert name!='biwi_eth'
            p=Path(plan['data_root'])/source['file'];assert sha256(p)==source['sha256'];data[str(p)]=source['sha256']
        assert rec not in provenance['train_recordings'] and provenance['validation_recordings']==[rec]
        n=provenance['training_targets'];steps=int(np.ceil(n/128))
        assert selected['total_target_visits']==30*n and selected['total_optimizer_steps']==30*steps
        curves=read_csv(root/'training_curve.csv');assert [int(c['epoch']) for c in curves]==list(range(1,31))
        for c in curves:
            assert int(c['targets_seen'])==n and int(c['steps'])==steps
            assert c['parameters_finite']=='True'
            assert all(np.isfinite(float(c[k])) for k in ['train_loss','train_rec','train_kl','gradient_L2_max','K1_prior_ADE_m','K1_prior_FDE_m','minADE20_m','minFDE20_m'])
        expected=min(curves,key=lambda c:(float(c['K1_prior_ADE_m']),int(c['epoch'])))
        assert selected['best']['epoch']==int(expected['epoch'])
        assert selected['last']['epoch']==30
        assert abs(selected['best']['K1_prior_ADE_m']-float(expected['K1_prior_ADE_m']))<1e-12
        assert abs(selected['best']['K1_prior_FDE_m']-float(expected['K1_prior_FDE_m']))<1e-12
        smoke=json.load(open(root/'smoke_5epoch.json'));assert smoke['passed'] and smoke['extra_epochs']==0
        restoration=json.load(open(root/'checkpoint_restore_checks.json'))
        assert [r['epoch'] for r in restoration]==[1,5,10,20,30]
        assert all(r['strict_model_restore'] and r['optimizer_state_identical'] and r['rng_resume_verified'] and r['supervised_stochastic_loss_max_difference']==0 for r in restoration)
        for cp in read_csv(root/'checkpoint_inventory.csv'):
            assert sha256(cp['path'])==cp['sha256']
            assert subprocess.run(['git','check-ignore','-q',cp['path']]).returncode==0
            checkpoints.append(cp)
        assert not (root/'failure.json').exists()
        runs.append(dict(recording=rec,condition=condition,epochs=30,target_visits=30*n,optimizer_steps=30*steps,
            code_commit=commit,seed=17,best_epoch=selected['best']['epoch'],last_epoch=30,
            source_sha_verified_against_training_commit=True,checkpoint_restore_checks=5))
    for rec in plan['manifests']:
        a,b=[r for r in runs if r['recording']==rec]
        assert a['target_visits']==b['target_visits'] and a['optimizer_steps']==b['optimizer_steps']
    execution=json.load(open(ROOT/'analysis/execution.json'))
    assert not execution['ETH_evaluated'] and execution['optimizer_updates']==0 and not execution['selection_changed']
    assert sha256(execution['trajectory_packet'])==execution['trajectory_packet_sha256']
    assert subprocess.run(['git','check-ignore','-q',execution['trajectory_packet']]).returncode==0
    assert execution['maximum_diagnostic_error_match_difference_m']<2e-6
    assert len(json.load(open(ROOT/'analysis/budget_and_pairing_checks.json')))==4
    result=json.load(open(ROOT/'analysis/research_decision.json'))
    assert not result['extra_training_authorized']
    assert result['all_passed']==all(c['passed'] for c in result['checks'])
    assert result['checks'][-1]==dict(gate='integrity',passed=True)
    suites=ET.parse(ROOT/'full_tests.xml').getroot().findall('testsuite')
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert tests['tests']>=375 and tests['errors']==tests['failures']==tests['skipped']==0
    scheduler=json.load(open(ROOT/'scheduler_completion.json'))
    assert scheduler['launched_runs']==4 and scheduler['all_succeeded'] and scheduler['extra_runs']==0
    historical=set(r['path'] for r in protection['files'])
    changed=set(subprocess.check_output(['git','diff',plan['base_commit'],'--name-only'],text=True).splitlines())
    assert not historical&changed
    report_names=['01_FROZEN_PROTOCOL.md','02_SAMPLING_DISTRIBUTION.md','03_TRAINING_RESULTS.md',
        '04_MOTION_ERROR_ANALYSIS.md','05_PAIRED_COMPARISON.md','06_RESEARCH_DECISION.md','STAGE6P1_REPORT.md']
    assert all((Path('reports/stage6p1')/name).stat().st_size>300 for name in report_names)
    subprocess.run(['git','diff','--check'],check=True)
    json_new(ROOT/'delivery_verification.json',dict(historical_files_checked=len(protection['files']),historical_SHA_failures=failures,
        prior_source_protocol_results_checkpoints_unchanged=True,runs=runs,new_formal_runs=4,epochs_each=30,
        additional_seed_or_epoch_runs=0,verified_checkpoint_files=len(checkpoints),full_tests=tests,
        raw_non_ETH_SHA=data,configuration_sha256=sha256('configs/stage6p1/protocol.json'),
        manifest_SHA=plan['manifests'],all_pairing_and_budget_checks_passed=True,
        current_hypothesis_continue_gate=result['all_passed'],ETH_evaluated=False,automatic_push=False,
        elapsed_seconds=time.perf_counter()-started,
        final_commit='Containing Git delivery commit; training commit is separately recorded above. No circular self-SHA.'))
    print('Delivery verification passed; tests',tests,'checkpoints',len(checkpoints),flush=True)


if __name__=='__main__':main()
