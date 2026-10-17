"""Final source/protocol/evidence checks, never launch training or push."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

OUT = Path('results/tables/stage7p2')


def load(name):
    return json.load(open(OUT/name))


def main():
    protection = load('preservation_and_versions.json')
    assert protection['protected_files'] == 11182 and not protection['old_SHA_failures']
    assert protection['official_source_files_verified'] == 43
    assert protection['official_data_files_verified'] == 74 and protection['author_weight_files_verified'] == 14
    data = load('official_data_audit.json')
    assert data['all_expected_recordings_present'] and len(data['whole_loader_checks']) == 148
    assert len(data['official_files']) == 74 and len(data['splits']) == 5
    assert all(r['shared_frame_count'] == 0 for r in data['train_val_frame_overlap_checks'])
    assert all(r['shared_exact20_moving_position_fingerprints'] == 0 for r in load('cross_sequence_duplicate_audit.json')['checks'])
    weights = load('official_checkpoint_strict_loading.json')
    assert weights['all_folds_strict'] and not weights['used_strict_False']
    assert all(r['strict'] and r['both_parts_required'] for r in weights['cases'])
    download = load('official_checkpoint_acquisition.json')
    archive = next(r for r in download['download_steps'] if 'archive' in r)
    assert sha256(archive['archive']) == archive['sha256'] == weights['archive_sha256']
    evaluation = load('evaluation_equivalence.json')
    assert evaluation['official_eval_CLI_passed'] and evaluation['independent_ADE_FDE_winners_verified']
    assert evaluation['only_samples_evaluated'] and evaluation['GT_reconstruction_directory_excluded']
    assert all(r['returncode'] == 0 for r in load('original_CLI_admission.json')['cases'])
    smoke = load('smoke_summary.json')
    assert smoke['all_passed'] and len(smoke['cases']) == 6 and smoke['actual_optimizer_updates'] == 12
    assert smoke['complete_epochs'] == 0 and not smoke['ETH_test_scores_computed']
    assert not smoke['model_core_modified']
    for r in smoke['cases']:
        assert r['people'] in [1,2,57] and r['passed'] and not r['context_truncated']
        for name, s in r['stages'].items():
            assert s['optimizer_updates'] == 1 and s['complete_epochs'] == 0
            assert s['gradients_finite'] and s['gradient_L2'] > 0
            assert s['strict_model_optimizer_scheduler_restore'] and s['prediction_restore_max_difference'] == 0
            assert not s['resource']['OOM'] and s['model_context_people'] == r['people']
            assert sha256(s['checkpoint']) == s['checkpoint_sha256']
            if name == 'DLow':
                assert s['base_CVAE_parameters_unchanged']
    assert sha256('scripts/stage7p2/smoke.py') == smoke['source_SHA256']
    closure = load('execution_source_closure.json')
    assert closure['chronology_deviation'] and not closure['pre_smoke_complete_evidence_commit_claim']
    assert closure['no_restarts_or_extra_budget'] and closure['all_optimizer_updates'] == 12
    for path, value in closure['executed_source_files_equal_that_commit'].items():
        assert sha256(path) == value, path
    outcome = load('failures/commit_order_actual_outcome.json')
    assert outcome['all_six_smoke_cases_completed_before_full_evidence_commit'] and outcome['no_extra_optimizer_updates']
    gates = load('go_no_go.json')
    assert gates['decision'] == 'GO_STAGED_OFFICIAL_BASELINE_REPRODUCTION'
    assert not gates['full_fivefold_performance_reproduced'] and not gates['ETH_test_scores_computed']
    plan = json.load(open('configs/stage7p2/proposed_full_reproduction_plan.json'))
    assert plan['status'] == 'PROPOSED_NOT_EXECUTED' and plan['complete_all_five_before_paper_comparison']
    assert plan['CVAE_epochs'] == 100 and plan['DLow_epochs'] == 50
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(r.attrib.get(k,0)) for r in suites) for k in ['tests','failures','errors','skipped']}
    assert tests == dict(tests=407,failures=0,errors=0,skipped=0)
    reports = sorted(Path('reports/stage7p2').glob('*.md'))
    assert len(reports) == 7 and all(p.stat().st_size > 500 for p in reports)
    old = {r['path'] for r in json.load(open('configs/stage7p2/protected_artifacts.json'))['files']}
    changed = subprocess.check_output(['git','diff','81a2e239e8ee79c07d19e319401c0f96c0b5e7ec','--name-only'],text=True).splitlines()
    new = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    assert not old.intersection(changed)
    allowed = ['configs/stage7p2/','reports/stage7p2/','results/tables/stage7p2/','scripts/stage7p2/',
        'src/agentformer_benchmark/','tests/test_agentformer_standard_benchmark.py']
    for path in sorted(set(changed+new)):
        assert any(path.startswith(p) for p in allowed), path
        assert Path(path).stat().st_size < 10*1024**2, path
        if not path.endswith('.gz'):
            text = Path(path).read_text()
            assert not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',text), path
    subprocess.run(['git','diff','--check'],check=True)
    json_new(OUT/'delivery_verification.json',dict(historical_files_unchanged=11182,
        official_source_files=43,official_data_files=74,strict_CVAE_DLow_fold_pairs=5,
        real_smoke_cases=6,engineering_optimizer_updates=12,full_epochs=0,
        full_tests=tests,evaluation_equivalence=True,author_weight_archive_sha256=archive['sha256'],
        decision=gates['decision'],standard_fivefold_performance_reproduced=False,
        chronology_deviation_disclosed=True,complete_pre_smoke_evidence_commit_claim=False,
        ETH_test_metrics_computed=False,automatic_push=False,
        source_SHA256={str(p):sha256(p) for p in [*Path('scripts/stage7p2').glob('*.py'),*Path('src/agentformer_benchmark').glob('*.py'),Path('tests/test_agentformer_standard_benchmark.py')]},
        proposed_plan_SHA256=sha256('configs/stage7p2/proposed_full_reproduction_plan.json'),
        size_and_sensitive_content_checks=True,
        commit='Containing final delivery Git commit; tested code closure to 93e8a5a recorded separately'))
    print('Delivery verified:', tests, '; staged GO, no formal training/push',flush=True)


if __name__ == '__main__':
    main()
