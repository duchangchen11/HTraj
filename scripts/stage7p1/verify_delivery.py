"""Verify the delivered evidence and its declared NO-GO, never train/push."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

OUT = Path('results/tables/stage7p1')


def load(path):
    return json.load(open(OUT/path))


def main():
    history = load('history_and_source_verification.json')
    assert history['protected_files'] == 11103 and not history['historical_SHA_failures']
    assert len(history['official_source_checks']) == 43
    assert all(r['SHA256_equal'] and r['git_blob_equal'] for r in history['official_source_checks'])
    setup = load('execution_setup.json')
    for path, value in setup['source_SHA256'].items():
        assert sha256(path) == value, path
    data = load('data_audit.json')
    for path, value in data['raw_data_sha256'].items():
        assert sha256(path) == value
        assert 'biwi_eth' not in path
    for s in data['summaries']:
        assert s['exact_ids_history_and_GT_SHA'] and s['valid_supports_unchanged']
        assert sha256(s['target_manifest']) == s['target_manifest_sha256']
        assert sha256(s['source_manifest']['path']) == s['source_manifest']['sha256']
    counterfactuals = list((OUT/'counterfactuals').glob('*.json'))
    assert len(counterfactuals) == 10
    for path in counterfactuals:
        r = json.load(open(path))
        assert r['passed'] and r['K1_prediction_max_difference'] == r['K20_prediction_max_difference'] == 0
        assert sha256(r['raw_path']) == r['raw_sha256'] and r['details']['changed_future_rows'] > 0
        assert subprocess.run(['git', 'check-ignore', '-q', r['raw_path']]).returncode == 0
    cases = [json.load(open(p)) for p in (OUT/'real_cases').glob('*.json')]
    assert len(cases) == 10 and all('unexpected_failure' not in r for r in cases)
    assert {r['people'] for r in cases} >= {1, 2, 3, 6, 9, 13, 15}
    assert sum(r['original_loss_status'] == 'PASS_ALL_CONTEXT_LABELS_COMPLETE' for r in cases) == 7
    assert sum(r['original_loss_status'] == 'BLOCKED_UNLABELED_CONTEXT' for r in cases) == 3
    for r in cases:
        assert r['optimizer_updates'] == 0 and r['strict_checkpoint_restore'] and r['restored_prediction_max_difference'] == 0
        assert sha256(r['checkpoint']) == r['checkpoint_sha256']
        assert subprocess.run(['git', 'check-ignore', '-q', r['checkpoint']]).returncode == 0
    for p in (OUT/'negative_probes').glob('*.json'):
        r = json.load(open(p))
        assert r['mask_only_solution_refuted'] and r['optimizer_updates'] == 0
        assert not r['completions'][2]['loss_finite']
    resource = load('maximum_context_resource.json')
    assert resource['people'] == 73 and resource['prediction_finite'] and not resource['resource']['OOM']
    assert resource['training_memory_feasibility'] == 'NOT_ESTABLISHED' and not resource['context_truncated']
    summary = load('execution_summary.json')
    decision = load('go_no_go.json')
    assert decision['decision'] == 'NO-GO' and not decision['strict_causal_original_training_admitted']
    assert summary['formal_epochs'] == summary['optimizer_updates'] == 0 and not summary['ETH_evaluated']
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(r.attrib.get(k, 0)) for r in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests == dict(tests=395, failures=0, errors=0, skipped=0)
    assert 'ModuleNotFoundError' in (OUT/'initial_test_launcher_failure.txt').read_text()
    reports = sorted(Path('reports/stage7p1').glob('*.md'))
    assert len(reports) == 7 and all(p.stat().st_size > 500 for p in reports)
    old = {r['path'] for r in json.load(open('configs/stage7p1/protected_artifacts.json'))['files']}
    changed = subprocess.check_output(['git', 'diff', '546c256755139edd59b1ffe09c184bc05260a1e8', '--name-only'], text=True).splitlines()
    assert not old.intersection(changed)
    # All additions must belong to this admission stage; never stage raw cache.
    allowed = ['configs/stage7p1/', 'reports/stage7p1/', 'results/tables/stage7p1/',
        'scripts/stage7p1/', 'src/agentformer_admission/', 'tests/test_agentformer_causal_admission.py']
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    for path in sorted(set(changed+untracked)):
        assert any(path.startswith(a) for a in allowed), path
        assert Path(path).stat().st_size < 10*1024**2, path
        if not path.endswith('.gz'):
            contents = Path(path).read_text()
            assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}', contents), path
    subprocess.run(['git', 'diff', '--check'], check=True)
    json_new(OUT/'delivery_verification.json', dict(historical_files_unchanged=history['protected_files'],
        original_official_source_blobs_verified=43, tested_source_commit=setup['code_commit'],
        tested_source_SHA256=setup['source_SHA256'], full_tests=tests,
        admission_decision='NO-GO', raw_data_counterfactuals_passed=10,
        real_cases=10, true_loss_backward_passes=7, mixed_label_original_training_blocked=3,
        checkpoints_strictly_restored=10, complete_context_max_N=73,
        new_admission_config_SHA256={str(p): sha256(p) for p in Path('configs/stage7p1').glob('*.json')},
        final_delivery_code_SHA256={str(p): sha256(p) for p in [*Path('scripts/stage7p1').glob('*.py'), *Path('src/agentformer_admission').glob('*.py'), Path('tests/test_agentformer_causal_admission.py')]},
        formal_epochs=0, optimizer_updates=0, ETH_evaluated=False, automatic_push=False,
        raw_data_and_checkpoint_git_ignored=True,
        size_and_sensitive_content_checks_passed=True,
        final_commit='Containing delivery commit; source execution commit and per-file SHA above avoid circular commit SHA'))
    print('Stage7.1 delivery verified:', tests, 'decision NO-GO; no training or push', flush=True)


if __name__ == '__main__':
    main()
