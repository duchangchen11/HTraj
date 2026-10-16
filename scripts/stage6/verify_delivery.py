"""Read-only integrity checks for the Stage6 non-training delivery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import csv
import gzip
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

ROOT = Path('results/tables/stage6')
BASE = '6d6c741520ce09a5a72e375ffa7a17d4206b7d97'
BRANCH = 'feat/stage6-research-reassessment'


def load(path):
    return json.loads(Path(path).read_text())


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def ignored(path):
    return subprocess.run(['git', 'check-ignore', '-q', str(path)]).returncode == 0


def main():
    started = time.perf_counter()
    assert command('git', 'rev-parse', 'HEAD') == BASE
    assert command('git', 'branch', '--show-current') == BRANCH
    protection = load('configs/stage6/protected_artifacts.json')
    assert protection['base_commit'] == BASE
    failures = [r['path'] for r in protection['files']
                if not Path(r['path']).is_file() or sha256(r['path']) != r['sha256']]
    assert not failures, failures
    print('Historical SHA256 unchanged:', len(protection['files']), flush=True)
    plan = load('configs/stage6/diagnostic_protocol.json')
    assert not plan['training'] and not plan['ETH_evaluated']
    data = {}
    manifests = {}
    for recording, entry in plan['manifests'].items():
        assert sha256(entry['path']) == entry['sha256']
        manifest = load(entry['path'])
        assert not manifest['ETH_evaluated']
        assert not manifest['original_validation_promoted_to_train']
        assert manifest['validation_recording'] == recording
        assert sha256(manifest['origin_manifest']) == manifest['origin_manifest_sha256']
        for name, source in manifest['source_files'].items():
            assert 'eth' not in name.lower()
            path = Path(plan['data_root']) / source['file']
            assert sha256(path) == source['sha256']
            data[str(path)] = source['sha256']
        train = {x['recording'] for x in manifest['splits']['train']}
        validation = {x['recording'] for x in manifest['splits']['validation']}
        assert recording not in train and validation == {recording}
        assert not train & validation and all('eth' not in x.lower() for x in train)
        manifests[entry['path']] = entry['sha256']
    execution = load(ROOT / 'diagnosis/execution.json')
    assert execution['code_parent_commit'] == BASE
    assert execution['new_training_runs'] == execution['optimizer_updates'] == 0
    assert execution['ETH_evaluated'] is False
    assert execution['restored_combinations'] == 12
    assert execution['maximum_error_match_difference_m'] <= plan['inference_match_tolerance_m']
    for key in ('source_SHA', 'input_SHA'):
        for path, expected in execution[key].items():
            assert sha256(path) == expected, path
    for run in execution['inference_runs']:
        assert run['strict_load'] and run['optimizer_updates'] == 0
        assert sha256(run['checkpoint']) == run['checkpoint_sha256']
        assert ignored(run['checkpoint'])
    assert sha256(execution['trajectory_packet_path']) == execution['trajectory_packet_sha256']
    assert ignored(execution['trajectory_packet_path'])

    public = {}
    for name in ('SGCN', 'SingularTrajectory'):
        metadata = load(ROOT / f'public/{name}_git_fallback.json')
        assert metadata['success'] and metadata['ETH_scores_read'] is False
        for row in metadata['files']:
            assert row['status'] == 200 and row['git_blob_verified']
            content = Path(row['local_path']).read_bytes()
            assert len(content) == row['bytes']
            assert hashlib.sha256(content).hexdigest() == row['sha256']
            blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            assert blob == row['git_blob']
            assert ignored(row['local_path'])
        public[name] = dict(commit=metadata['commit'], verified_files=len(metadata['files']))
    sgcn = load(ROOT / 'public/SGCN_functional_execution.json')
    supplement = load(ROOT / 'public/structural_supplement.json')
    for obj in (sgcn, supplement):
        assert obj['optimizer_updates'] == obj['new_training_runs'] == 0
        assert obj['ETH_evaluated'] is False
    assert sgcn['real_scene_evaluation'] is False and sgcn['official_source_modified'] is False
    assert len(sgcn['cases']) == 12
    assert sum(not case['forward_success'] for case in sgcn['cases']) == 2
    assert all(case['people'] == 1 for case in sgcn['cases'] if not case['forward_success'])

    shards = load(ROOT / 'diagnosis/step_shard_index.json')
    assert sha256(shards['original_stream']) == shards['original_sha256']
    assert ignored(shards['original_stream'])
    total = 0
    # Compare every retained derived row, not just file sizes or totals.
    counts = {}
    for row in shards['shards']:
        assert sha256(row['path']) == row['sha256']
        with gzip.open(row['path'], 'rt', newline='') as stream:
            lines = list(csv.DictReader(stream))
        assert len(lines) == row['rows']
        counts[row['path']] = lines
        total += len(lines)
    assert len(shards['shards']) == 18 and total == shards['total_rows'] == 152280
    originals = {}
    with gzip.open(shards['original_stream'], 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['recording'], row['seed'], row['model'])
            originals.setdefault(key, []).append(row)
    for rows in counts.values():
        first = rows[0]
        assert rows == originals.pop((first['recording'], first['seed'], first['model']))
    assert not originals

    root = ET.parse(ROOT / 'full_tests.xml').getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
             for k in ('tests', 'errors', 'failures', 'skipped')}
    assert tests == dict(tests=360, errors=0, failures=0, skipped=0)
    reports = [
        '01_STAGE5P5_FAILURE_ANALYSIS.md', '02_PUBLIC_GNN_BASELINE_REVIEW.md',
        '03_RESEARCH_PROBLEM_COMPARISON.md', '04_CHAPTER4_METHOD_DIRECTION.md',
        'STAGE6_REPORT.md',
    ]
    for name in reports:
        assert (Path('reports/stage6') / name).stat().st_size > 300
    direction = load(ROOT / 'research_direction.json')
    assert direction['primary']['requires_separate_authorization']
    assert not direction['backup']['training_authorized']
    assert direction['current_stage_new_training_runs'] == 0 and not direction['ETH_evaluated']
    changed = command('git', 'diff', 'HEAD', '--name-only').splitlines()
    assert not {r['path'] for r in protection['files']} & set(changed)
    subprocess.run(['git', 'diff', '--check'], check=True)
    source = sorted(Path('scripts/stage6').glob('*.py')) + [
        Path('src/evaluation/stage6_motion.py'), Path('tests/test_stage6_motion.py')]
    result = dict(
        base_commit=BASE, branch=BRANCH, stage6_source_sha256={str(p): sha256(p) for p in source},
        historical_files_checked=len(protection['files']), historical_SHA_failures=failures,
        historical_tracked_paths_changed=False, manifests_sha256=manifests,
        non_ETH_raw_data_sha256=data, frozen_best_checkpoints_verified=12,
        inference_maximum_error_match_difference_m=execution['maximum_error_match_difference_m'],
        public_source_checks=public, expected_SGCN_single_person_interface_failures=2,
        public_failures_reported=True, per_step_rows_preserved=total,
        per_step_values_equal_to_original_stream=True, full_tests=tests,
        reports=[str(Path('reports/stage6') / n) for n in reports],
        new_training_runs=0, optimizer_updates=0, ETH_evaluated=False,
        next_stage_training_requires_new_authorization=True, automatic_push=False,
        elapsed_seconds=time.perf_counter() - started,
        commit_provenance='Base SHA plus source SHA above; delivery commit is the Git commit containing this file (no circular self-SHA).',
    )
    json_new(ROOT / 'delivery_verification.json', result)
    print('Delivery verification passed:', tests, 'step rows', total, flush=True)


if __name__ == '__main__':
    main()
