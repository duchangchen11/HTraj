"""Validate archived checksums, fixed finite runs and honest provenance."""
import csv
import json
import math
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from hashlib import sha256 as digest


def sha(path): return digest(Path(path).read_bytes()).hexdigest()
def js(path): return json.loads(Path(path).read_text())
def csv_rows(path): return list(csv.DictReader(Path(path).open()))


def main():
    root = Path('results/tables/stage3p7'); conf = Path('configs/stage3p7/protocol.json'); p = js(conf)
    protected = js('configs/stage3p7/protected_artifacts.json')
    for r in protected['files']: assert sha(r['path']) == r['sha256'], r['path']
    assert not p['ETH_evaluated'] and not p['B_run'] and not p['extra_seeds_automatically_run']
    assert p['epochs'] == 30 and p['seed'] == 17 and p['neural_models'] == ['M0', 'M2', 'M3']
    assert p['schedule'].startswith('constant')
    old = js('configs/stage3p6/protocol.json'); assert old['folds'] == p['folds'] and old['source_files'] == p['source_files']
    for fold in p['folds']:
        assert sha(fold['manifest']) == fold['manifest_sha256']
        m = js(fold['manifest']); train, val = m['splits']['train'], m['splits']['validation']
        assert not {w['recording'] for w in train} & {w['recording'] for w in val}
        assert all(w['scene'] != 'ETH' and w['end_frame']-w['start_frame'] == 190 for w in train+val)
    for info in p['source_files'].values():
        assert info['scene'] != 'ETH' and sha(Path(p['raw_root'])/info['file']) == info['sha256']
    totals = dict(neural_runs=0, epochs=0, optimizer_updates=0); times = 0.; initial_states = set(); sources = set()
    new_checkpoints = []
    for path in root.glob('*/M*/seed_17/summary.json'):
        s = js(path); c = js(path.with_name('config.json')); rows = csv_rows(path.with_name('epochs.csv'))
        assert s['status'] == 'completed' and s['parameter_count'] == 35352 and s['seed'] == 17 and not s['ETH_evaluated']
        assert c['protocol_sha256'] == s['protocol_sha256'] == sha(conf)
        assert c['source_commit'] == s['source_commit']
        subprocess.run(['git', 'merge-base', '--is-ancestor', s['source_commit'], 'HEAD'], check=True)
        sources.add(s['source_commit']); initial_states.add(s['initial_parameters_sha256'])
        for f, h in c['source_sha256'].items(): assert sha(f) == h, f
        assert [int(r['epoch']) for r in rows] == list(range(1, 31))
        assert all(float(r['learning_rate']) == .001 for r in rows)
        for r in rows:
            assert all(math.isfinite(float(r[k])) for k in ('train_online_ADE_loss_m', 'validation_ADE_m', 'validation_FDE_m'))
        best = min(rows, key=lambda r: float(r['validation_ADE_m']))
        assert s['best_epoch'] == int(best['epoch']) and s['best_FDE_m'] == float(best['validation_FDE_m'])
        assert s['last_ADE_m'] == float(rows[-1]['validation_ADE_m']) and s['last_epoch'] == 30
        assert s['optimizer_steps'] == math.ceil(s['training_windows']/8)*30
        for endpoint in ('best', 'last'):
            f = s[endpoint+'_checkpoint_path']; assert sha(f) == s[endpoint+'_checkpoint_sha256']
            assert subprocess.run(['git', 'check-ignore', '-q', f]).returncode == 0
            new_checkpoints.append(dict(path=f, sha256=sha(f)))
            groups = csv_rows(path.with_name(f'groups_{endpoint}_validation.csv'))
            assert len(groups) == 28 and groups[0]['group'] == 'all'
            assert all(int(r['ADE_count']) == int(r['FDE_count']) for r in groups)
        totals['neural_runs'] += 1; totals['epochs'] += len(rows); totals['optimizer_updates'] += s['optimizer_steps']; times += s['training_seconds']
    assert totals == dict(neural_runs=6, epochs=180, optimizer_updates=42930)
    assert len(sources) == len(initial_states) == 1 and len(new_checkpoints) == 12
    test_counts = {}
    for name in ('pretraining_tests_final.xml', 'unit_tests.xml'):
        suites = ET.parse(root/name).getroot()
        assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures'))
        test_counts[name] = sum(int(s.get('tests', 0)) for s in suites)
        assert test_counts[name] >= 183
    a = js(root/'analysis_provenance.json'); assert a['neural_runs'] == 6 and a['executed_epochs'] == 180 and not a['extra_training']
    assert a['pair_rows'] == len(csv_rows(root/'paired_target_errors.csv')) == 14100
    assert a['group_rows'] == len(csv_rows(root/'group_comparison.csv')) == 560
    assert len(csv_rows(root/'all_epoch_curves.csv')) == 180
    assert all(r['compatible_within_1e_minus6'] == 'True' for r in csv_rows(root/'M0_reproducibility.csv'))
    d = js(root/'research_decision.json'); assert not d['extra_seeds_run'] and not d['further_training_started']
    assert not list(Path('experiments/stage3p7').rglob('seed_42')) and not list(Path('experiments/stage3p7').rglob('seed_2026'))
    for name in ('01_MOTION_REPRESENTATION_AUDIT.md', '02_COORDINATE_NORMALIZATION.md', '03_BASELINE_RESULTS.md',
                 '04_GENERALIZATION_ANALYSIS.md', '05_RESEARCH_DECISION.md', 'STAGE3P7_REPORT.md'):
        assert (Path('reports/stage3p7')/name).stat().st_size > 500
    result = dict(status='passed', protected_artifacts=len(protected['files']), old_checkpoints=protected['old_checkpoints'],
        **totals, training_seconds=times, new_checkpoints_local_only=new_checkpoints, tests=test_counts,
        M1_new_training=False, M0_reproduced=True, ETH_evaluated=False, extra_seeds_run=False, auto_push=False)
    out = root/'DELIVERY_AUDIT.json'
    if out.exists():
        assert js(out) == result
    else:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'new_checkpoints_local_only'}))


if __name__ == '__main__': main()
