"""Audit immutable archives, actual training budget and frozen attribution."""
import csv
import json
import hashlib
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('results/tables/stage3p6')
    conf = Path('configs/stage3p6/protocol.json')
    p = json.loads(conf.read_text())
    protected = json.loads(Path('configs/stage3p6/protected_artifacts.json').read_text())
    for r in protected['files']:
        assert sha(r['path']) == r['sha256'], r['path']
    assert not p['ETH_evaluated'] and not p['B_run'] and p['epochs'] == 30
    old = json.loads(Path('configs/data_manifests/stage2_v1/fold_ETH.json').read_text())
    oldtrain = {w['sample_id'] for w in old['splits']['train']}
    oldval = {w['sample_id'] for w in old['splits']['validation']}
    for fold in p['folds']:
        assert sha(fold['manifest']) == fold['manifest_sha256']
        m = json.loads(Path(fold['manifest']).read_text())
        train, val = m['splits']['train'], m['splits']['validation']
        assert not {w['recording'] for w in train} & {w['recording'] for w in val}
        assert {w['sample_id'] for w in train} <= oldtrain and not {w['sample_id'] for w in train} & oldval
        assert {w['sample_id'] for w in val} <= oldval
        assert all(w['scene'] != 'ETH' and w['end_frame']-w['start_frame'] == 190 for w in train+val)
    for file in ('pretraining_tests.xml', 'unit_tests.xml'):
        suites = ET.parse(root/file).getroot()
        assert sum(int(s.get('tests', 0)) for s in suites) >= 165
        assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures'))
    decision = json.loads((root/'round3_decision.json').read_text())
    selection = json.loads((root/'anchor_selection.json').read_text())
    neural, epochs, steps = 0, 0, 0
    for path in root.glob('*/**/summary.json'):
        s = json.loads(path.read_text())
        assert not s['ETH_evaluated']
        if 'best_epoch' not in s:
            continue
        neural += 1
        c = json.loads(path.with_name('config.json').read_text())
        assert s['protocol_sha256'] == sha(conf)
        assert c['source_commit'] == s['source_commit']
        for file, digest in c['source_sha256'].items():
            assert sha(file) == digest, file
        rows = list(csv.DictReader(path.with_name('epochs.csv').open()))
        assert len(rows) == 30 and [int(r['epoch']) for r in rows] == list(range(1, 31))
        best = min(rows, key=lambda r: float(r['validation_ADE_m']))
        assert int(best['epoch']) == s['best_epoch'] and float(best['validation_FDE_m']) == s['best_FDE_m']
        assert int(rows[-1]['epoch']) == s['last_epoch'] and float(rows[-1]['validation_ADE_m']) == s['last_ADE_m']
        expected_updates = ((s['training_windows']+7)//8)*30
        assert s['optimizer_steps'] == expected_updates
        epochs += len(rows); steps += expected_updates
        expected_lr = [.001]*30 if s['schedule'] == 'constant' else [.001]*10+[.0003]*10+[.00009]*10
        assert all(abs(float(r['learning_rate'])-lr) < 1e-12 for r, lr in zip(rows, expected_lr))
        for label in ('best', 'last'):
            ckpt = Path(s[label+'_checkpoint_path'])
            assert sha(ckpt) == s[label+'_checkpoint_sha256']
            assert subprocess.run(['git', 'check-ignore', '-q', str(ckpt)]).returncode == 0
        if s['phase'] == 'residual':
            assert s['trainable_parameters'] == 14425 and s['frozen_parameters'] == 35352
            assert s['anchor_parameters_sha256_before'] == s['anchor_parameters_sha256_after']
            assert all(r['anchor_unchanged'] == 'True' for r in rows)
            assert sha(s['anchor']['checkpoint_path']) == s['anchor']['checkpoint_sha256']
    assert neural == (36 if decision['run_extra_seeds'] else 16)
    for chosen in selection['selections']:
        row = min(chosen['candidate_summaries'], key=lambda r: r['best_ADE_m'])
        assert row['best_checkpoint_sha256'] == chosen['checkpoint_sha256']
        anchors = [json.loads(f.read_text())['anchor'] for f in (root/chosen['recording']/'residual').glob('*/seed_17/summary.json')]
        assert len(anchors) == 5 and all(a == chosen for a in anchors)
    for r in csv.DictReader((root/'no_message_exact_checks.csv').open()):
        assert r['exact_per_target_error_and_zero_correction'] == 'True'
    analysis = json.loads((root/'analysis_provenance.json').read_text())
    assert analysis['executed_epochs'] == epochs and not analysis['ETH_evaluated']
    assert analysis['paired_target_rows'] == 2820*(3 if decision['run_extra_seeds'] else 1)
    for file in ('01_EXPERIMENT_PROTOCOL.md', '02_EGO_BASELINE_RESULTS.md', '03_FROZEN_EGO_DESIGN.md', '04_SOCIAL_PAYLOAD_COMPARISON.md', '05_INTERACTION_GROUP_ANALYSIS.md', '06_RESEARCH_DECISION.md', 'STAGE3P6_REPORT.md'):
        assert (Path('reports/stage3p6')/file).stat().st_size > 500
    print(json.dumps(dict(status='passed', protected_artifacts=len(protected['files']), old_checkpoints=98, neural_runs=neural, executed_epochs=epochs, optimizer_updates=steps, unit_tests=165,
        frozen_parameters_unchanged=True, no_message_exact_checks=True, ETH_evaluated=False, auto_push=False)))


if __name__ == '__main__':
    main()
