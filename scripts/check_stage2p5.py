"""Verify scope, protected artifacts, complete epochs and actual provenance."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    protected = json.loads((ROOT/'configs/stage2p5/protected_artifacts.json').read_text())
    for path, digest in protected.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    protocol_path = ROOT/'configs/stage2p5/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    expected_protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    actual = ROOT/'results/tables/stage2p5'
    first = list(csv.DictReader((actual/'round1/summary.csv').open()))
    assert {row['model'] for row in first} == set(protocol['round1_models']) and len(first) == 10
    selection = json.loads((actual/'selection.json').read_text())
    runs = first[:]
    if selection['selected_social_models']:
        second = list(csv.DictReader((actual/'round2/summary.csv').open()))
        assert {row['model'] for row in second} == {'PedBase', *selection['selected_social_models']}
        assert len(second) == 3*(1+len(selection['selected_social_models']))
        assert len(selection['selected_social_models']) <= 2
        for model in {r['model'] for r in second}:
            assert {int(r['seed']) for r in second if r['model'] == model} == {17,42,2026}
        runs += second
    for run in runs:
        assert run['status'] == 'completed' and run['test_evaluated'] == 'False'
        assert run['protocol_sha256'] == expected_protocol_sha
        assert int(run['optimizer_steps']) == (960 if run['phase'] == 'round1' else 3840)
        epochs = list(csv.DictReader((actual/run['phase']/run['model']/f"seed_{run['seed']}"/'epochs.csv').open()))
        assert [int(r['epoch']) for r in epochs] == list(range(1,31))
        chosen = min(epochs, key=lambda r: float(r['validation_ADE_m']))
        assert int(chosen['epoch']) == int(run['best_epoch'])
        for row in epochs:
            for key, value in row.items():
                if key in ('source_commit', 'seed', 'epoch') or value == '':
                    continue
                assert math.isfinite(float(value)), (run['model'], key)
        local = ROOT/'experiments/stage2p5'/run['phase']/run['model']/f"seed_{run['seed']}"/'checkpoints'
        assert hashlib.sha256((local/'best.pt').read_bytes()).hexdigest() == run['best_checkpoint_sha256']
        assert hashlib.sha256((local/'last.pt').read_bytes()).hexdigest() == run['last_checkpoint_sha256']
        subprocess.run(['git','cat-file','-e',run['source_commit']+'^{commit}'], cwd=ROOT, check=True)
    mechanisms = list(csv.DictReader((actual/'mechanism_interventions.csv').open()))
    assert len(mechanisms) == 36
    assert all(r['status'] == 'executed' and r['test_evaluated'] == 'False' for r in mechanisms)
    receipt = json.loads((actual/'inference_provenance.json').read_text())
    for path_field, sha_field in [('inference_runner', 'inference_runner_sha256')]:
        assert hashlib.sha256((ROOT/receipt[path_field]).read_bytes()).hexdigest() == receipt[sha_field]
    assert hashlib.sha256((ROOT/'src/training/social_diagnostics.py').read_bytes()).hexdigest() == receipt['inference_engine_sha256']
    assert hashlib.sha256((actual/'mechanism_interventions.csv').read_bytes()).hexdigest() == receipt['result_sha256']
    value_controls = list(csv.DictReader((actual/'value_vs_support_controls.csv').open()))
    assert len(value_controls) == 8
    final_controls = list(csv.DictReader((actual/'final_candidate_interventions.csv').open()))
    assert len(final_controls) == 9*3*len(selection['selected_social_models'])
    cases = {row['case'] for row in mechanisms}
    for model in selection['selected_social_models']:
        for seed in [17, 42, 2026]:
            rows = [row for row in final_controls if row['model']==model and int(row['seed'])==seed]
            assert {row['case'] for row in rows} == cases and len(rows) == 9
    for table, runner in [(value_controls, 'scripts/stage2p5_value_controls.py'),
                           (final_controls, 'scripts/stage2p5_final_attributions.py')]:
        runner_sha = hashlib.sha256((ROOT/runner).read_bytes()).hexdigest()
        for row in table:
            assert row['status'] == 'executed' and row['test_evaluated'] == 'False'
            assert row['inference_runner_sha256'] == runner_sha
            assert all(math.isfinite(float(row[key])) for key in ('ADE_m', 'FDE_m'))
    groups = list(csv.DictReader((actual/'group_comparison.csv').open()))
    assert len(groups) == len(runs)*9*2
    aggregates = list(csv.DictReader((actual/'round2_mean_std.csv').open()))
    assert {row['model'] for row in aggregates} == {'PedBase', *selection['selected_social_models']}
    assert all(int(row['n']) == 3 for row in aggregates)
    suites = ET.parse(actual/'unit_tests.xml').getroot()
    assert sum(int(suite.attrib.get('tests', 0)) for suite in suites) >= 86
    assert all(int(suite.attrib.get(key, 0)) == 0 for suite in suites for key in ('failures', 'errors'))
    for filename in ['01_PROBLEM_DIAGNOSIS.md','02_FUSION_DESIGN.md','03_ABLATION_RESULTS.md',
                     '04_SOCIAL_SCENE_ANALYSIS.md','05_TRAINING_COMPARISON.md','STAGE2P5_REPORT.md']:
        assert (ROOT/'reports/stage2p5'/filename).is_file(), filename
    tracked = subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines()
    assert not any(path.endswith(('.pt','.pth','.ckpt','.npz')) for path in tracked)
    print(f'PASS: {len(runs)} complete runs, 30 epochs each, {len(mechanisms)+len(value_controls)+len(final_controls)} executed control rows (noops labelled); Stage2 bytes preserved; no ETH test/weights tracked.')


if __name__ == '__main__':
    main()
