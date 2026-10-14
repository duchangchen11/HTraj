"""Read-only final checks of real split/result artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ['01_DATA_PROTOCOL.md', '02_INTERACTION_COVERAGE.md', 'INTERACTION_COVERAGE.md',
               '03_MODEL_IMPLEMENTATION.md', '04_UNIT_TEST_RESULTS.md', '05_SMOKE_TRAINING.md', 'STAGE2_REPORT.md']
    for name in reports:
        assert (ROOT/'reports/stage2'/name).is_file(), name
    manifests = ROOT/'configs/data_manifests/stage2_v1'
    for path in manifests.glob('fold_*.json'):
        content = json.loads(path.read_text())
        splits = content['splits']
        for split in ('train', 'validation'):
            assert all(w['scene'] != content['held_out_scene'] for w in splits[split])
        assert all(w['scene'] == content['held_out_scene'] for w in splits['test'])
        for recording in content['source_files']:
            train = [w for w in splits['train'] if w['recording'] == recording]
            val = [w for w in splits['validation'] if w['recording'] == recording]
            if train and val:
                assert max(w['end_frame'] for w in train) < min(w['start_frame'] for w in val)
        for item in content['source_files'].values():
            raw = ROOT/'data/raw/eth_ucy'/item['file']
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == item['sha256']
    result_root = ROOT/'results/tables/stage2_smoke_v1'
    rows = list(csv.DictReader((result_root/'summary.csv').open()))
    assert {r['model'] for r in rows} == {'PedBase', 'FirstOrder-GNN', 'Dynamic-GNN', 'TOHI-Net-A'}
    assert len({r['data_manifest_sha256'] for r in rows}) == 1
    assert len({r['subset_manifest_sha256'] for r in rows}) == 1
    assert len({r['config_sha256'] for r in rows}) == 1
    assert len({r['source_commit'] for r in rows}) == 1
    assert len({r['optimizer_steps'] for r in rows}) == 1
    for row in rows:
        assert row['test_evaluated'] == 'False' and row['seed'] == '17'
        assert row['status'] == 'completed' and row['epochs'] == '5'
        assert float(row['final_train_loss']) < float(row['initial_train_loss'])
        for field in ['validation_ADE_m', 'validation_FDE_m', 'actual_training_seconds', 'cuda_peak_allocated_MiB']:
            assert math.isfinite(float(row[field])) and float(row[field]) > 0
        config_sha = hashlib.sha256((ROOT/'configs/stage2_smoke.json').read_bytes()).hexdigest()
        assert config_sha == row['config_sha256']
        subprocess.run(['git', 'cat-file', '-e', row['source_commit']+'^{commit}'], cwd=ROOT, check=True)
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    assert not any(x.startswith(('data/raw/eth_ucy/', 'data/processed/')) and not x.endswith('.gitkeep') for x in tracked)
    assert not any(x.endswith(('.pt', '.pth', '.ckpt', '.npz')) for x in tracked)
    print('PASS: five actual manifests isolated; raw hashes valid; four finite equal-budget smoke runs; no tracked data/weights.')


if __name__ == '__main__':
    main()
