"""List and inspect only the new lightweight Stage6 delivery artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gzip
import re
import subprocess
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new


def main():
    folders = ('configs/stage6', 'scripts/stage6', 'reports/stage6', 'results/tables/stage6')
    candidates = [p for folder in folders for p in Path(folder).rglob('*') if p.is_file()]
    candidates += list(Path('experiments/stage6').glob('*.txt'))
    candidates += [Path('experiments/stage6/.gitignore'), Path('src/evaluation/stage6_motion.py'),
                   Path('tests/test_stage6_motion.py')]
    output = Path('results/tables/stage6/delivery_inventory.json')
    candidates = sorted(set(candidates) - {output})
    paths = [p for p in candidates if subprocess.run(
        ['git', 'check-ignore', '-q', str(p)], check=False).returncode != 0]
    old = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', '6d6c741520ce09a5a72e375ffa7a17d4206b7d97'],
        text=True).splitlines())
    assert not old & {str(p) for p in paths}
    markers = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|'
                         rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                         rb'AKIA[0-9A-Z]{16}')
    records = []
    for path in paths:
        assert path.stat().st_size < 5_000_000, path
        assert path.suffix not in {'.pth', '.pt', '.ckpt', '.npy', '.npz', '.pkl', '.pickle'}, path
        assert not any(s in {'cache', 'venv', 'checkpoints', 'raw'} for s in path.parts), path
        content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        assert not markers.search(content), f'Potential credential in {path}'
        records.append(dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size))
    json_new(output, dict(files=records, file_count_excluding_this_inventory=len(records),
        total_bytes_excluding_this_inventory=sum(r['bytes'] for r in records),
        maximum_bytes=max(r['bytes'] for r in records),
        all_files_new_since_stage5p5=True, old_files_included=False,
        raw_coordinates_weights_external_source_environments_ignored=True,
        high_confidence_credential_marker_matches=0,
        safety_scope='Filename/size/known credential-marker check; not a universal secret detector.',
        inventory_self_sha='Not included; use the containing delivery Git commit.',
        derivative_results_only=True))
    print('Lightweight artifacts:', len(records), 'total bytes:', sum(r['bytes'] for r in records))


if __name__ == '__main__':
    main()
