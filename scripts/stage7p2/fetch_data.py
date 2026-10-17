"""Acquire official converted ETH/UCY bytes at the pinned Git commit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import concurrent.futures
import hashlib
import json
import subprocess
import time
import requests
from src.socialvae_baseline.artifacts import json_new

COMMIT = 'e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5'
ROOT = Path('data/raw/agentformer_official')/COMMIT
OUT = Path('results/tables/stage7p2')


def acquire(entry):
    git_blob, path = entry
    url = f'https://raw.githubusercontent.com/Khrylx/AgentFormer/{COMMIT}/{path}'
    destination = ROOT/Path(path).relative_to('datasets/eth_ucy')
    attempts = []
    started = time.perf_counter()
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            content = response.content
            blob = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            assert blob == git_blob, 'Git blob mismatch'
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as f:
                f.write(content)
            return dict(path=str(destination), upstream_path=path, url=url, git_blob=git_blob,
                sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), git_blob_verified=True,
                attempts=attempts, seconds=time.perf_counter()-started, success=True)
        except Exception as e:
            attempts.append(dict(attempt=attempt+1, error=repr(e)))
    return dict(path=str(destination), upstream_path=path, url=url, git_blob=git_blob,
        attempts=attempts, success=False, seconds=time.perf_counter()-started)


def main():
    tree = subprocess.check_output(['git', '--git-dir=experiments/stage7/cache/AgentFormer.git',
        'ls-tree', '-r', COMMIT, 'datasets/eth_ucy'], text=True)
    entries = []
    for line in tree.splitlines():
        mode_type_sha, path = line.split('\t')
        entries.append((mode_type_sha.split()[-1], path))
    assert len(entries) == 74
    # One HTTP request for identical Git blobs; copy exact bytes to each fold.
    unique = {}
    for entry in entries:
        unique.setdefault(entry[0], entry)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(acquire, unique.values()))
    mapping = {r['git_blob']: r for r in downloaded}
    files = []
    for blob, path in entries:
        first = mapping[blob]
        target = ROOT/Path(path).relative_to('datasets/eth_ucy')
        r = dict(first, path=str(target), upstream_path=path,
            url=f'https://raw.githubusercontent.com/Khrylx/AgentFormer/{COMMIT}/{path}')
        if first['success'] and target != Path(first['path']):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(Path(first['path']).read_bytes())
            r['identical_blob_copied_from'] = first['path']
        files.append(r)
    json_new(OUT/'official_data_acquisition.json', dict(commit=COMMIT, expected_files=74,
        unique_blobs=len(unique), files=files, all_files_verified=all(r['success'] for r in files),
        provenance='Author-maintained converted files, not a substitute made from HTraj five recordings',
        datasets_redistributed_in_git=False))
    print('Official data:', sum(r['success'] for r in files), '/', len(files), 'unique blobs', len(unique), flush=True)


if __name__ == '__main__':
    main()
