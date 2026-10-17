"""Verify old artifacts and all acquired official bytes, read-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import subprocess
import time
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

OUT = Path('results/tables/stage7p2')


def main():
    begin = time.perf_counter()
    protected = json.load(open('configs/stage7p2/protected_artifacts.json'))['files']
    failures = [r['path'] for r in protected if not Path(r['path']).is_file() or sha256(r['path']) != r['sha256']]
    original = json.load(open('results/tables/stage7/sources/AgentFormer.json'))
    retry = json.load(open('results/tables/stage7/sources/AgentFormer_single_retry.json'))
    sources = {r['path']: r for r in original['files'] if 'error' not in r}
    sources.update({r['path']: r for r in retry['attempts']})
    for r in sources.values():
        blob = Path(r['local_path']).read_bytes()
        assert hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() == r['git_blob']
        assert hashlib.sha256(blob).hexdigest() == r['sha256']
    data = json.load(open(OUT/'official_data_acquisition.json'))
    for r in data['files']:
        blob = Path(r['path']).read_bytes()
        assert hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() == r['git_blob']
        assert hashlib.sha256(blob).hexdigest() == r['sha256']
        assert subprocess.run(['git','check-ignore','-q',r['path']]).returncode == 0
    weights = json.load(open(OUT/'official_checkpoint_acquisition.json'))
    for r in weights['files']:
        assert sha256(r['path']) == r['sha256']
        assert subprocess.run(['git','check-ignore','-q',r['path']]).returncode == 0
    checkpoints = []
    for p in (OUT/'smoke').glob('*.json'):
        for stage in json.load(open(p))['stages'].values():
            assert sha256(stage['checkpoint']) == stage['checkpoint_sha256']
            assert subprocess.run(['git','check-ignore','-q',stage['checkpoint']]).returncode == 0
            checkpoints.append(stage['checkpoint'])
    actual_source = {}
    for p in [*Path('scripts/stage7p2').glob('*.py'), *Path('src/agentformer_benchmark').glob('*.py'), Path('tests/test_agentformer_standard_benchmark.py')]:
        actual_source[str(p)] = sha256(p)
    json_new(OUT/'preservation_and_versions.json', dict(protected_files=len(protected),
        old_SHA_failures=failures, official_source_files_verified=len(sources),
        official_data_files_verified=len(data['files']), author_weight_files_verified=len(weights['files']),
        smoke_checkpoints_verified=len(checkpoints), source_SHA256_at_delivery=actual_source,
        no_data_or_weights_tracked=True, elapsed_seconds=time.perf_counter()-begin))
    assert not failures
    print('Preserved historical files:', len(protected), 'official source/data/weights:', len(sources), len(data['files']), len(weights['files']), flush=True)


if __name__ == '__main__':
    main()
