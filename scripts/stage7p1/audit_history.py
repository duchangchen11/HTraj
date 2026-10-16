"""Read-only historical/source/checkpoint identity verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import time
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new


def main():
    start = time.perf_counter()
    p = json.load(open('configs/stage7p1/protected_artifacts.json'))
    failures = []
    for r in p['files']:
        if not Path(r['path']).is_file() or sha256(r['path']) != r['sha256']:
            failures.append(r['path'])
    original = json.load(open('results/tables/stage7/sources/AgentFormer.json'))
    retry = json.load(open('results/tables/stage7/sources/AgentFormer_single_retry.json'))
    sources = {r['path']: r for r in original['files'] if 'error' not in r}
    sources.update({r['path']: r for r in retry['attempts']})
    source_checks = []
    for r in sources.values():
        blob = Path(r['local_path']).read_bytes()
        sha_ok = hashlib.sha256(blob).hexdigest() == r['sha256']
        git_ok = hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() == r['git_blob']
        source_checks.append(dict(path=r['local_path'], sha256=r['sha256'], git_blob=r['git_blob'],
            SHA256_equal=sha_ok, git_blob_equal=git_ok))
    assert len(source_checks) == len(original['files']) == 43
    checkpoints = []
    for path in Path('results/tables/stage7p1/real_cases').glob('*.json'):
        r = json.load(open(path))
        checkpoints.append(dict(path=r['checkpoint'], sha256=r['checkpoint_sha256'],
            SHA256_equal=sha256(r['checkpoint']) == r['checkpoint_sha256']))
    result = dict(protected_files=len(p['files']), historical_SHA_failures=failures,
        official_commit=original['commit'], official_source_checks=source_checks,
        new_random_checkpoints=checkpoints, automatic_push=False, ETH_evaluated=False,
        elapsed_seconds=time.perf_counter()-start)
    json_new('results/tables/stage7p1/history_and_source_verification.json', result)
    assert not failures and all(r['git_blob_equal'] and r['SHA256_equal'] for r in source_checks)
    assert len(checkpoints) == 10 and all(r['SHA256_equal'] for r in checkpoints)
    print('Historical files unchanged:', len(p['files']), '; official blobs:', len(source_checks), '; new checkpoint hashes:', len(checkpoints), flush=True)


if __name__ == '__main__':
    main()
