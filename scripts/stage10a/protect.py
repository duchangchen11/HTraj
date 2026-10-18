"""Opaque historical bytes, including ignored checkpoints and test predictions.

Never interpret any historic test trajectories/metrics. Reuses checksum metadata
only; incompatible expected hashes fail rather than silently replacing them.
"""
from common import *


def main():
    index = CONF / 'protected_files.csv.gz'
    if '--freeze' in sys.argv:
        expected = {}
        def add(path, digest):
            if path in expected and expected[path] != digest:
                raise ValueError('Conflicting historic SHA ' + path)
            expected[path] = digest
        for r in read_csv(ROOT / 'configs/stage9/protected_files.csv.gz'):
            add(r['path'], r['sha256'])
        probe = json.loads((ROOT / 'results/tables/stage8/resonance_probe.json').read_text())
        add(probe['checkpoint_restore']['path'], probe['checkpoint_restore']['sha256'])
        for item in json.loads((ROOT / 'configs/stage8/protocol.json').read_text())['prediction_SHA_indices']:
            add(item['path'], item['sha256'])
            for r in read_csv(ROOT / item['path']):
                add(r['path'], r.get('sha256', r.get('SHA256')))
        for r in read_csv(ROOT / 'results/tables/stage9/all_checkpoint_SHA256.csv'):
            add(r['path'], r['sha256'])
        e = json.loads((ROOT / 'results/tables/stage9/validation_execution.json').read_text())
        for r in e['models']:
            add(r['path'], r['sha256'])
        for split, r in json.loads((ROOT / 'results/tables/stage9/data_manifest.json').read_text())['splits'].items():
            add('experiments/stage9/cache/' + split + '_data.pt', r['cache_SHA'])
        for p in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'):
            if p:
                add(p, sha(ROOT / p))
        write_csv(index, [dict(path=p, sha256=s) for p, s in sorted(expected.items())])
    rows = read_csv(index)
    failures = [r['path'] for r in rows if not (ROOT / r['path']).is_file() or sha(ROOT / r['path']) != r['sha256']]
    write_json(OUT / ('history_before.json' if '--freeze' in sys.argv else 'history_after.json'),
               dict(checked=len(rows), failures=failures, passed=not failures,
                    test_bytes_opaque=True, prediction_errors_parsed=False, execution_commit=commit()))
    print('Historical SHA checked:', len(rows), 'failures:', len(failures), flush=True)
    assert not failures, failures[:10]


if __name__ == '__main__':
    main()
