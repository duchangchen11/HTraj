"""CPU-only, exclusive Stage11C-1 historical SHA protection.

The inherited list is immutable evidence. No old file is rewritten, including
failed experiments. Hashing sealed artifacts is an integrity check, not loading
or evaluating the sealed data.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = '29fe2036c034cd0701e672be69f448b401ee76dd'
CONF = ROOT / 'configs/stage11c1r'
OUT = ROOT / 'results/tables/stage11c1r'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def jnew(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Must run CPU-only'
    started = time.perf_counter()
    path = CONF / 'protected_files.csv.gz'
    if not args.after:
        inherited = ROOT / 'configs/stage11c1/protected_files.csv.gz'
        with gzip.open(inherited, 'rt') as stream:
            records = {row['path']: row['sha256'] for row in csv.DictReader(stream)}
        inherited_count = len(records)
        names = subprocess.check_output(
            ['git', 'ls-tree', '-r', '--name-only', BASE], cwd=ROOT, text=True
        ).splitlines()
        names += [str(p.relative_to(ROOT)) for p in
                  (ROOT / 'experiments/stage11c1/cache').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
        for name in names:
            actual = sha(ROOT / name)
            assert name not in records or records[name] == actual, name
            records[name] = actual
        CONF.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'xt', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'sha256'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(dict(path=k, sha256=v) for k, v in sorted(records.items()))
    else:
        inherited_count = None
    with gzip.open(path, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    bad = [row['path'] for row in rows
           if not (ROOT / row['path']).is_file() or sha(ROOT / row['path']) != row['sha256']]
    result = dict(passed=not bad, files=len(rows), changed=bad, base=BASE,
                  inherited_files=inherited_count, protection_manifest_SHA=sha(path),
                  scope='Stage1 through Stage11C1; inherited list + base tracked files + C1 cache',
                  sealed_artifact_bytes_hashed_only=True, test_loaded=False,
                  wall_seconds=time.perf_counter() - started)
    jnew(OUT / ('history_after.json' if args.after else 'history_before.json'), result)
    print(json.dumps(result), flush=True)
    assert not bad, bad[:10]


if __name__ == '__main__':
    main()
