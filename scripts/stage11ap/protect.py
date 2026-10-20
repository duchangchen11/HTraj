"""Opaque historical byte audit; never interpret old test predictions."""
from common import *
def main():
    index = CONF / 'protected_files.csv.gz'
    if '--freeze' in sys.argv:
        with gzip.open(ROOT/'configs/stage11a/protected_files.csv.gz', 'rt') as f:
            rows = {r['path']:r['sha256'] for r in csv.DictReader(f)}
        for p in subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0'):
            if p:
                s = sha(ROOT/p)
                if p in rows: assert s == rows[p], p
                rows[p] = s
        for p in (ROOT/'experiments/stage11a/cache').rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                rows[str(p.relative_to(ROOT))] = sha(p)
        cnew(index, [dict(path=p,sha256=s) for p,s in sorted(rows.items())])
    with gzip.open(index,'rt') as f: rows = list(csv.DictReader(f))
    bad = [r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path']) != r['sha256']]
    jnew(OUT/('history_before.json' if '--freeze' in sys.argv else 'history_after.json'),
         dict(passed=not bad, files=len(rows), failures=bad, base_commit=BASE, old_test_metrics_interpreted=False))
    assert not bad, bad[:5]
    print('Protected:', len(rows), 'historical files')
if __name__ == '__main__': main()
