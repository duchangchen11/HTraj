"""No old metric interpretation; old predictions/checkpoints opaque SHA only."""
from common import *

def main():
    path=CONF/'protected_files.csv.gz'
    if '--freeze' in sys.argv:
        rows={r['path']:r['sha256'] for r in cread(ROOT/'configs/stage10a/protected_files.csv.gz')}
        for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'):
            if p:
                digest=sha(ROOT/p)
                if p in rows:assert rows[p]==digest,p
                rows[p]=digest
        for p in (ROOT/'experiments/stage10a/cache').rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                rows[str(p.relative_to(ROOT))]=sha(p)
        cnew(path,[dict(path=p,sha256=s) for p,s in sorted(rows.items())])
    rows=cread(path)
    failures=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    jnew(OUT/('history_before.json' if '--freeze' in sys.argv else 'history_after.json'),dict(passed=not failures,files=len(rows),failures=failures,old_test_metrics_interpreted=False,commit=commit()))
    assert not failures,failures[:10]
    print('Protected historic files',len(rows),flush=True)

if __name__=='__main__':main()
