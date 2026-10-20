from .common import *
def main():
    p=CONF/'protected_files.csv.gz'
    if '--freeze' in sys.argv:
        with gzip.open(ROOT/'configs/stage11ap/protected_files.csv.gz','rt') as f: rows={r['path']:r['sha256'] for r in csv.DictReader(f)}
        for name in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'):
            if not name:continue
            s=sha(ROOT/name)
            if name in rows:assert rows[name]==s,name
            rows[name]=s
        for q in (ROOT/'experiments/stage11ap/cache').rglob('*'):
            if q.is_file() and '__pycache__' not in q.parts:rows[str(q.relative_to(ROOT))]=sha(q)
        cnew(p,[dict(path=k,sha256=v) for k,v in sorted(rows.items())])
    with gzip.open(p,'rt') as f:rows=list(csv.DictReader(f))
    failures=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    jnew(OUT/('history_before.json' if '--freeze' in sys.argv else 'history_after.json'),dict(passed=not failures,files=len(rows),failures=failures,base=BASE,old_test_scores_read=False))
    assert not failures,failures[:5]
    print('History SHA pass',len(rows),flush=True)
if __name__=='__main__':main()
