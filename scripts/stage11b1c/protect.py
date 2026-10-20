from .common import *

def main():
    path=CONF/'protected_files.csv.gz';freeze='--freeze' in sys.argv
    if freeze:
        with gzip.open(ROOT/'configs/stage11b1r/protected_files.csv.gz','rt') as f:rows={r['path']:r['sha256'] for r in csv.DictReader(f)}
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
        for name in names:
            s=sha(ROOT/name)
            if name in rows:assert rows[name]==s,name
            rows[name]=s
        for p in (ROOT/'experiments/stage11b1r/cache').rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:rows[str(p.relative_to(ROOT))]=sha(p)
        cnew(path,[dict(path=k,sha256=v) for k,v in sorted(rows.items())])
    with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    jnew(OUT/('history_before.json' if freeze else 'history_after.json'),dict(passed=not bad,files=len(rows),failures=bad,base=BASE))
    assert not bad,bad[:5]
    print('History SHA PASS',len(rows),flush=True)

if __name__=='__main__':main()
