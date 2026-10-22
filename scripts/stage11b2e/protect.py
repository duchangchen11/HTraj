from .common import *

def main():
    cpu_only();freeze='--freeze' in sys.argv;path=CONF/'protected_files.csv.gz'
    if freeze:
        with gzip.open(ROOT/'configs/stage11b2d/protected_files.csv.gz','rt') as f:
            records={r['path']:r['sha256'] for r in csv.DictReader(f)}
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
        names += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/stage11b2d/cache').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        for n in names:
            s=sha(ROOT/n);assert n not in records or records[n]==s,n;records[n]=s
        csvnew(path,[dict(path=k,sha256=v) for k,v in sorted(records.items())])
    with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    jnew(OUT/('history_before.json' if freeze else 'history_after.json'),dict(passed=not bad,files=len(rows),changed=bad,base=BASE))
    assert not bad,bad[:10]
    print('HISTORY SHA PASS',len(rows),flush=True)

if __name__=='__main__':main()
