from .common import *
import sys
def main():
    freeze='--freeze' in sys.argv;p=CONF/'protected_files.csv.gz'
    if freeze:
        with gzip.open(ROOT/'configs/stage11b2b/protected_files.csv.gz','rt') as f:rows={r['path']:r['sha256'] for r in csv.DictReader(f)}
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
        names+= [str(q.relative_to(ROOT)) for q in (ROOT/'experiments/stage11b2b/cache').rglob('*') if q.is_file() and '__pycache__' not in q.parts]
        for name in names:
            s=sha(ROOT/name)
            if name in rows:assert rows[name]==s,name
            rows[name]=s
        with gzip.open(p,'xt',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','sha256'],lineterminator='\n');w.writeheader()
            w.writerows(dict(path=k,sha256=v) for k,v in sorted(rows.items()))
    with gzip.open(p,'rt') as f:rows=list(csv.DictReader(f))
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    jnew(OUT/('history_before.json' if freeze else 'history_after.json'),dict(passed=not bad,files=len(rows),changed=bad,base=BASE))
    assert not bad,bad[:10]
    print('History protection PASS',len(rows),flush=True)
if __name__=='__main__':main()
