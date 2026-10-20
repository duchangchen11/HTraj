"""Measure whole GPU worker wall time, including Python/library startup."""
from .common import *
import sys

def main():
    name=sys.argv[1]
    assert name in {'diagnose_original','run_checks','distribution','fresh_process'}
    prior=sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('parent_process_*.json'))
    assert prior<3000,'No new GPU work after50min; reserve safe closing time'
    start=time.perf_counter()
    with (OUT/f'{name}_stdout.txt').open('x') as f:
        try:
            r=subprocess.run([sys.executable,'-m','scripts.stage11b1r.'+name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                             timeout=max(1,3300-prior),check=False)
            code=r.returncode
        except subprocess.TimeoutExpired:
            code=124
    elapsed=time.perf_counter()-start
    jnew(OUT/f'parent_process_{name}.json',dict(worker=name,returncode=code,wall_seconds=elapsed,
         cumulative_seconds=prior+elapsed,hard_budget_seconds=3600,stopped_before_seconds=3300,code_commit=commit()))
    print(name,'return',code,'wall',round(elapsed,3),'cumulative',round(prior+elapsed,3),flush=True)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
