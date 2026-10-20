from .common import *
import os

def main():
    name=sys.argv[1];assert name in {'preflight','train'}
    prior=parent_seconds();start=time.perf_counter();source=commit()
    env=os.environ.copy();env['HTRAJ_PROCESS_MONOTONIC_START']=str(start)
    with (OUT/f'{name}_stdout.txt').open('x') as f:
        try:r=subprocess.run([sys.executable,'-u','-m','scripts.stage11b1c.'+name],cwd=ROOT,env=env,
            stdout=f,stderr=subprocess.STDOUT,timeout=max(1,28800-prior-120),check=False);code=r.returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.perf_counter()-start
    jnew(OUT/f'process_{name}.json',dict(worker=name,started_commit=source,returncode=code,wall_seconds=elapsed,
        prior_seconds=prior,cumulative_seconds=prior+elapsed,hard_seconds=28800,
        interpretation='Whole GPU worker process wall, including import/load/save; final CPU-only reporting excluded'))
    print(name,'return',code,'wall',round(elapsed,3),'cumulative',round(prior+elapsed,3),flush=True)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
