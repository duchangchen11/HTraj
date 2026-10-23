"""Parent whole-process accounting, including imports, save, and normal exit."""
from .common import *
def main():
    name=sys.argv[1];assert name in {'preflight','train'}
    prior=parent_seconds();start=time.perf_counter();env=os.environ.copy()
    env['HTRAJ_PROCESS_MONOTONIC_START']=str(start)
    with (OUT/f'{name}_stdout.txt').open('x') as f:
        try:
            r=subprocess.run([sys.executable,'-u','-m','scripts.stage11c1.'+name],cwd=ROOT,env=env,
                stdout=f,stderr=subprocess.STDOUT,timeout=max(1,18000-prior-60),check=False)
            code=r.returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.perf_counter()-start
    jnew(OUT/f'process_{name}.json',dict(worker=name,started_commit=commit(),returncode=code,
        wall_seconds=elapsed,prior_seconds=prior,cumulative_seconds=prior+elapsed,
        hard_seconds=18000,safety_seconds=16800,timeout_includes_exit_reserve=60,
        interpretation='Whole GPU-related subprocess wall; no concurrent GPU workers'))
    print(name,code,elapsed,prior+elapsed,flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':main()
