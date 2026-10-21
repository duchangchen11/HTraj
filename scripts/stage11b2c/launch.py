"""Parent wall clock charges the entire GPU worker including imports and exit."""
from .common import *
import os,threading

def main():
    name=sys.argv[1];assert name in {'precheck','extract'}
    prior=parent_seconds();assert prior<6600
    start=time.perf_counter();env=os.environ.copy();env['HTRAJ_PROCESS_MONOTONIC_START']=str(start)
    source=commit()
    stopped=threading.Event()
    def monitor():
        with (OUT/f'GPU_card_{name}.csv').open('x') as g:
            g.write('elapsed_seconds,timestamp,index,name,memory_used_MiB,memory_total_MiB,utilization_gpu_percent,temperature_C,power_W\n')
            while not stopped.is_set():
                try:
                    line=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                    g.write(str(time.perf_counter()-start)+','+line+'\n');g.flush()
                except Exception as e:g.write('# monitor error '+str(e)+'\n');g.flush()
                stopped.wait(15)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    with (OUT/f'{name}_stdout.txt').open('x') as f:
        try:r=subprocess.run([sys.executable,'-u','-m','scripts.stage11b2c.'+name],cwd=ROOT,env=env,
            stdout=f,stderr=subprocess.STDOUT,timeout=max(1,7200-prior-120),check=False);code=r.returncode
        except subprocess.TimeoutExpired:code=124
    stopped.set();thread.join(timeout=15)
    elapsed=time.perf_counter()-start
    jnew(OUT/f'process_{name}.json',dict(worker=name,started_commit=source,returncode=code,wall_seconds=elapsed,
        prior_seconds=prior,cumulative_seconds=prior+elapsed,hard_seconds=7200,safety_seconds=6600,
        interpretation='Whole GPU process including Python startup/load/checks/train/validation/save/exit; CPU-only reporting excluded'))
    print(name,'return',code,'wall',round(elapsed,3),'cumulative',round(prior+elapsed,3),flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':main()
