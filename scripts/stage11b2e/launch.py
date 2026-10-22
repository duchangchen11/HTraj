"""Entire GPU subprocess wall meter, including startup, loading and shutdown."""
from .common import *
import threading

def main():
    mode=sys.argv[1];assert mode in {'regression','decode'}
    prior=parent_seconds();assert prior<3300-60
    start=time.perf_counter();env=os.environ.copy();env['HTRAJ_GPU_PROCESS_START']=str(start)
    stop=threading.Event()
    def monitor():
        with (OUT/f'GPU_card_{mode}.csv').open('x') as f:
            f.write('elapsed_seconds,timestamp,index,name,memory_used_MiB,memory_total_MiB,utilization_gpu_percent,temperature_C,power_W\n')
            while not stop.is_set():
                try:
                    line=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                    f.write(str(time.perf_counter()-start)+','+line+'\n');f.flush()
                except Exception as ex:f.write('# monitor error '+str(ex)+'\n');f.flush()
                stop.wait(10)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    with (OUT/f'{mode}_stdout.txt').open('x') as f:
        try:
            r=subprocess.run([sys.executable,'-u','-m','scripts.stage11b2e.worker',mode],cwd=ROOT,env=env,
                stdout=f,stderr=subprocess.STDOUT,timeout=max(1,3300-prior-30));code=r.returncode
        except subprocess.TimeoutExpired:code=124
    stop.set();thread.join(timeout=6);elapsed=time.perf_counter()-start
    jnew(OUT/f'process_{mode}.json',dict(worker=mode,returncode=code,wall_seconds=elapsed,prior_seconds=prior,
        cumulative_seconds=prior+elapsed,hard_seconds=3600,safety_seconds=3300,source_commit=commit(),
        scope='Entire child process including startup/imports/model load/regression or all-target decode/save and exit; monitor shutdown included'))
    print(mode,'exit',code,'GPU process wall',elapsed,'cumulative',prior+elapsed,flush=True)
    assert prior+elapsed<3600
    if code:raise SystemExit(code)

if __name__=='__main__':main()
