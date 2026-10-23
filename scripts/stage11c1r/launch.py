"""Cumulative parent wall accounting includes failures and whole worker lifetime."""
from .common import *
def main():
    name=sys.argv[1];assert name in {'diagnostic_train','profile'}
    verify_frozen();prior=parent_seconds();expected={'diagnostic_train':750,'profile':1500}[name]
    if prior+expected+60>=2250 or ORIGINAL_USED+prior+expected+60>=18000:
        raise SystemExit('NO_STAGE_START: conservative stage admission insufficient')
    started=time.perf_counter();env=os.environ.copy();env['HTRAJ_PROCESS_MONOTONIC_START']=str(started)
    env['HTRAJ_PRIOR_GPU_SECONDS']=str(prior)
    deadline=min(2400-prior-45,2250-prior,18000-ORIGINAL_USED-prior-45)
    with (OUT/f'{name}_stdout.txt').open('x') as f:
        try:
            process=subprocess.run([sys.executable,'-u','-m','scripts.stage11c1r.'+name],cwd=ROOT,env=env,
                stdout=f,stderr=subprocess.STDOUT,timeout=deadline,check=False);code=process.returncode
        except subprocess.TimeoutExpired:code=124
    wall=time.perf_counter()-started
    jnew(OUT/f'process_{name}.json',dict(worker=name,source_commit=commit(),returncode=code,wall_seconds=wall,
        Stage11C1RUsed=prior+wall,OriginalStage11C1Used=ORIGINAL_USED,CombinedUsed=ORIGINAL_USED+prior+wall,
        hard_seconds=2400,safety_seconds=2250,parent_timeout_seconds=deadline,
        includes='Startup,imports,model load,training/inference,serialization and process exit; failures included'))
    print(name,'return',code,'wall',wall,'stage_total',prior+wall,flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':main()
