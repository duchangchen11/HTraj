"""Fixed work-conserving queues: at most 2 Full + 2 Placebo, exactly 8 runs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import asyncio,json,time
from src.socialvae_baseline.artifacts import json_new
ROOT=Path('results/tables/stage5p5')

async def main():
 plan=json.load(open('configs/stage5p5/protocol.json'));start=time.perf_counter();events=[]
 log=(ROOT/'scheduler_execution.jsonl').open('x',buffering=1)
 def event(kind,**data):
  r=dict(event=kind,elapsed_seconds=time.perf_counter()-start,**data);events.append(r)
  log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
 queues={kind:asyncio.Queue() for kind in plan['models']}
 for kind in plan['models']:
  for seed in plan['new_seeds']:
   for rec in ['crowds_zara02','biwi_hotel']:queues[kind].put_nowait((seed,rec))
 async def worker(kind,worker_id):
  queue=queues[kind]
  while not queue.empty():
   seed,rec=queue.get_nowait();run_id=f'seed{seed}_{kind}_{rec}'
   directory=Path('experiments/stage5p5/logs');directory.mkdir(parents=True,exist_ok=True)
   with (directory/f'{run_id}.txt').open('x') as stream:
    event('launch',run_id=run_id,model=kind,seed=seed,recording=rec,worker=worker_id)
    proc=await asyncio.create_subprocess_exec(sys.executable,'scripts/stage5p5/train.py','--model',kind,'--seed',str(seed),'--recording',rec,
     stdout=stream,stderr=asyncio.subprocess.STDOUT)
    code=await proc.wait()
   event('finish',run_id=run_id,model=kind,seed=seed,recording=rec,returncode=code,worker=worker_id)
   queue.task_done()
 await asyncio.gather(*(worker(k,i) for k in plan['models'] for i in range(2)))
 log.close()
 json_new(ROOT/'scheduler_completion.json',dict(actual_wall_seconds=time.perf_counter()-start,events=events,
  launched_runs=sum(e['event']=='launch' for e in events),all_succeeded=all(e['returncode']==0 for e in events if e['event']=='finish'),
  model_hyperparameter_or_seed_selection=False,extra_runs=0))
 with (ROOT/'scheduler_execution.txt').open('x') as f:f.write((ROOT/'scheduler_execution.jsonl').read_text())

if __name__=='__main__':asyncio.run(main())
