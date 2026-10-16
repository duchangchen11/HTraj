"""Launch exactly four frozen, independent runs; no adaptive scheduling choices."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import asyncio,json,time
from src.socialvae_baseline.artifacts import json_new


async def main():
    root=Path('results/tables/stage6p1');plan=json.load(open('configs/stage6p1/protocol.json'))
    start=time.perf_counter();events=[]
    log=(root/'scheduler_execution.txt').open('x',buffering=1)
    def event(kind,**values):
        row=dict(event=kind,elapsed_seconds=time.perf_counter()-start,**values)
        events.append(row);line=json.dumps(row);log.write(line+'\n');print(line,flush=True)
    async def worker(run):
        name=run['name'];directory=Path('experiments/stage6p1/logs');directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'{name}.txt').open('x') as stream:
            event('launch',**run)
            process=await asyncio.create_subprocess_exec(sys.executable,'scripts/stage6p1/train.py',
                '--recording',run['recording'],'--condition',run['condition'],stdout=stream,stderr=asyncio.subprocess.STDOUT)
            code=await process.wait()
        event('finish',**run,returncode=code)
    await asyncio.gather(*(worker(r) for r in plan['formal_runs']))
    log.close()
    json_new(root/'scheduler_completion.json',dict(actual_wall_seconds=time.perf_counter()-start,events=events,
        launched_runs=4,all_succeeded=all(e['returncode']==0 for e in events if e['event']=='finish'),extra_runs=0,
        settings_changed_after_validation=False))


if __name__=='__main__':asyncio.run(main())
