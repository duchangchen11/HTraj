"""Read-only status: never selects seeds or modifies an execution budget."""
import json
from pathlib import Path
for seed in [42,2026]:
 for kind in ['Full','PlaceboSocial']:
  for rec in ['crowds_zara02','biwi_hotel']:
   p=Path('results/tables/stage5p5/runs')/f'seed{seed}'/kind/rec/'execution.jsonl'
   if not p.exists():continue
   events=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
   rr=[r for r in events if r['event']=='epoch']
   if not rr:print(seed,kind,rec,'initializing');continue
   last=rr[-1];best=min(rr,key=lambda x:x['K1_ADE'])
   print(json.dumps(dict(seed=seed,model=kind,recording=rec,epochs=len(rr),status=events[-1]['event'],
    best_epoch=best['epoch'],best_ADE=round(best['K1_ADE'],6),best_FDE=round(best['K1_FDE'],6),
    last_ADE=round(last['K1_ADE'],6),last_loss=round(last['train_loss'],6),seconds_per_epoch=round(last['epoch_wall_seconds'],2),
    free_GiB=round(last['free_disk_bytes']/2**30,2))))
