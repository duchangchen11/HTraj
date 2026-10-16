"""Compact read-only monitor for the fixed eight-run queue."""
import json
from pathlib import Path
completed=0;started=0;total_epochs=0
for seed in [42,2026]:
 for model in ['Full','PlaceboSocial']:
  for rec in ['crowds_zara02','biwi_hotel']:
   p=Path(f'results/tables/stage5p5/runs/seed{seed}/{model}/{rec}/execution.jsonl')
   if not p.exists():continue
   started+=1;events=[]
   for line in p.read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:pass
   rr=[r for r in events if r['event']=='epoch'];total_epochs+=len(rr)
   status=events[-1]['event'] if events else 'initializing'
   if status=='completed':completed+=1;continue
   if rr:
    best=min(rr,key=lambda r:r['K1_ADE'])
    print(f"seed{seed} {model} {rec}: {len(rr)}/100, best e{best['epoch']} {best['K1_ADE']:.6f}/{best['K1_FDE']:.6f}, {status}")
   else:print(f'seed{seed} {model} {rec}: initializing or failed; {status}')
print(f'{completed}/8 completed, {started}/8 started, {total_epochs}/800 epochs')
