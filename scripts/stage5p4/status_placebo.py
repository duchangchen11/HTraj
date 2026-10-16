"""Read-only progress, all completed epochs rather than latest-point selection."""
import json
from pathlib import Path
for rec in ['crowds_zara02','biwi_hotel']:
    f=Path('results/tables/stage5p4/placebo_social')/rec/'execution.jsonl'
    if not f.exists():continue
    events=[]
    for line in f.read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    rows=[e for e in events if e['event']=='epoch']
    if rows:
        best=min(rows,key=lambda r:r['K1_ADE']);last=rows[-1]
        print(json.dumps(dict(recording=rec,completed=len(rows),status=events[-1]['event'],
            best_epoch=best['epoch'],best_ADE=best['K1_ADE'],best_FDE=best['K1_FDE'],
            best_minADE20=best['minADE20'],last_loss=last['train_loss'],
            last_ADE=last['K1_ADE'],last_FDE=last['K1_FDE'],last_epoch_seconds=last['epoch_wall_seconds'],
            last_disk_free_GiB=last['free_disk_bytes']/2**30)))
