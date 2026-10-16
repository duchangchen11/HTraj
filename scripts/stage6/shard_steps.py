"""Preserve immutable full stream locally and track every row in small shards."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

root=Path('results/tables/stage6/diagnosis');source=root/'per_step_motion_diagnostics.csv.gz'
parts=defaultdict(list)
for r in read_csv(source):parts[(r['recording'],r['seed'],r['model'])].append(r)
out=root/'step_shards';out.mkdir();records=[]
for k,rr in sorted(parts.items()):
 p=out/f'{k[0]}_seed{k[1]}_{k[2]}.csv.gz';csv_new(p,rr)
 records.append(dict(path=str(p),sha256=sha256(p),rows=len(rr),bytes=p.stat().st_size))
json_new(root/'step_shard_index.json',dict(original_stream=str(source),original_sha256=sha256(source),original_retained_locally=True,
 shards=records,total_rows=sum(len(r) for r in parts.values()),no_rows_discarded=True))
print('All step diagnostics preserved in',len(parts),'shards')
