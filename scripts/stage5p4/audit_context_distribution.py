"""History-only operative-context distribution, full frozen training/validation pools."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time
import numpy as np
import torch
from src.socialvae_causal.data import CausalDataset
from src.socialvae_capacity.placebo import synthetic_context
from src.socialvae_causal.io import csv_new
from src.socialvae_baseline.artifacts import json_new
plan=json.load(open('configs/stage5p4/placebo_protocol.json'));rows=[];start=time.perf_counter()
torch.set_num_threads(1)
for rec,manifest in plan['manifests'].items():
 for split in ['train','validation']:
  ds=CausalDataset(plan['data_root'],manifest['path'],split)
  values={kind:dict(count=[],distance=[],speed=[],relative_speed=[],entry_times=[],context_width=[]) for kind in ['real','synthetic']}
  for i in range(len(ds)):
   item=ds[i];x=item['x'];real=item['neighbor'][:8]
   fake=synthetic_context(torch.as_tensor(x,dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
   for kind,n in [('real',real),('synthetic',fake)]:
    dp=n[:,:,:2]-x[:,:2][:,None,:]
    distance=np.linalg.norm(dp,axis=-1);valid=distance<=2
    v=values[kind];v['count'].extend(valid.sum(axis=1).tolist());v['distance'].extend(distance[valid].tolist());v['context_width'].append(n.shape[1])
    usable=valid[1:];speed=np.linalg.norm(n[1:,:,2:4],axis=-1)/.4
    relative=np.linalg.norm(n[1:,:,2:4]-x[1:,2:4][:,None,:],axis=-1)/.4
    v['speed'].extend(speed[usable].tolist());v['relative_speed'].extend(relative[usable].tolist())
    entered=valid & ~np.vstack([np.zeros((1,n.shape[1]),dtype=bool),valid[:-1]])
    v['entry_times'].extend(np.nonzero(entered)[0].tolist())
  for kind,v in values.items():
   for field,xx in v.items():
    a=np.asarray(xx,dtype=np.float64)
    rows.append(dict(recording=rec,split=split,context=kind,statistic=field,target_windows=len(ds),observations=len(xx),
     mean=float(a.mean()) if a.size else None,P05=float(np.percentile(a,5)) if a.size else None,
     P50=float(np.percentile(a,50)) if a.size else None,P95=float(np.percentile(a,95)) if a.size else None,
     min=float(a.min()) if a.size else None,max=float(a.max()) if a.size else None,
     fraction_zero=float((a==0).mean()) if a.size else None))
csv_new('results/tables/stage5p4/context_distribution.csv',rows)
json_new('results/tables/stage5p4/context_distribution_audit.json',dict(actual_wall_seconds=time.perf_counter()-start,
 future_used=False,original_neighbor_packet_only_history=True,all_supervised_target_pools=True,
 synthetic_count=4,always_within_radius=True,future_slots='all padding',
 caveat='operative real context packet masks missing source velocities; frame0 only position participates. Counts include zero-message frames. Synthetic contexts have complete 8-frame support; not distribution-matched.'))
print('All history-only context distributions saved',len(rows))
