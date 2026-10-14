"""Separate empty legacy graph from physically available historical neighbors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
from collections import Counter
import torch
from torch.utils.data import DataLoader
from src.data.dataset import TrajectoryDataset
from src.data.benchmark_audit import apply_protocol,PROTOCOLS
from src.data.collate import collate_trajectories
from src.models.ego_encoder import motion_inputs
from src.models.relation_routing import receiver_candidates
from src.evaluation.benchmark_audit import history_features
from src.data.preprocessing import sha256,write_json_new


def main():
    torch.set_num_threads(1);manifest='configs/stage3p5/folds/val_students003.json';data=TrajectoryDataset('data/raw/eth_ucy',manifest,'validation');stats={p:Counter() for p in PROTOCOLS}
    for raw in DataLoader(data,batch_size=8,collate_fn=collate_trajectories):
        raw_e=motion_inputs(raw['observed'],raw['observed_valid'],raw['timestamps']);pos=raw_e['positions'][:,-1];n=pos.shape[1]
        distance=(pos[:,:,None]-pos[:,None,:]).norm(dim=-1);physical=raw['observed_valid'][:,-1,:,None]&raw['observed_valid'][:,-1,None,:]&(distance<=4)&~torch.eye(n,dtype=torch.bool)[None]
        for p in PROTOCOLS:
            # apply_protocol operates sample axes T,N; equivalent batched masks.
            b=dict(raw)
            if p.startswith('A_'):
                full=raw['observed_valid'].all(1);future=raw['future_valid'].all(1)
                b['observed_valid']=raw['observed_valid']&full[:,None];b['future_valid']=raw['future_valid']&(full&future)[:,None]
            g,_=history_features(b);e=motion_inputs(b['observed'],b['observed_valid'],b['timestamps']);eligible,_=receiver_candidates(e,7)
            supervised=b['future_valid'].any(1);empty=g['no_neighbors']&supervised;reliable=eligible.any(-1)
            stats[p].update(targets=int(supervised.sum()),empty_legacy_graph_targets=int(empty.sum()),empty_graph_with_physical_neighbors=int((empty&physical.any(-1)).sum()),
                empty_graph_with_reliable_protocol_source=int((empty&reliable).sum()),empty_graph_one_history_targets=int((empty&(raw['observed_valid'].sum(1)==1)).sum()))
    write_json_new('results/tables/stage3p5/neighbor_eligibility_audit.json',dict(protocols={p:dict(v) for p,v in stats.items()},
        history_only_geometry=True,future_GT_used_for_groups=False,GT_availability_only_defines_evaluable_target_population=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),ETH_evaluated=False))


if __name__=='__main__':main()
