"""Quantify preserved h_source,t-1 payload's motion validity, history only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
from collections import Counter
import torch
from torch.utils.data import DataLoader
from src.training.order_verification import datasets
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256,write_json_new
from src.models.ego_encoder import motion_inputs
from src.models.relation_routing import receiver_candidates


def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage3/protocol.json').read_text());train,val=datasets(p);output={}
    for split,data in [('train',train),('development',val)]:
        c=Counter()
        for batch in DataLoader(data,batch_size=8,collate_fn=collate_trajectories):
            e=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps'])
            for t in range(1,8):
                mask,_=receiver_candidates(e,t);known_before=e['continuity'][:,t-1]
                c['all_messages']+=int(mask.sum());c['payload_prior_motion_missing']+=int((mask & ~known_before[:,None,:]).sum())
                if t==7:
                    one=e['valid'].sum(1)==1;selected=mask & one[:,:,None]
                    known=selected & known_before[:,None,:]
                    c['one_frame_relation_edges']+=int(selected.sum());c['one_frame_edges_prior_motion_known']+=int(known.sum())
                    c['one_frame_neighbor_targets']+=int(selected.any(-1).sum());c['one_frame_targets_any_prior_motion_payload']+=int(known.any(-1).sum())
        output[split]=dict(c)
    write_json_new('results/tables/stage3/payload_freshness_audit.json',dict(splits=output,history_only=True,test_evaluated=False,
        current_source_velocity_always_observable=True,message_timing='legacy previous ego retained; missing-velocity flags are not fabricated speeds',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__)))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
