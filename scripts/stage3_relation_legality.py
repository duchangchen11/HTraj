"""History-only graph legality/budget and backend sensitivity audit."""
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
    torch.set_num_threads(1);p=json.loads(Path('configs/stage3/protocol.json').read_text());_,val=datasets(p);stats={device:Counter() for device in ('cpu','cuda')};backend=Counter()
    for batch in DataLoader(val,batch_size=8,collate_fn=collate_trajectories):
        masks={}
        for device in ('cpu','cuda'):
            e=motion_inputs(batch['observed'].to(device),batch['observed_valid'].to(device),batch['timestamps'].to(device))
            for t in range(8):
                real,_=receiver_candidates(e,t);random,_=receiver_candidates(e,t,random=True);b,i,j=random.nonzero(as_tuple=True)
                assert not random.diagonal(dim1=-2,dim2=-1).any() and e['valid'][b,t,i].all() and e['continuity'][b,t,j].all()
                distance=(e['positions'][b,t,i]-e['positions'][b,t,j]).norm(dim=-1)
                stats[device].update(real_edges=int(real.sum()),random_edges=int(random.sum()),replaced_edges=int((random & ~real).sum()),outside_radius_edges=int((distance>4).sum()),
                    receivers_degree_changed=int((random.sum(-1)!=real.sum(-1)).sum()),incoming_L1_difference=int((random.sum(-1)-real.sum(-1)).abs().sum()))
                masks[(device,t,'real')]=real.cpu();masks[(device,t,'random')]=random.cpu()
        for t in range(8):
            for kind in ('real','random'):backend[kind+'_symmetric_difference_edges']+=int((masks[('cpu',t,kind)]^masks[('cuda',t,kind)]).sum())
    output={}
    for device,c in stats.items():
        output[device]=dict(c,replacement_fraction=c['replaced_edges']/c['random_edges'],outside_radius_fraction=c['outside_radius_edges']/c['random_edges'])
    write_json_new('results/tables/stage3/relation_legality.json',dict(backends=output,cpu_cuda_difference=dict(backend),history_only=True,test_evaluated=False,
        training_and_main_evaluation_device='cuda',hash='float32 position/time sinusoid; deterministic within backend but no cross-device bitwise guarantee; ties may reduce budget',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__)))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
