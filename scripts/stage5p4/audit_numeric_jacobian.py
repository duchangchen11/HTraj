"""Central-difference neighbor-input Jacobian despite official no_grad.

Fixed epsilon=0.01m, at most three observed source position coordinates within
1.9m of target, no GT-based coordinate choice. Diagnostic only, no new training.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from src.socialvae_capacity.model import new_model
from src.socialvae_causal.data import CausalDataset,collate_examples
from src.socialvae_causal.training import seed_all
from src.socialvae_baseline.artifacts import json_new

plan=json.load(open('configs/stage5p4/ego_protocol.json'));rows=[]
for rec in ['crowds_zara02','biwi_hotel']:
    data=CausalDataset(plan['data_root'],plan['manifests'][rec]['path'],'validation')
    indices=np.linspace(0,len(data)-1,8,dtype=int).tolist();items=[data[i] for i in indices]
    x,y,n=collate_examples(items,'cuda')
    distance=(n[:8,:,:,:2]-x[:8,:,:2].unsqueeze(2)).norm(dim=-1)
    coordinates=torch.nonzero(distance<1.9)[:3].tolist()
    for label in ['EgoCapacity']:
        seed_all(17)
        source=Path('results/tables/stage5p4/ego_capacity')/rec
        selected=json.load(open(source/'selection.json'))
        model=new_model()
        model.load_state_dict(torch.load(selected['best']['path'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
        for t,b,j in coordinates:
            plus=n.clone();minus=n.clone();plus[t,b,j,0]+=.01;minus[t,b,j,0]-=.01
            with torch.no_grad():a=model(x,plus);c=model(x,minus)
            derivative=(a-c)/.02
            maximum=float(derivative.abs().max())
            if label=='EgoCapacity':assert maximum==0.
            rows.append(dict(recording=rec,model=label,coordinate=[t,b,j,0],epsilon_m=.01,
                max_abs_central_difference_jacobian=maximum,
                same_target_max_abs_jacobian=float(derivative[:,b].abs().max()),
                other_target_max_abs_jacobian=float(derivative[:,[i for i in range(len(items)) if i!=b]].abs().max()),
                checkpoint_sha256=selected['best']['sha256'],target_indices=indices,
                interpretation='finite difference functional derivative, not autograd; position-only artificial input diagnostic'))
        del model
json_new('results/tables/stage5p4/numerical_input_jacobian.json',rows)
print('EgoCapacity numerical input Jacobians exactly zero; target-only input independence checked numerically',len(rows))
