"""Post-run fixed-weight CUDA check, no optimizer or extra training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from src.socialvae_causal.data import CausalDataset,collate_examples
from src.socialvae_causal.training import new_model,seed_all
from src.socialvae_baseline.evaluation import batch_noise,fixed_draws
from src.socialvae_baseline.artifacts import json_new,state_info

plan=json.load(open('configs/stage5p2/protocol.json'));rows=[]
for rec in ['crowds_zara02','biwi_hotel']:
    seed_all(17)
    selection=json.load(open(f'results/tables/stage5p2/{rec}/selection.json'))
    model=new_model();model.load_state_dict(torch.load(selection['best']['path'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
    before=state_info(model)
    data=CausalDataset(plan['data_root'],plan['manifests'][rec]['path'],'validation')
    indices=np.linspace(0,len(data)-1,16,dtype=int).tolist()
    items=[data[i] for i in indices]
    permutation=np.random.default_rng(17).permutation(len(items));inverse=np.argsort(permutation)
    def predict(items):
        x,y,n=collate_examples(items,'cuda')
        with torch.no_grad():
            prior=model(x,n)
            with fixed_draws(model.p_z,batch_noise(items,False,'cuda')):sample=model(x,n,20)
        return prior,sample
    prior,sample=predict(items)
    pp,sp=predict([items[i] for i in permutation]);pp=pp[:,inverse];sp=sp[:,:,inverse]
    halves=[predict(items[:7]),predict(items[7:])]
    hp=torch.cat([p[0] for p in halves],1);hs=torch.cat([p[1] for p in halves],2)
    for name,a,b in [('permutation_K1',prior,pp),('permutation_K20',sample,sp),('batch_partition_K1',prior,hp),('batch_partition_K20',sample,hs)]:
        difference=float((a-b).abs().max())
        close=bool(torch.allclose(a,b,rtol=1e-5,atol=1e-5))
        rows.append(dict(recording=rec,test=name,target_indices=indices,max_absolute_difference_m=difference,atol=1e-5,rtol=1e-5,passed=close,epsilon_order_partition_invariant=True))
    assert before==state_info(model) and not model.p_z._forward_hooks and not model.q_z._forward_hooks
    del model
json_new('results/tables/stage5p2/CUDA_sampling_invariance_diagnostic.json',rows)
print(json.dumps(rows,indent=2))
