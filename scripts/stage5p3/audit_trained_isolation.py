"""Fixed trained checkpoints: interventions are diagnostics, not new metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
from src.socialvae_attribution.model import new_model,OfficialSocialVAE
from src.socialvae_causal.data import CausalDataset,collate_examples,PADDING
from src.socialvae_baseline.evaluation import fixed_draws,batch_noise
from src.socialvae_baseline.artifacts import json_new,state_info
from src.socialvae_causal.training import seed_all

plan=json.load(open('configs/stage5p3/protocol.json'));rows=[]
for rec in ['crowds_zara02','biwi_hotel']:
    data=CausalDataset(plan['data_root'],plan['manifests'][rec]['path'],'validation')
    indices=np.linspace(0,len(data)-1,8,dtype=int).tolist();items=[data[i] for i in indices]
    x,y,n=collate_examples(items,'cuda')
    visible=n[:8,:,:,0]<1e8
    position=n.clone();position[:8,:,:,:2]+=visible[:,:,:,None]*torch.tensor([.3,-.2],device='cuda')
    velocity=n.clone();velocity[:8,:,:,2:4]+=visible[:,:,:,None]*torch.tensor([.2,-.1],device='cuda')
    entry=n.clone();entry[:6]=float(PADDING)
    variants={'position_shift':position,'velocity_feature_change':velocity,'count_empty':n[:,:,:0],
        'count_duplicate':n.repeat(1,1,2,1),'permutation':n.flip(2),'entry_delay':entry}
    for label in ['Full','NoSocial']:
        seed_all(17)
        source=Path('results/tables/stage5p2' if label=='Full' else 'results/tables/stage5p3')/rec
        selection=json.load(open(source/'selection.json'))
        model=OfficialSocialVAE(12,2,256).cuda() if label=='Full' else new_model()
        model.load_state_dict(torch.load(selection['best']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        before=state_info(model)
        def predictions(neighbor,mode):
            if mode=='K1':model.eval();return (model(x,neighbor),)
            if mode=='K20':
                model.eval()
                with fixed_draws(model.p_z,batch_noise(items,False,'cuda')):return (model(x,neighbor,20),)
            model.train()
            if mode=='posterior':
                h,b=model.enc(x,neighbor,y=y);d=model.q_z(model.rnn_fy_init(h),b[0]);return h,b,d.loc,d.scale
            with fixed_draws(model.q_z,batch_noise(items,True,'cuda')):return tuple(model.loss(*model(x,y,neighbor)).values())
        with torch.no_grad():
            for mode in ['K1','K20','posterior','training_loss']:
                reference=predictions(n,mode)
                for intervention,neighbor in variants.items():
                    changed=predictions(neighbor,mode)
                    maximum=max(float((a-b).abs().max()) for a,b in zip(reference,changed))
                    equal=all(torch.equal(a,b) for a,b in zip(reference,changed))
                    if label=='NoSocial':assert equal,(rec,mode,intervention,maximum)
                    rows.append(dict(recording=rec,model=label,mode=mode,intervention=intervention,max_abs_output_difference=maximum,
                        strictly_equal=equal,checkpoint_sha256=selection['best']['sha256'],target_indices=indices))
        model.eval();neighbor=n.detach().clone().requires_grad_()
        output=model(x,neighbor)
        jac=torch.autograd.grad(output.sum(),neighbor,allow_unused=True)[0]
        assert jac is None
        rows.append(dict(recording=rec,model=label,mode='raw_neighbor_Jacobian',intervention='autograd',max_abs_output_difference=0.,
            strictly_equal=True,checkpoint_sha256=selection['best']['sha256'],target_indices=indices,
            interpretation='unused for NoSocial; upstream feature no_grad for Full, NOT proof of Full isolation'))
        assert before==state_info(model)
        del model
json_new('results/tables/stage5p3/trained_isolation_audit.json',rows)
print('PASS: trained NoSocial exact intervention independence, Full diagnostics preserved',len(rows))
