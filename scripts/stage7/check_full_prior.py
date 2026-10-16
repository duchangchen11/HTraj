"""Supplement: retain learned prior, couple epsilon by agent, no weight updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time
from unittest.mock import patch
import numpy as np
import torch
from scripts.stage7.admit import agentformer,scalar_error
from src.public_admission.inputs import synthetic_scene,agentformer_packet
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256


def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    meta,build,_,_,settings=agentformer(True)
    from model.common.dist import Normal
    model=build().eval();rows=[]
    def infer(h,epsilon,future=None):
        packet=agentformer_packet(h)
        if future is not None:packet['fut_motion_3D']=[torch.tensor(v,dtype=torch.float32) for v in future]
        model.set_data(packet)
        def sampled(distribution):
            e=epsilon.permute(1,0,2).reshape(-1,32).to(distribution.mu)
            assert e.shape==distribution.mu.shape
            return distribution.mu+distribution.sigma*e
        with patch.object(Normal,'sample',sampled):
            prediction,_=model.inference(mode='infer',sample_num=len(epsilon))
        prior=model.data['p_z_dist_infer']
        return prediction.permute(1,0,2,3),prior.mu.detach().clone(),prior.logvar.detach().clone()
    for n in [1,2,5,12]:
        h,y=synthetic_scene(n);p=np.roll(np.arange(n),1);inverse=np.argsort(p)
        with torch.no_grad():
            zero=torch.zeros(1,n,32);a,mu,logvar=infer(h,zero)
            b,pmu,plogvar=infer(h[p],zero[:,p]);b=b[:,inverse]
            poison,_,_=infer(h,zero,np.full_like(y,np.nan))
            eps=torch.randn(20,n,32);k20,_,_=infer(h,eps);p20,_,_=infer(h[p],eps[:,p]);p20=p20[:,inverse]
            cp=Path(f'experiments/stage7/cache/AgentFormer_full_prior_N{n}.pth')
            with cp.open('xb') as f:torch.save(model.state_dict(),f)
            restored=build().eval();restored.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
            old=model;model=restored;recovered,_,_=infer(h,zero);model=old
        row=dict(people=n,prior_mean_permutation_max=scalar_error(mu,pmu[inverse]),prior_logvar_permutation_max=scalar_error(logvar,plogvar[inverse]),
            K1_prior_mean_permutation_max=scalar_error(a,b),K20_coupled_epsilon_permutation_max=scalar_error(k20,p20),
            future_poison_max=scalar_error(a,poison),strict_restore_max=scalar_error(a,recovered),checkpoint=str(cp),checkpoint_sha256=sha256(cp),
            passed=bool(torch.allclose(a,b,atol=1e-5,rtol=1e-5) and torch.allclose(k20,p20,atol=1e-5,rtol=1e-5)
                and torch.allclose(mu,pmu[inverse],atol=1e-5,rtol=1e-5) and torch.equal(a,poison) and torch.equal(a,recovered)))
        rows.append(row)
    h,_=synthetic_scene(2);extra,_=synthetic_scene(3);extra+=100.
    with torch.no_grad():
        base,_,_=infer(h,torch.zeros(1,2,32));extended,_,_=infer(np.concatenate([h,extra]),torch.zeros(1,5,32))
    probe=dict(added_people=3,offset_m=100.,same_original_histories=True,
        original_agent_prediction_max_difference_m=scalar_error(base,extended[:,:2]),
        interpretation='Synthetic fixed-weight context dependence, not observed forecast error or proven degradation; scene centering and attention both change')
    json_new('results/tables/stage7/AgentFormer_full_prior_checks.json',dict(repository_commit=meta['commit'],cases=rows,all_passed=all(r['passed'] for r in rows),
        distant_context_probe=probe,model_source_modified=False,noise_control='mu+sigma*agent-coupled epsilon; original learned prior retained; no GT-based selection',
        official_pretrained_loaded=False,formal_training_runs=0,optimizer_updates=0,ETH_evaluated=False,
        source_SHA=sha256(__file__),elapsed_seconds=time.perf_counter()-start))
    print('Full learned-prior and coupled-epsilon checks:',rows,'context probe',probe,flush=True)


if __name__=='__main__':main()
