"""Bounded actual synthetic tests. No optimizer.step or real-scene evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,importlib.util,json,time,traceback
import numpy as np
import torch
import yaml
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new
from src.public_admission.inputs import synthetic_scene,agentformer_packet,pecnet_packet

ROOT=Path('results/tables/stage7');CACHE=Path('experiments/stage7/cache')


def scalar_error(a,b):return float((a-b).detach().abs().max())


def pecnet():
    meta=json.load(open(ROOT/'sources/PECNet.json'));folder=Path(meta['source_directory'])/'PECNet'
    spec=importlib.util.spec_from_file_location('stage7_pecnet_models',folder/'utils/models.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    loss_spec=importlib.util.spec_from_file_location('stage7_pecnet_social',folder/'utils/social_utils.py');social=importlib.util.module_from_spec(loss_spec);loss_spec.loader.exec_module(social)
    cfg=yaml.safe_load((folder/'config/optimal.yaml').read_text())
    args=[cfg[k] for k in ['enc_past_size','enc_dest_size','enc_latent_size','dec_size','predictor_hidden_size','non_local_theta_size',
        'non_local_phi_size','non_local_g_size','fdim','zdim','nonlocal_pools','non_local_dim','sigma','past_length','future_length']]
    def build():return module.PECNet(*args,False).double().cuda()
    def predict(model,h,eps,future=None):
        x,pos,mask=[v.cuda() for v in pecnet_packet(h)];predictions=[]
        # Evaluation-only control of latent draws: input features, weights,
        # generated-goal conditioning and social computation remain unchanged.
        for noise in eps:
            def hook(_m,inputs):
                value=inputs[0].clone();value[:,-cfg['zdim']:]=noise.to(value)*cfg['sigma'];return (value,)
            handle=model.decoder.register_forward_pre_hook(hook)
            try:goal=model(x,pos,device=torch.device('cuda'))
            finally:handle.remove()
            path=model.predict(x,goal,mask,pos).reshape(len(h),11,2)
            output=torch.cat((path,goal[:,None]),1)/cfg['data_scale']
            output=output+torch.tensor(h[:,:1],device='cuda',dtype=torch.float64)
            predictions.append(output)
        return torch.stack(predictions)
    def loss(model,h,y):
        x,pos,mask=[v.cuda() for v in pecnet_packet(h)]
        truth=torch.tensor((y-h[:,:1])*cfg['data_scale'],device='cuda',dtype=torch.float64)
        out=model(x,pos,dest=truth[:,-1],mask=mask,device=torch.device('cuda'))
        parts=social.calculate_loss(truth[:,-1],*out[:3],torch.nn.MSELoss(),truth[:,:-1].reshape(len(h),-1),out[3])
        return parts[0]+cfg['kld_reg']*parts[1]+cfg['adl_reg']*parts[2]
    return meta,build,predict,loss,dict(latent_dim=cfg['zdim'],configuration=cfg,compatibility_changes=[])


def agentformer(compat):
    meta=json.load(open(ROOT/'sources/AgentFormer.json'));folder=Path(meta['source_directory'])
    sys.path[:0]=[str(CACHE/'deps'),str(folder)]
    changes=[]
    if compat:
        import torch.nn.modules.linear as linear
        if not hasattr(linear,'_LinearWithBias'):
            linear._LinearWithBias=torch.nn.Linear
            changes.append('Runtime alias removed private _LinearWithBias to nn.Linear (default bias=True); source bytes untouched')
    from easydict import EasyDict
    from model.agentformer import AgentFormer
    cfg=EasyDict(yaml.safe_load((folder/'cfg/eth_ucy/hotel/hotel_agentformer_pre.yml').read_text()))
    def build():
        model=AgentFormer(cfg);model.set_device(torch.device('cuda'));return model
    def predict(model,h,eps,future=None):
        packet=agentformer_packet(h)
        if future is not None:packet['fut_motion_3D']=[torch.as_tensor(v,dtype=torch.float32) for v in future]
        model.set_data(packet)
        def hook(_m,args,kwargs):
            distribution=model.data['p_z_dist_infer'] if model.data['p_z_dist_infer'] is not None else None
            # Fixed actual z is permitted by the official decoder interface;
            # zero is a latent-control path, not the learned-prior mean.
            return args,dict(kwargs,z=eps.permute(1,0,2).reshape(-1,cfg.nz).to('cuda'))
        handle=model.future_decoder.register_forward_pre_hook(hook,with_kwargs=True)
        try:output,_=model.inference(mode='infer',sample_num=len(eps))
        finally:handle.remove()
        return output.permute(1,0,2,3)
    def loss(model,h,y):
        packet=agentformer_packet(h);packet['fut_motion_3D']=[torch.tensor(v,dtype=torch.float32) for v in y]
        packet['fut_motion_mask']=[torch.ones(12) for _ in h]
        model.set_data(packet);model();return model.compute_loss()[0]
    return meta,build,predict,loss,dict(latent_dim=cfg.nz,configuration=dict(cfg),compatibility_changes=changes)


def run(name,compat):
    start=time.perf_counter();result=dict(candidate=name,synthetic_only=True,formal_training_runs=0,optimizer_updates=0,
        ETH_evaluated=False,pretrained_checkpoint_tested=False,attempt='compat' if compat else 'native',cases=[])
    try:
        torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        meta,build,predict,loss,settings=pecnet() if name=='PECNet' else agentformer(compat)
        result.update(repository_commit=meta['commit'],**settings)
        model=build();result['parameter_count']=sum(p.numel() for p in model.parameters())
        for n in [1,2,5,12]:
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();h,y=synthetic_scene(n)
            row=dict(people=n)
            try:
                model.eval();eps=torch.zeros(1,n,settings['latent_dim'])
                with torch.no_grad():
                    a=predict(model,h,eps)
                    p=np.random.default_rng(900+n).permutation(n);inverse=np.argsort(p)
                    if n>1 and np.array_equal(p,np.arange(n)):p=np.roll(p,1);inverse=np.argsort(p)
                    b=predict(model,h[p],eps[:,p])[:,inverse]
                    poison=predict(model,h,eps,np.full_like(y,np.nan))
                    changed=predict(model,h,eps,y+1000)
                    e=torch.randn(20,n,settings['latent_dim'])
                    k20=predict(model,h,e);k20p=predict(model,h[p],e[:,p])[:,inverse]
                row.update(output_shape=list(a.shape),finite_forward=bool(torch.isfinite(a).all() and torch.isfinite(k20).all()),
                    K1_fixed_latent_permutation_max=scalar_error(a,b),K20_coupled_latent_permutation_max=scalar_error(k20,k20p),
                    poisoned_future_max_difference=scalar_error(a,poison),changed_future_max_difference=scalar_error(a,changed))
                model.train();model.zero_grad(set_to_none=True);objective=loss(model,h,y)
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite original objective')
                objective.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
                row.update(original_loss=float(objective.detach()),gradients_finite=bool(grads and all(torch.isfinite(g).all() for g in grads)),
                    gradient_nonzero_scalars=sum(int((g!=0).sum()) for g in grads))
                model.eval();path=CACHE/f'{name}_{result["attempt"]}_N{n}_random_state.pth'
                with path.open('xb') as out:torch.save(model.state_dict(),out)
                restored=build();restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);restored.eval()
                with torch.no_grad():recovered=predict(restored,h,eps)
                row.update(strict_random_checkpoint_restore=True,checkpoint_sha256=sha256(path),checkpoint=str(path),
                    restored_prediction_max_difference=scalar_error(a,recovered),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    elapsed_seconds=time.perf_counter()-tick)
                row['passed']=bool(row['finite_forward'] and row['gradients_finite'] and torch.allclose(a,b,atol=1e-5,rtol=1e-5)
                    and torch.allclose(k20,k20p,atol=1e-5,rtol=1e-5) and torch.equal(a,poison) and torch.equal(a,changed) and torch.equal(a,recovered))
                del restored
            except Exception:row.update(passed=False,traceback=traceback.format_exc())
            result['cases'].append(row);print(name,n,row.get('passed'),row.get('K1_fixed_latent_permutation_max'),flush=True)
        result['all_cases_passed']=all(r['passed'] for r in result['cases'])
    except Exception:result.update(all_cases_passed=False,setup_traceback=traceback.format_exc())
    result.update(elapsed_seconds=time.perf_counter()-start,torch=torch.__version__,GPU=torch.cuda.get_device_name(),
        source_SHA256=sha256(__file__))
    path=ROOT/f'{name}_{result["attempt"]}_admission.json';json_new(path,result)
    print('Admission saved:',path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True,choices=['AgentFormer','PECNet']);parser.add_argument('--compat',action='store_true')
    args=parser.parse_args();run(args.candidate,args.compat)
