"""Limited, frozen EqMotion functional validation. Never executes main()."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,subprocess
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.evaluation.public_baseline import only_function,load_module,non_eth_fold
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4');UPSTREAM=Path('third_party/stage4/eqmotion').resolve();ASSETS=Path('data/external/stage4/assets/eqmotion').resolve()


def make_cases(fold,p,preprocess):
    rec={'hotel':'biwi_hotel','zara1':'crowds_zara01'}[fold];budget=p['eqmotion_neighbor_budget'][fold]
    cfg=dict(dataset=fold,past_frames=8,future_frames=12,frame_skip=1,min_past_frames=8,min_future_frames=12,traj_scale=1.)
    pp=preprocess(str(ASSETS/'eth_ucy/data'),rec,cfg,None,split='test',phase='testing')
    past=[];future=[];valid=[];keys=[]
    for frame in range(pp.init_frame+7,pp.init_frame+pp.num_fr-12):
        d=pp(frame)
        if d is None:continue
        x=torch.stack(d['pre_motion_3D']).numpy();y=torch.stack(d['fut_motion_3D']).numpy();agents=len(x)
        for i in range(agents):
            # Exact target-centred neighbor policy of official processing.
            if agents<budget:
                order=np.array([i]+[j for j in range(agents) if j!=i]);count=agents
            else:
                distance=np.linalg.norm(x[:,-1]-x[i:i+1,-1],axis=-1)
                count=min(int((distance<5).sum()),budget);order=np.argsort(distance)[:count]
                assert order[0]==i
            a=np.zeros((budget,8,2));b=np.zeros((budget,12,2));a[:count]=x[order];b[:count]=y[order]
            past.append(a);future.append(b);valid.append(count)
            keys.append(dict(fold=fold,recording=rec,frame=int(frame),target_id=int(d['valid_id'][i]),valid_agents=count))
            if len(keys)>=p['eqmotion_max_target_samples_per_fold']:break
        if len(keys)>=p['eqmotion_max_target_samples_per_fold']:break
    assert past
    return TensorDataset(torch.tensor(np.asarray(past),dtype=torch.float32),torch.tensor(np.asarray(future),dtype=torch.float32),torch.tensor(valid,dtype=torch.int32)),keys


def forward(model,loc,num_valid):
    vel=torch.zeros_like(loc);vel[:,:,1:]=loc[:,:,1:]-loc[:,:,:-1];vel[:,:,0]=vel[:,:,1]
    nodes=vel.square().sum(-1).sqrt().detach()
    return model(nodes,loc.detach(),vel,num_valid)


def geometry(model,loc,num):
    theta=loc.new_tensor(.731);R=torch.stack([torch.stack([theta.cos(),-theta.sin()]),torch.stack([theta.sin(),theta.cos()])])
    shift=loc.new_tensor([3.,-2.]);reflection=loc.new_tensor([[-1.,0.],[0.,1.]])
    y,categories=forward(model,loc,num);checks=[]
    for name,xx,expected in [('rotation',loc@R.T,y@R.T),('translation',loc+shift,y+shift),('reflection',loc@reflection.T,y@reflection.T)]:
        yy,cc=forward(model,xx,num)
        # Only real agents participate in geometry assertions.
        mask=torch.arange(loc.shape[1],device=loc.device)[None]<num[:,None]
        delta=(yy-expected).norm(dim=-1)[mask]
        category_difference=max(float((a-b).abs().max()) for a,b in zip(cc,categories))
        checks.append(dict(transform=name,valid_agent_prediction_mean_error_m=float(delta.mean()),valid_agent_prediction_max_error_m=float(delta.max()),
            category_max_difference=category_difference,finite=bool(torch.isfinite(yy).all()),only_prefix_input=True,
            changed_samples=int(loc.shape[0])))
    # Reverse real agents within each example, leaving the padded suffix fixed.
    order=torch.arange(loc.shape[1],device=loc.device).expand(loc.shape[0],-1).clone()
    for b,count in enumerate(num.tolist()):order[b,:count]=order[b,:count].flip(0)
    xx=loc.gather(1,order[:,:,None,None].expand_as(loc))
    yy,cc=forward(model,xx,num)
    expected=y.gather(1,order[:,:,None,None,None].expand_as(y))
    category_difference=0.
    for actual,original in zip(cc,categories):
        wanted=original.gather(1,order[:,:,None,None].expand_as(original))
        wanted=wanted.gather(2,order[:,None,:,None].expand_as(wanted))
        category_difference=max(category_difference,float((actual-wanted).abs().max()))
    delta=(yy-expected).norm(dim=-1)[mask]
    checks.append(dict(transform='agent_permutation',valid_agent_prediction_mean_error_m=float(delta.mean()),
        valid_agent_prediction_max_error_m=float(delta.max()),category_max_difference=category_difference,
        finite=bool(torch.isfinite(yy).all()),only_prefix_input=True,changed_samples=int((num>1).sum())))
    return checks


class NoUpdate:
    def zero_grad(self):pass


def main():
    p=json.loads(Path('configs/stage4/protocol.json').read_text());assert not p['training'] and not p['ETH_evaluated']
    if (ROOT/'eqmotion_functional_results.csv').exists():raise FileExistsError('No overwriting run')
    torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17);np.random.seed(17)
    # np.int was an alias for Python int. No numeric transformation changes.
    if 'int' not in np.__dict__:np.int=int
    sys.path.insert(0,str(UPSTREAM))
    from eth_ucy.model_t import EqMotion
    pp=load_module('stage4_eqmotion_preprocessor',UPSTREAM/'eth_ucy/preprocessor.py').preprocess
    namespace=dict(torch=torch,np=np,args=SimpleNamespace(test_scale=1.,n_layers=4),constant=1)
    test,fn_sha=only_function(UPSTREAM/'main_eth_diverse.py','test',namespace)
    metrics=[];targets=[];configurations=[];geometry_rows=[]
    for fold in p['eqmotion_folds']:
        non_eth_fold(fold);t0=time.perf_counter();dataset,keys=make_cases(fold,p,pp);preparation=time.perf_counter()-t0
        loader=DataLoader(dataset,batch_size=p['eqmotion_batch_size'],shuffle=False,num_workers=0)
        checkpoint=ASSETS/'eth_ucy/saved_models'/(fold+'_ckpt.pth.tar')
        model=EqMotion(in_node_nf=8,in_edge_nf=2,hidden_nf=64,in_channel=8,hid_channel=p['eqmotion_channels'][fold],out_channel=12,
            device='cuda',n_layers=4,recurrent=True,norm_diff=False,tanh=False).eval()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);incompat=model.load_state_dict(saved['state_dict'],strict=True)
        outputs=[]
        def hook(module,inputs,output):outputs.append(output[0].detach().cpu().numpy().copy())
        handle=model.register_forward_hook(hook)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():fde,ade=test(model,NoUpdate(),0,loader,backprop=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;handle.remove()
        pred=np.concatenate(outputs)[:,0];truth=dataset.tensors[1][:,0].numpy()
        error=np.linalg.norm(pred-truth[:,None],axis=-1);a=error.mean(-1);f=error[:,:,-1]
        assert abs(a.min(-1).mean()-ade)<1e-6 and abs(f.min(-1).mean()-fde)<1e-6
        for mode,aa,ff in [('K1_head0',a[:,0],f[:,0]),('K20_independent_min',a.min(-1),f.min(-1))]:
            metrics.append(dict(model='EqMotion',fold=fold,scope='first128_target_centered_functional_cases',mode=mode,target_windows=len(keys),
                ADE_m=float(aa.mean()),FDE_m=float(ff.mean()),parameters=sum(v.numel() for v in model.parameters()),evaluation_seconds=seconds,
                preparation_seconds=preparation,peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha256=sha256(checkpoint),
                official_function_numeric_match=True,training=False,ETH_evaluated=False))
            targets.extend(dict(k,case_index=i,mode=mode,ADE_m=float(aa[i]),FDE_m=float(ff[i])) for i,k in enumerate(keys))
        batch=next(iter(loader));loc=batch[0].cuda();num=batch[2].cuda()
        # Padding entries shift together too; only valid agents are checked.
        with torch.no_grad():checks=geometry(model,loc,num)
        geometry_rows.extend(dict(fold=fold,**r) for r in checks)
        configurations.append(dict(fold=fold,neighbor_budget=p['eqmotion_neighbor_budget'][fold],channels=p['eqmotion_channels'][fold],
            total_cases=len(keys),unique_target_ids=len({k['target_id'] for k in keys}),frame_first=keys[0]['frame'],frame_last=keys[-1]['frame'],
            mean_valid_agents=float(dataset.tensors[2].float().mean()),strict_checkpoint_keys=True,checkpoint_sha256=sha256(checkpoint)))
        print(json.dumps(dict(fold=fold,targets=len(keys),K20_ADE=ade,K20_FDE=fde,seconds=seconds,neighbor_budget=p['eqmotion_neighbor_budget'][fold])),flush=True)
        del model;torch.cuda.empty_cache()
    csv_new(ROOT/'eqmotion_functional_results.csv',metrics);csv_new(ROOT/'eqmotion_target_errors.csv',targets);csv_new(ROOT/'eqmotion_geometry_checks.csv',geometry_rows)
    write_json_new(ROOT/'eqmotion_execution.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256(__file__),protocol_sha256=sha256('configs/stage4/protocol.json'),official_test_function_sha256=fn_sha,
        configurations=configurations,upstream_sources={str(f.relative_to(UPSTREAM)):sha256(f) for f in UPSTREAM.rglob('*.py')},
        scope='bounded functional validation, not full EqMotion ETH/UCY performance reproduction',training=False,ETH_evaluated=False,paper_reproduced=False,
        compatibility=['numpy np.int alias=int','explicit test-only preprocess data root instead of hardcoded GPFS path','extract test() only to avoid --test fallthrough training','strict checkpoint key loading']))


if __name__=='__main__':main()
