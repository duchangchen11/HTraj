"""Stage3.5 one recording, two named protocols, fresh seed17 baseline pilot."""
from pathlib import Path
import csv,json,subprocess,time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .smoke import seed_all,move_batch,synchronize
from .social_diagnostics import csv_new
from ..data.benchmark_audit import BenchmarkDataset,assert_recording_isolation
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256,write_json_new
from ..evaluation.benchmark_audit import history_features,target_errors,GROUPS
from ..evaluation.metrics import masked_l2_loss
from ..models.relation_routing import HistoricalCV
from ..models.social_diagnostics_models import DIAGNOSTIC_MODELS

ROOT=Path('results/tables/stage3p5');CONF=Path('configs/stage3p5/protocol.json')


def data_load(protocol):
    indices=json.loads(Path(protocol['pilot_indices']).read_text());manifest=indices['manifest'];assert_recording_isolation(json.loads(Path(manifest).read_text()))
    result={}
    for view in protocol['protocols']:
        train=Subset(BenchmarkDataset(protocol['raw_root'],manifest,'train',view),indices['protocols'][view]['train'])
        val=Subset(BenchmarkDataset(protocol['raw_root'],manifest,'validation',view),indices['protocols'][view]['validation'])
        batches=[]
        for b in DataLoader(val,batch_size=8,collate_fn=collate_trajectories):
            b['groups'],b['history_features']=history_features(b);batches.append(b)
        result[view]=(train,batches)
    return result


@torch.no_grad()
def evaluate(model,batches,device,intervention=None,rows=False):
    model.eval();sums={g:dict(ADE_sum=0.,ADE_count=0,FDE_sum=0.,FDE_count=0,history_frames_sum=0.,motion_frames_sum=0.,neighbor_degree_sum=0.) for g in GROUPS};individual=[]
    for original in batches:
        b=move_batch(original,device);args=[b['observed'],b['observed_valid'],b['timestamps']]
        pred=model(*args,intervention=intervention) if intervention else model(*args)
        if not torch.isfinite(pred).all():raise FloatingPointError('Prediction')
        ade,fde,a,f,count=target_errors(pred,b['future'],b['future_valid']);ade,fde,a,f,count=[t.cpu() for t in (ade,fde,a,f,count)]
        for g,mask in original['groups'].items():
            valid=a&mask;final=f&mask;s=sums[g];s['ADE_sum']+=float(ade[valid].sum());s['ADE_count']+=int(valid.sum());s['FDE_sum']+=float(fde[final].sum());s['FDE_count']+=int(final.sum())
            for key,out in [('history_frames','history_frames_sum'),('continuous_motion_frames','motion_frames_sum'),('mean_neighbor_degree','neighbor_degree_sum')]:s[out]+=float(original['history_features'][key][valid].sum())
        if rows:
            for bi,ni in a.nonzero().tolist():
                fs=original['history_features'];individual.append(dict(sample_id=original['sample_id'][bi],recording=original['recording'][bi],target_id=int(original['agent_ids'][bi,ni]),
                    ADE_m=float(ade[bi,ni]),FDE_m=float(fde[bi,ni]) if f[bi,ni] else None,future_points=int(count[bi,ni]),
                    **{k:float(v[bi,ni]) for k,v in fs.items()},**{g:bool(mask[bi,ni]) for g,mask in original['groups'].items()}))
    result=[dict(group=g,**s,ADE_m=s['ADE_sum']/s['ADE_count'] if s['ADE_count'] else None,FDE_m=s['FDE_sum']/s['FDE_count'] if s['FDE_count'] else None,
        mean_history_frames=s['history_frames_sum']/s['ADE_count'] if s['ADE_count'] else None,mean_motion_frames=s['motion_frames_sum']/s['ADE_count'] if s['ADE_count'] else None,
        mean_degree=s['neighbor_degree_sum']/s['ADE_count'] if s['ADE_count'] else None) for g,s in sums.items()]
    return dict(ADE_m=result[0]['ADE_m'],FDE_m=result[0]['FDE_m'],groups=result,targets=individual)


def save_eval(output,endpoint,r):
    csv_new(output/f'groups_{endpoint}.csv',r['groups']);csv_new(output/f'targets_{endpoint}.csv',r['targets'])


def train_one(name,view,p,train,batches,source):
    output=ROOT/'pilot'/view/name;local=Path('experiments/stage3p5/pilot')/view/name/'checkpoints'
    if output.exists() or local.exists():raise FileExistsError('Do not overwrite '+str(output))
    output.mkdir(parents=True);local.mkdir(parents=True);seed_all(p['seed']);device=torch.device('cuda')
    model=DIAGNOSTIC_MODELS[name](hidden_dim=64).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    loader=DataLoader(train,batch_size=p['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(p['seed']),collate_fn=collate_trajectories,num_workers=0)
    idx=json.loads(Path(p['pilot_indices']).read_text());manifest=idx['manifest'];config=dict(model=name,protocol_name=view,seed=p['seed'],protocol=p,source_commit=source,
        manifest_sha256=sha256(manifest),pilot_indices_sha256=sha256(p['pilot_indices']),protocol_sha256=sha256(CONF),
        source_sha256={str(x):sha256(x) for x in Path('src').rglob('*.py')},training_windows=len(train),validation_windows=sum(len(b['sample_id']) for b in batches),
        parameter_count=sum(v.numel() for v in model.parameters()),weights_initialized_from_scratch=True,ETH_evaluated=False,
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,hardware=torch.cuda.get_device_name(),cpu_threads=1))
    write_json_new(output/'config.json',config);best=None;curves=[];seconds=0.;steps=0;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,p['epochs']+1):
        model.train();synchronize(device);t0=time.perf_counter();loss_sum=0.;targets=0
        for original in loader:
            b=move_batch(original,device);optimizer.zero_grad(set_to_none=True);pred=model(b['observed'],b['observed_valid'],b['timestamps']);loss=masked_l2_loss(pred,b['future'],b['future_valid'])
            if not torch.isfinite(loss):raise FloatingPointError('Loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True);optimizer.step();steps+=1
            n=int(b['future_valid'].any(1).sum());loss_sum+=float(loss)*n;targets+=n
        synchronize(device);dt=time.perf_counter()-t0;seconds+=dt;r=evaluate(model,batches,device)
        row=dict(epoch=epoch,train_loss=loss_sum/targets,validation_ADE_m=r['ADE_m'],validation_FDE_m=r['FDE_m'],training_seconds=dt,training_targets=targets,source_commit=source,seed=17);curves.append(row)
        with (output/'epochs.csv').open('x' if epoch==1 else 'a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n')
            if epoch==1:w.writeheader()
            w.writerow(row)
        if best is None or r['ADE_m']<best['ADE_m']:
            best=r;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps(dict(protocol=view,model=name,epoch=epoch,ADE=r['ADE_m'],FDE=r['FDE_m'],best_epoch=best_epoch,seconds=dt)),flush=True)
    last=r;last_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for endpoint,state,epoch in [('best',best_state,best_epoch),('last',last_state,30)]:
        torch.save(dict(model=state,epoch=epoch,config=config),local/f'{endpoint}.pt');model.load_state_dict(state);ev=evaluate(model,batches,device,rows=True);save_eval(output,endpoint,ev)
    summary=dict(model=name,protocol_name=view,seed=17,best_epoch=best_epoch,best_ADE_m=best['ADE_m'],best_FDE_m=best['FDE_m'],last_epoch=30,last_ADE_m=last['ADE_m'],last_FDE_m=last['FDE_m'],
        parameter_count=config['parameter_count'],training_windows=len(train),validation_windows=config['validation_windows'],optimizer_steps=steps,training_seconds=seconds,total_seconds=time.perf_counter()-start,
        peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,source_commit=source,manifest_sha256=config['manifest_sha256'],protocol_sha256=config['protocol_sha256'],
        best_checkpoint_sha256=sha256(local/'best.pt'),last_checkpoint_sha256=sha256(local/'last.pt'),ETH_evaluated=False,status='completed')
    write_json_new(output/'summary.json',summary);print('COMPLETED '+json.dumps(summary),flush=True)
    del model,optimizer;torch.cuda.empty_cache()


def main():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('Commit source and protocol first')
    import xml.etree.ElementTree as ET
    suites=ET.parse(ROOT/'pretraining_tests.xml').getroot();assert all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures'))
    p=json.loads(CONF.read_text());assert p['epochs']==30 and not p['ETH_evaluated'];torch.set_num_threads(1);source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    data=data_load(p)
    for view,(train,batches) in data.items():
        out=ROOT/'pilot'/view/'CV';out.mkdir(parents=True);r=evaluate(HistoricalCV(),batches,torch.device('cpu'),rows=True);save_eval(out,'analytical',r)
        write_json_new(out/'summary.json',dict(model='CV',protocol_name=view,ADE_m=r['ADE_m'],FDE_m=r['FDE_m'],parameters=0,source_commit=source,
            manifest_sha256=sha256(json.loads(Path(p['pilot_indices']).read_text())['manifest']),ETH_evaluated=False,velocity='last two consecutive observed points, unknown -> explicitly stationary'))
        for name in ('PedBase','Dynamic-GNN-RN'):train_one(name,view,p,train,batches,source)


if __name__=='__main__':main()
