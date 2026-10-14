"""Stage3 preregistered bounded training; separate archived frozen references."""
from pathlib import Path
import argparse,csv,json,subprocess,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from .smoke import seed_all,move_batch,synchronize
from .social_diagnostics import csv_new
from .order_verification import datasets
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256,write_json_new
from ..models.relation_routing import build_routing_model,HistoricalCV,SPECS
from ..models.social_diagnostics_models import DIAGNOSTIC_MODELS
from ..evaluation.metrics import masked_l2_loss
from ..evaluation.relation_routing import cache_validation,evaluate

ROOT=Path('results/tables/stage3');PROTOCOL=Path('configs/stage3/protocol.json')


def gradients(model):
    sums={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise FloatingPointError(name)
            group='projection' if 'projections' in name else 'relation' if name.startswith('graph.') else name.split('.')[0]
            sums[group]=sums.get(group,0.)+float(p.grad.detach().square().sum())
    return {k+'_gradient_L2':v**.5 for k,v in sums.items()}


def audit_frozen(p,batches):
    source='90d987799d06bef4e213952342da231e93792a41'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',source,'src'],text=True).splitlines();hashes={}
    for path in paths:
        assert Path(path).read_bytes()==subprocess.check_output(['git','show',source+':'+path]),path
        hashes[path]=sha256(path)
    shared=['held_out_scene','test_evaluation','data_manifest','raw_root','epochs','batch_size','learning_rate','gradient_clip',
            'cpu_threads','hidden_dim','prediction_steps','initial_residual_gate','loss','metrics','best_epoch_rule']
    summaries=[]
    for name in p['frozen_models']:
        for seed in (17,42,2026):
            origin=Path('results/tables/stage2p5/round2')/name/f'seed_{seed}'
            config=json.loads((origin/'config.json').read_text());s=json.loads((origin/'summary.json').read_text())
            assert all(config['protocol'][k]==p[k] for k in shared)
            assert config['source_commit']==source and config['protocol']['large_subset']==p['subset_manifest']
            assert config['data_manifest_sha256']==sha256(p['data_manifest']) and config['subset_manifest_sha256']==sha256(p['subset_manifest'])
            out=ROOT/'frozen'/name/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            for endpoint in ('best','last'):
                path=Path('experiments/stage2p5/round2')/name/f'seed_{seed}/checkpoints/{endpoint}.pt'
                assert sha256(path)==s[endpoint+'_checkpoint_sha256']
                model=DIAGNOSTIC_MODELS[name](hidden_dim=64);model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model'])
                result=evaluate(model,batches,torch.device('cpu'))
                assert abs(result['ADE_m']-s[endpoint+'_ADE_m'])<1e-6 and abs(result['FDE_m']-s[endpoint+'_FDE_m'])<1e-6
                csv_new(out/f'groups_{endpoint}.csv',result['groups'])
                write_json_new(out/f'diagnostics_{endpoint}.json',dict(**result['diagnostics'],reevaluated_ADE_m=result['ADE_m'],reevaluated_FDE_m=result['FDE_m'],checkpoint_sha256=sha256(path),source_commit=source))
            summaries.append(dict(**s,trained_in_stage3=False,origin_config=str(origin/'config.json'),origin_epochs=str(origin/'epochs.csv'),origin_summary=str(origin/'summary.json')))
    csv_new(ROOT/'frozen_baselines.csv',summaries)
    write_json_new(ROOT/'frozen_baseline_audit.json',dict(status='passed',source_commit=source,numerical_source_sha256=hashes,shared_fields=shared,
        unused_field_difference='max_order: Stage3 direct models1 versus old protocol3; PedBase ignores graph/order kwargs, DynamicRN constructs one channel independently of max_order',test_evaluated=False))
    cv=evaluate(HistoricalCV(),batches,torch.device('cpu'));out=ROOT/'CV';out.mkdir()
    csv_new(out/'groups.csv',cv['groups']);write_json_new(out/'summary.json',dict(model='CV',parameter_count=0,ADE_m=cv['ADE_m'],FDE_m=cv['FDE_m'],velocity='last consecutive two observed positions; unknown -> stationary explicitly marked fallback',test_evaluated=False))


def train_one(name,seed,p,train,batches,source,device):
    output=ROOT/'new'/name/f'seed_{seed}';local=Path('experiments/stage3/new')/name/f'seed_{seed}/checkpoints'
    if output.exists() or local.exists():raise FileExistsError('Never overwrite run '+str(output))
    output.mkdir(parents=True);local.mkdir(parents=True);seed_all(seed)
    model=build_routing_model(name).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    loader=DataLoader(train,batch_size=p['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate_trajectories,num_workers=0)
    dead=sum(v.numel() for k,v in model.named_parameters() if k.startswith('fusion.scores.') or (name=='Routing-Uniform' and k.startswith('graph.relation_score.')))
    config=dict(model=name,seed=seed,protocol=p,source_commit=source,source_sha256={str(x):sha256(x) for x in Path('src').rglob('*.py')},
        data_manifest_sha256=sha256(p['data_manifest']),subset_manifest_sha256=sha256(p['subset_manifest']),protocol_sha256=sha256(PROTOCOL),
        parameter_count=sum(v.numel() for v in model.parameters()),structurally_dead_parameters=dead,hidden_dim=64,training_windows=1024,validation_windows=96,
        started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),test_evaluated=False,tracked_source_dirty_at_start=False,
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,hardware=torch.cuda.get_device_name(),cpu_threads=1,scatter_bitwise_determinism=False))
    write_json_new(output/'config.json',config);rows=[];best=None;best_state=None;steps=0;training_seconds=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,p['epochs']+1):
        model.train();synchronize(device);epoch_start=time.perf_counter();loss_sum=0.;count=0;grads=[]
        for original in loader:
            batch=move_batch(original,device)
            if not batch['future_valid'].any():continue
            optimizer.zero_grad(set_to_none=True);prediction=model(batch['observed'],batch['observed_valid'],batch['timestamps'])
            loss=masked_l2_loss(prediction,batch['future'],batch['future_valid'])
            if not torch.isfinite(loss):raise FloatingPointError('Loss')
            loss.backward();grads.append(gradients(model));torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            targets=int(batch['future_valid'].any(1).sum());loss_sum+=float(loss)*targets;count+=targets;steps+=1
        synchronize(device);duration=time.perf_counter()-epoch_start;training_seconds+=duration
        result=evaluate(model,batches,device,full_groups=False)
        row=dict(epoch=epoch,train_loss=loss_sum/max(count,1),validation_ADE_m=result['ADE_m'],validation_FDE_m=result['FDE_m'],training_seconds=duration,
                 seed=seed,source_commit=source,**result['diagnostics'],**{k:float(np.mean([g[k] for g in grads if k in g])) for k in sorted(set().union(*(g.keys() for g in grads)))})
        rows.append(row)
        with (output/'epochs.csv').open('x' if epoch==1 else 'a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
            if epoch==1:writer.writeheader()
            writer.writerow(row)
        if best is None or result['ADE_m']<best['ADE_m']:
            best=result;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps(dict(model=name,seed=seed,epoch=epoch,ADE=result['ADE_m'],FDE=result['FDE_m'],best_epoch=best_epoch,training_seconds=duration)),flush=True)
    last_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    endpoints={}
    for endpoint,state,epoch in [('best',best_state,best_epoch),('last',last_state,30)]:
        torch.save(dict(model=state,epoch=epoch,config=config),local/f'{endpoint}.pt');model.load_state_dict(state)
        evaluated=evaluate(model,batches,device);csv_new(output/f'groups_{endpoint}.csv',evaluated['groups']);write_json_new(output/f'diagnostics_{endpoint}.json',evaluated['diagnostics']);endpoints[endpoint]=evaluated
    summary=dict(model=name,seed=seed,status='completed',trained_in_stage3=True,parameter_count=config['parameter_count'],structurally_dead_parameters=dead,
        best_epoch=best_epoch,best_ADE_m=best['ADE_m'],best_FDE_m=best['FDE_m'],last_epoch=30,last_ADE_m=result['ADE_m'],last_FDE_m=result['FDE_m'],
        optimizer_steps=steps,actual_training_seconds=training_seconds,total_run_seconds=time.perf_counter()-start,cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        source_commit=source,data_manifest_sha256=config['data_manifest_sha256'],subset_manifest_sha256=config['subset_manifest_sha256'],protocol_sha256=config['protocol_sha256'],
        best_checkpoint_sha256=sha256(local/'best.pt'),last_checkpoint_sha256=sha256(local/'last.pt'),test_evaluated=False,training_windows=1024,validation_windows=96)
    write_json_new(output/'summary.json',summary);print('COMPLETED '+json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+');parser.add_argument('--seeds',nargs='+',type=int,default=[17]);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('Commit tracked source/config before run')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();p=json.loads(PROTOCOL.read_text());assert not p['test_evaluation'] and p['epochs']==30
    torch.set_num_threads(1);train,val=datasets(p);batches=cache_validation(val)
    if not (ROOT/'frozen_baseline_audit.json').exists():audit_frozen(p,batches)
    if args.audit_only:return
    import xml.etree.ElementTree as ET
    suites=ET.parse(ROOT/'pretraining_tests.xml').getroot();assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors'))
    assert sum(int(s.get('tests',0)) for s in suites)>=140
    if args.seeds!=[17]:
        selection=json.loads((ROOT/'round2_selection.json').read_text());assert all(n in selection['selected_models'] for n in args.models)
    for name in args.models or p['new_models']:
        assert name in SPECS
        for seed in args.seeds:train_one(name,seed,p,train,batches,source,torch.device('cuda'))


if __name__=='__main__':main()
