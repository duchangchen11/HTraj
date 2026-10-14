"""Stage2.6 fixed development protocol; independently train every new model."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .smoke import seed_all, move_batch, synchronize
from .social_diagnostics import csv_new, gradient_values
from ..data.dataset import TrajectoryDataset
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256, write_json_new
from ..models.order_verification import build_order_model
from ..models.social_diagnostics_models import DIAGNOSTIC_MODELS
from ..evaluation.metrics import metric_sums, masked_l2_loss
from ..evaluation.order_verification import VERIFICATION_GROUPS, verification_groups, extract_measured_debug, summarize_measured_debug

ROOT = Path('results/tables/stage2p6')
PROTOCOL = Path('configs/stage2p6/protocol.json')


def datasets(protocol):
    subset = json.loads(Path(protocol['subset_manifest']).read_text())
    manifest = json.loads(Path(protocol['data_manifest']).read_text())
    assert subset['data_manifest_sha256'] == sha256(protocol['data_manifest'])
    assert not subset['test_used'] and not protocol['test_evaluation']
    assert len(subset['train']['indices']) == 1024 and len(subset['validation']['indices']) == 96
    assert all(w['scene'] != 'ETH' for s in ('train','validation') for w in manifest['splits'][s])
    train = Subset(TrajectoryDataset(protocol['raw_root'],protocol['data_manifest'],'train'),subset['train']['indices'])
    val = Subset(TrajectoryDataset(protocol['raw_root'],protocol['data_manifest'],'validation'),subset['validation']['indices'])
    return train, val


def cached_validation(data, batch_size):
    batches = []
    for batch in DataLoader(data,batch_size=batch_size,collate_fn=collate_trajectories):
        batch['groups'] = verification_groups(batch)
        batches.append(batch)
    return batches


@torch.no_grad()
def evaluate(model,batches,device,reference=None,full_groups=True):
    model.eval()
    sums = {group:dict(ade_sum=0.,ade_count=0,fde_sum=0.,fde_count=0,observed_targets=0,
                      observed_windows=0,ade_windows=0) for group in (VERIFICATION_GROUPS if full_groups else ['all'])}
    diagnostics, predictions = [], []
    change, change_count = 0., 0
    for index,original in enumerate(batches):
        batch = move_batch(original,device)
        prediction, debug = model(batch['observed'],batch['observed_valid'],batch['timestamps'],return_debug=True)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation')
        diagnostics.append(extract_measured_debug(model,debug))
        predictions.append(prediction.cpu())
        for group,mask in original['groups'].items():
            if group not in sums:
                continue
            valid = batch['future_valid'] & mask.to(device)[:,None]
            for key,value in metric_sums(prediction,batch['future'],valid).items():
                sums[group][key] += value.item()
            sums[group]['observed_targets'] += int(mask.sum())
            sums[group]['observed_windows'] += int(mask.any(-1).sum())
            sums[group]['ade_windows'] += int(valid.any((1,2)).sum())
        if reference is not None:
            delta = (prediction.cpu()-reference[index]).norm(dim=-1)
            mask = original['observed_valid'][:,-1,None].expand_as(delta)
            change += float(delta[mask].sum()); change_count += int(mask.sum())
    groups = [dict(group=name,**v,ADE_m=v['ade_sum']/v['ade_count'] if v['ade_count'] else None,
                   FDE_m=v['fde_sum']/v['fde_count'] if v['fde_count'] else None) for name,v in sums.items()]
    all_values = groups[0]
    assert all_values['ade_count'] == 1202 and all_values['fde_count'] == 839
    summary = summarize_measured_debug(diagnostics)
    if reference is not None:
        summary['prediction_change_L2_m'] = change/max(change_count,1)
    return dict(ADE_m=all_values['ADE_m'],FDE_m=all_values['FDE_m'],groups=groups,
                diagnostics=summary,predictions=predictions)


def groups_from_cached_predictions(predictions,batches):
    """Group the exact selected GPU predictions on CPU; no new model selection."""
    rows=[]
    for group in VERIFICATION_GROUPS:
        sums=dict(ade_sum=0.,ade_count=0,fde_sum=0.,fde_count=0,observed_targets=0,observed_windows=0,ade_windows=0)
        for prediction,batch in zip(predictions,batches):
            mask=batch['groups'][group]
            valid=batch['future_valid'] & mask[:,None]
            for key,value in metric_sums(prediction,batch['future'],valid).items():
                sums[key]+=value.item()
            sums['observed_targets']+=int(mask.sum()); sums['observed_windows']+=int(mask.any(-1).sum())
            sums['ade_windows']+=int(valid.any((1,2)).sum())
        rows.append(dict(group=group,**sums,ADE_m=sums['ade_sum']/sums['ade_count'] if sums['ade_count'] else None,
                         FDE_m=sums['fde_sum']/sums['fde_count'] if sums['fde_count'] else None))
    return rows


def audit_frozen(protocol,batches):
    """Check shared numerical code/hyperparameters and both frozen endpoints."""
    old_source = '90d987799d06bef4e213952342da231e93792a41'
    numerical_paths = subprocess.check_output(['git','ls-tree','-r','--name-only',old_source,'src'],text=True).splitlines()
    hashes = {}
    for path in numerical_paths:
        assert Path(path).read_bytes() == subprocess.check_output(['git','show',old_source+':'+path]),path
        hashes[path] = sha256(path)
    shared = ['held_out_scene','test_evaluation','data_manifest','raw_root','epochs','batch_size','learning_rate',
              'gradient_clip','cpu_threads','hidden_dim','prediction_steps','max_order','max_neighbors','radius',
              'initial_residual_gate','loss','metrics','best_epoch_rule','convergence_rule']
    rows, endpoints = [], []
    for name in protocol['frozen_models']:
        for seed in protocol['seeds']:
            origin = Path('results/tables/stage2p5/round2')/name/f'seed_{seed}'
            config = json.loads((origin/'config.json').read_text())
            summary = json.loads((origin/'summary.json').read_text())
            assert all(config['protocol'][k] == protocol[k] for k in shared),(name,seed,'protocol')
            assert config['source_commit'] == old_source and config['training_windows'] == 1024
            assert config['protocol']['large_subset'] == protocol['subset_manifest']
            assert config['data_manifest_sha256'] == sha256(protocol['data_manifest'])
            assert config['subset_manifest_sha256'] == sha256(protocol['subset_manifest'])
            for endpoint in ('best','last'):
                checkpoint_path = Path('experiments/stage2p5/round2')/name/f'seed_{seed}/checkpoints/{endpoint}.pt'
                assert sha256(checkpoint_path) == summary[endpoint+'_checkpoint_sha256']
                checkpoint = torch.load(checkpoint_path,weights_only=True,map_location='cpu')
                model = DIAGNOSTIC_MODELS[name](hidden_dim=64).eval()
                model.load_state_dict(checkpoint['model'])
                result = evaluate(model,batches,torch.device('cpu'))
                assert abs(result['ADE_m']-summary[endpoint+'_ADE_m']) < 1e-6
                assert abs(result['FDE_m']-summary[endpoint+'_FDE_m']) < 1e-6
                output = ROOT/'frozen'/name/f'seed_{seed}'
                output.mkdir(parents=True,exist_ok=True)
                csv_new(output/f'groups_{endpoint}.csv',result['groups'])
                endpoints.append(dict(model=name,seed=seed,checkpoint=endpoint,**result['diagnostics'],
                    reevaluated_ADE_m=result['ADE_m'],reevaluated_FDE_m=result['FDE_m'],inference_device='cpu',
                    source_commit=old_source,checkpoint_sha256=sha256(checkpoint_path)))
            rows.append(dict(**summary,trained_in_stage2p6=False,origin_summary=str(origin/'summary.json'),
                             origin_epochs=str(origin/'epochs.csv'),origin_config=str(origin/'config.json')))
    csv_new(ROOT/'frozen_baselines.csv',rows)
    csv_new(ROOT/'frozen_endpoint_diagnostics.csv',endpoints)
    write_json_new(ROOT/'frozen_baseline_audit.json',dict(status='passed',model_source_commit=old_source,
        shared_protocol_fields=shared,numerical_file_sha256=hashes,metrics_tolerance_m=1e-6,
        validation_windows=96,best_and_last_checked=True,test_evaluated=False))


def train_one(name,seed,protocol,train,batches,source_commit,device):
    output = ROOT/'new'/name/f'seed_{seed}'
    local = Path('experiments/stage2p6/new')/name/f'seed_{seed}/checkpoints'
    if output.exists() or local.exists():
        raise FileExistsError('Never overwrite a run: '+str(output))
    output.mkdir(parents=True); local.mkdir(parents=True)
    seed_all(seed)
    model = build_order_model(name,hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    loader = DataLoader(train,batch_size=protocol['batch_size'],shuffle=True,
        generator=torch.Generator().manual_seed(seed),collate_fn=collate_trajectories,num_workers=0)
    config = dict(model=name,seed=seed,protocol=protocol,source_commit=source_commit,
        data_manifest_sha256=sha256(protocol['data_manifest']),subset_manifest_sha256=sha256(protocol['subset_manifest']),
        protocol_sha256=sha256(PROTOCOL),test_evaluated=False,tracked_source_dirty_at_start=False,
        parameter_count=sum(p.numel() for p in model.parameters()),hidden_dim=64,max_order=model.order,
        time_mode=model.time_mode,delta_width=model.delta_width,training_windows=1024,validation_windows=96,
        started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,hardware=torch.cuda.get_device_name(),
                         cpu_threads=torch.get_num_threads(),scatter_bitwise_determinism=False))
    write_json_new(output/'config.json',config)
    torch.cuda.reset_peak_memory_stats(device)
    best, best_state, best_epoch, steps = None,None,None,0
    rows=[]; training_seconds=0.; start=time.perf_counter()
    for epoch in range(1,protocol['epochs']+1):
        model.train(); synchronize(device); epoch_start=time.perf_counter()
        loss_sum,count=0.,0; gradients=[]
        for original in loader:
            batch=move_batch(original,device)
            if not batch['future_valid'].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['observed'],batch['observed_valid'],batch['timestamps'])
            loss=masked_l2_loss(prediction,batch['future'],batch['future_valid'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward(); gradients.append(gradient_values(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            targets=int(batch['future_valid'].any(1).sum())
            loss_sum+=float(loss)*targets; count+=targets; steps+=1
        synchronize(device); duration=time.perf_counter()-epoch_start; training_seconds+=duration
        validation=evaluate(model,batches,device,full_groups=False)
        grads={k:float(np.mean([r[k] for r in gradients if k in r])) for k in sorted(set().union(*(r.keys() for r in gradients)))}
        row=dict(epoch=epoch,train_loss=loss_sum/max(count,1),validation_ADE_m=validation['ADE_m'],
            validation_FDE_m=validation['FDE_m'],training_seconds=duration,source_commit=source_commit,seed=seed,
            **validation['diagnostics'],**grads)
        rows.append(row)
        path=output/'epochs.csv'
        if epoch==1:
            with path.open('x',newline='') as f:
                csv.DictWriter(f,fieldnames=list(row),lineterminator='\n').writeheader()
        with path.open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerow(row)
        if best is None or validation['ADE_m'] < best['ADE_m']:
            best,best_epoch=validation,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps(dict(model=name,seed=seed,epoch=epoch,loss=row['train_loss'],ADE=validation['ADE_m'],
                              FDE=validation['FDE_m'],best_epoch=best_epoch,training_seconds=duration)),flush=True)
    last=validation
    last_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for endpoint,state,saved_epoch in [('best',best_state,best_epoch),('last',last_state,30)]:
        torch.save(dict(model=state,epoch=saved_epoch,config=config),local/f'{endpoint}.pt')
    for endpoint,result in [('best',best),('last',last)]:
        csv_new(output/f'groups_{endpoint}.csv',groups_from_cached_predictions(result['predictions'],batches))
    tail=[r['train_loss'] for r in rows[-5:]]
    decrease=(tail[0]-tail[-1])/max(abs(tail[0]),1e-12)
    summary=dict(model=name,seed=seed,status='completed',trained_in_stage2p6=True,parameter_count=config['parameter_count'],
        best_epoch=best_epoch,best_ADE_m=best['ADE_m'],best_FDE_m=best['FDE_m'],last_epoch=30,
        last_ADE_m=last['ADE_m'],last_FDE_m=last['FDE_m'],optimizer_steps=steps,
        first_epoch_train_loss=rows[0]['train_loss'],last_epoch_train_loss=rows[-1]['train_loss'],
        tail5_relative_train_loss_decrease=decrease,convergence_flag=decrease<.02,
        actual_training_seconds=training_seconds,total_run_seconds=time.perf_counter()-start,
        cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        source_commit=source_commit,data_manifest_sha256=config['data_manifest_sha256'],
        subset_manifest_sha256=config['subset_manifest_sha256'],protocol_sha256=config['protocol_sha256'],
        test_evaluated=False,training_windows=1024,validation_windows=96,
        best_checkpoint_sha256=sha256(local/'best.pt'),last_checkpoint_sha256=sha256(local/'last.pt'))
    write_json_new(output/'summary.json',summary)
    print('COMPLETED '+json.dumps(summary),flush=True)
    del model,optimizer,prediction,loss,batch
    torch.cuda.empty_cache()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('Commit tracked source/config before any run')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['epochs']==30 and not protocol['test_evaluation'] and protocol['held_out_scene']=='ETH'
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    train,val=datasets(protocol); batches=cached_validation(val,8)
    audit_path=ROOT/'frozen_baseline_audit.json'
    if not audit_path.exists():
        audit_frozen(protocol,batches)
    else:
        assert json.loads(audit_path.read_text())['status']=='passed'
    if args.audit_only:
        return
    # The pretraining gate refers to actual pytest output, not a user assertion.
    import xml.etree.ElementTree as ET
    tests=ET.parse(ROOT/'pretraining_tests.xml').getroot()
    assert all(int(s.attrib.get(k,0))==0 for s in tests for k in ('failures','errors'))
    assert sum(int(s.attrib['tests']) for s in tests)>=103
    device=torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    summaries=[]
    for name in protocol['new_models']:
        for seed in protocol['seeds']:
            summaries.append(train_one(name,seed,protocol,train,batches,source,device))
    csv_new(ROOT/'new_summary.csv',summaries)


if __name__=='__main__':
    main()
