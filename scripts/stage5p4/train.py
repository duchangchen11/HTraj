"""EgoCapacity: frozen original VAE objective and fresh target-only residual capacity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import platform
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
from src.socialvae_causal.data import CausalDataset
from src.socialvae_causal.training import seed_all,train_step,timed_batch,save_checkpoint_new
from src.socialvae_causal.io import csv_new
from src.socialvae_capacity.model import new_model
from src.socialvae_capacity.training import verify_checkpoint,GradientTracker
from src.socialvae_baseline.evaluation import evaluate,evaluate_train_K1
from src.socialvae_baseline.artifacts import json_new,state_info
from src.data.preprocessing import sha256


def run(recording):
    plan=json.load(open('configs/stage5p4/ego_protocol.json'))
    assert recording in plan['manifests']
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==plan['branch']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(), 'Commit frozen code before training'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=plan['manifests'][recording]
    assert sha256(manifest['path'])==manifest['sha256']
    directory=Path('results/tables/stage5p4/ego_capacity')/recording;directory.mkdir(parents=True)
    cpdir=Path('experiments/stage5p4/checkpoints/ego_capacity')/recording;cpdir.mkdir(parents=True)
    started=time.perf_counter();epoch=0;batch=None;model=None;optimizer=None
    log=(directory/'execution.jsonl').open('x',buffering=1)
    def event(kind,**values):
        message=dict(event=kind,recording=recording,**values)
        text=json.dumps(message,ensure_ascii=False,allow_nan=False)
        log.write(text+'\n');print(text,flush=True)
    try:
        seed_all(plan['seed'])
        train=CausalDataset(plan['data_root'],manifest['path'],'train')
        val=CausalDataset(plan['data_root'],manifest['path'],'validation')
        assert recording not in train.records and set(val.records)=={recording}
        model=new_model();optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']);tracker=GradientTracker(model)
        initial=state_info(model)
        provenance=dict(code_commit=commit,plan_sha256=sha256('configs/stage5p4/ego_protocol.json'),manifest=manifest,
            seed=plan['seed'],fresh_random_initialization=True,checkpoint_loaded_before_training=False,
            initial_parameter_state=initial,training_targets=len(train),validation_targets=len(val),
            steps_per_epoch=math.ceil(len(train)/plan['batch_size']),train_recordings=sorted(train.records),
            validation_recordings=sorted(val.records),source_files=train.manifest['source_files'],
            python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            GPU=torch.cuda.get_device_name(),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            source_hashes={p:sha256(p) for p in ['src/socialvae_causal/training.py','src/socialvae_causal/data.py',
                'scripts/stage5p4/train.py','src/socialvae_baseline/evaluation.py','src/socialvae_capacity/model.py','src/socialvae_capacity/training.py']})
        json_new(directory/'provenance.json',provenance);event('fresh_initialization',**initial,code_commit=commit)
        curves=[];inventory=[];best=None;restore_checks=[]
        for epoch in range(1,plan['epochs']+1):
            begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            order=np.random.default_rng(plan['seed']+epoch).permutation(len(train))
            assert np.array_equal(np.sort(order),np.arange(len(train)))
            steps=[];train_seconds=data_seconds=0.;seen=0
            for step,start in enumerate(range(0,len(order),plan['batch_size']),1):
                indices=order[start:start+plan['batch_size']]
                batch,dt=timed_batch(train,indices);data_seconds+=dt
                tick=time.perf_counter();loss=train_step(model,optimizer,batch);tracker.update(model);torch.cuda.synchronize()
                duration=time.perf_counter()-tick;train_seconds+=duration
                if not all(math.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite training diagnostic')
                steps.append(dict(epoch=epoch,step=step,targets=len(indices),**loss,compute_seconds=duration,data_seconds=dt))
                seen+=len(indices)
            assert seen==len(train) and len(steps)==math.ceil(len(train)/128)
            summary,rows=evaluate(model,val,plan['batch_size'])
            extra=evaluate_train_K1(model,train) if epoch in plan['evaluation']['train_K1_epochs'] else dict(train_K1_prior_ADE_m=None,train_K1_prior_FDE_m=None)
            parameters=state_info(model)
            gradient_audit=tracker.snapshot(model,epoch)
            csv_new(directory/f'epochs/epoch{epoch:03d}_parameter_gradients.csv.gz',gradient_audit)
            gradients=[s['gradient_L2'] for s in steps]
            curve=dict(epoch=epoch,targets_seen=seen,steps=len(steps),
                train_loss=sum(s['targets']*s['loss'] for s in steps)/seen,
                train_rec=sum(s['targets']*s['rec'] for s in steps)/seen,
                train_kl=sum(s['targets']*s['kl'] for s in steps)/seen,
                gradient_L2_mean=float(np.mean(gradients)),gradient_L2_P95=float(np.percentile(gradients,95)),gradient_L2_max=max(gradients),
                **summary,**extra,**parameters,train_compute_seconds=train_seconds,data_seconds=data_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                free_disk_bytes=shutil.disk_usage('.').free)
            is_best=best is None or summary['K1_prior_ADE_m']<best['K1_prior_ADE_m']
            save_now=is_best or epoch==1 or epoch%10==0 or epoch==plan['epochs']
            if save_now:
                path=cpdir/f'epoch{epoch:03d}.pth'
                save_checkpoint_new(path,model,optimizer,epoch=epoch,recording=recording,code_commit=commit,
                    protocol=plan['protocol'],manifest_sha256=manifest['sha256'],seed=plan['seed'],metrics=curve)
                checksum=sha256(path)
                if epoch==1 or epoch%10==0:
                    restored=verify_checkpoint(path,model,optimizer,batch)
                    restore_checks.append(dict(epoch=epoch,checkpoint_sha256=checksum,**restored))
                inventory.append(dict(epoch=epoch,path=str(path),sha256=checksum,bytes=path.stat().st_size,
                    K1_prior_ADE_m=summary['K1_prior_ADE_m'],K1_prior_FDE_m=summary['K1_prior_FDE_m']))
                if is_best:best=inventory[-1]
            curve['epoch_wall_seconds']=time.perf_counter()-begin
            csv_new(directory/f'epochs/epoch{epoch:03d}_targets.csv.gz',rows)
            csv_new(directory/f'epochs/epoch{epoch:03d}_steps.csv.gz',steps)
            json_new(directory/f'epochs/epoch{epoch:03d}.json',curve)
            curves.append(curve)
            event('epoch',epoch=epoch,train_loss=curve['train_loss'],train_rec=curve['train_rec'],train_kl=curve['train_kl'],
                K1_ADE=summary['K1_prior_ADE_m'],K1_FDE=summary['K1_prior_FDE_m'],
                minADE20=summary['minADE20_m'],minFDE20=summary['minFDE20_m'],
                epoch_wall_seconds=curve['epoch_wall_seconds'],gradient_L2_max=curve['gradient_L2_max'],
                peak_allocated_bytes=curve['peak_allocated_bytes'],free_disk_bytes=curve['free_disk_bytes'])
        csv_new(directory/'lifetime_parameter_gradients.csv',gradient_audit)
        csv_new(directory/'training_curve.csv',curves)
        csv_new(directory/'checkpoint_inventory.csv',inventory)
        json_new(directory/'checkpoint_restore_checks.json',restore_checks)
        # Selected checkpoint is frozen by the stated ADE rule before final analysis.
        selected=torch.load(best['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(selected['model'],strict=True);optimizer.load_state_dict(selected['optimizer'])
        selected_check=verify_checkpoint(best['path'],model,optimizer,batch)
        best_summary,best_rows=evaluate(model,val)
        best_train=evaluate_train_K1(model,train)
        csv_new(directory/'best_paired_targets.csv.gz',best_rows)
        json_new(directory/'selection.json',dict(best=best,last=inventory[-1],best_validation=best_summary,
            best_train_K1=best_train,selected_best_restore=selected_check,
            completed_epochs=len(curves),training_targets_per_epoch=len(train),total_optimizer_steps=sum(c['steps'] for c in curves),
            actual_run_wall_seconds=time.perf_counter()-started,initial_parameter_state=initial,
            code_commit=commit,selection_rule=plan['model_selection'],training_completed=True))
        event('completed',best_epoch=best['epoch'],best_ADE=best['K1_prior_ADE_m'],best_FDE=best['K1_prior_FDE_m'],
            actual_run_wall_seconds=time.perf_counter()-started)
    except Exception:
        details=traceback.format_exc()
        if model is not None and optimizer is not None:
            save_checkpoint_new(cpdir/f'failure_epoch{epoch:03d}.pth',model,optimizer,epoch=epoch,traceback=details,
                batch=batch,code_commit=commit)
        json_new(directory/'failure.json',dict(epoch=epoch,traceback=details,training_completed=False,actual_run_wall_seconds=time.perf_counter()-started))
        log.write(json.dumps(dict(event='failed',epoch=epoch,traceback=details))+'\n')
        raise
    finally:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recording',required=True,choices=['crowds_zara02','biwi_hotel'])
    run(parser.parse_args().recording)
