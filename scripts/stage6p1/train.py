"""Four frozen sampling arms; unchanged Stage5.3 network, loss and evaluator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,math,platform,shutil,subprocess,time,traceback
import numpy as np
import torch
from src.data.preprocessing import sha256
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.io import csv_new,read_csv
from src.socialvae_causal.training import seed_all,train_step,timed_batch,save_checkpoint_new
from src.socialvae_attribution.model import new_model
from src.socialvae_attribution.training import verify_checkpoint,GradientTracker
from src.socialvae_stability.training import checkpoint_bytes,write_checkpoint_bytes_new
from src.socialvae_baseline.evaluation import evaluate,evaluate_train_K1
from src.socialvae_baseline.artifacts import json_new,state_info
from src.motion_sampling.policy import epoch_order,exposure

ROOT=Path('results/tables/stage6p1')


def run(recording,condition):
    plan=json.load(open('configs/stage6p1/protocol.json'))
    assert any(r['recording']==recording and r['condition']==condition for r in plan['formal_runs'])
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==plan['branch']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no']), 'Commit source/configuration first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=plan['manifests'][recording];assert sha256(manifest['path'])==manifest['sha256']
    directory=ROOT/'runs'/recording/condition;directory.mkdir(parents=True)
    cpdir=Path('experiments/stage6p1/checkpoints')/recording/condition;cpdir.mkdir(parents=True)
    started=time.perf_counter();epoch=0;batch=None;model=None;optimizer=None;seen=0
    log=(directory/'execution.jsonl').open('x',buffering=1)
    def event(kind,**values):
        text=json.dumps(dict(event=kind,recording=recording,condition=condition,**values),allow_nan=False)
        log.write(text+'\n');print(text,flush=True)
    try:
        seed_all(plan['seed'])
        train=CausalDataset(plan['data_root'],manifest['path'],'train')
        val=CausalDataset(plan['data_root'],manifest['path'],'validation')
        assert recording not in train.records and set(val.records)=={recording}
        pool=read_csv(ROOT/f'pools/{recording}_train.csv.gz')
        labels=np.array([int(r['category_index']) for r in pool]);assert len(labels)==len(train)
        model=new_model();optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
        tracker=GradientTracker(model);initial=state_info(model)
        expected=json.load(open(ROOT/'initialization_check.json'))
        assert initial['parameter_state_sha256']==expected['parameter_state_sha256']
        assert initial['parameter_count']==expected['parameter_count']
        provenance=dict(condition=condition,code_commit=commit,protocol_sha256=sha256('configs/stage6p1/protocol.json'),
            manifest=manifest,seed=17,fresh_random_initialization=True,checkpoint_loaded_before_training=False,
            initial_parameter_state=initial,training_targets=len(train),validation_targets=len(val),
            steps_per_epoch=math.ceil(len(train)/128),train_recordings=sorted(train.records),validation_recordings=sorted(val.records),
            source_files=train.manifest['source_files'],pool_sha256=sha256(ROOT/f'pools/{recording}_train.csv.gz'),
            model_effective_parameter_upper_bound=expected['structurally_effective_scalar_upper_bound'],
            python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),
            GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            source_hashes={str(p):sha256(p) for p in [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
                Path('src/motion_sampling/policy.py'),Path('src/socialvae_attribution/model.py'),Path('src/socialvae_attribution/training.py'),
                Path('src/socialvae_causal/data.py'),Path('src/socialvae_causal/training.py'),Path('src/socialvae_baseline/evaluation.py')]})
        json_new(directory/'run_config.json',dict(**plan,condition=condition,validation_recording=recording))
        json_new(directory/'provenance.json',provenance);event('fresh_initialization',**initial,code_commit=commit)
        curves=[];inventory=[];best=None;best_payload=None;restore_checks=[];sampling_rows=[];cumulative=np.zeros(len(train),dtype=np.int64)
        for epoch in range(1,31):
            begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            order=epoch_order(labels,condition,epoch,17);counts,classes=exposure(labels,order)
            if condition=='Uniform':assert (counts==1).all()
            steps=[];seen=0;train_seconds=data_seconds=0.
            for step,start in enumerate(range(0,len(order),128),1):
                indices=order[start:start+128]
                batch,dt=timed_batch(train,indices);data_seconds+=dt
                tick=time.perf_counter();loss=train_step(model,optimizer,batch);tracker.update(model);torch.cuda.synchronize()
                duration=time.perf_counter()-tick;train_seconds+=duration
                if not all(math.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite diagnostic')
                steps.append(dict(epoch=epoch,step=step,targets=len(indices),**loss,compute_seconds=duration,data_seconds=dt))
                seen+=len(indices)
            assert seen==len(train) and len(steps)==math.ceil(len(train)/128)
            cumulative+=counts
            visits=[dict(target_index=i,visits=int(n),cumulative_visits=int(cumulative[i])) for i,n in enumerate(counts)]
            csv_new(directory/f'epochs/epoch{epoch:03d}_target_visits.csv.gz',visits)
            for row in classes:
                row.update(epoch=epoch,total_visits=seen,order_sha256=array_sha(order),actual_full_order_consumed=True,
                    cumulative_unique_total=int((cumulative>0).sum()),cumulative_coverage_total=float((cumulative>0).mean()))
            csv_new(directory/f'epochs/epoch{epoch:03d}_sampling.csv',classes);sampling_rows+=classes
            summary,rows=evaluate(model,val,128)
            extra=evaluate_train_K1(model,train) if epoch in plan['train_K1_epochs'] else dict(train_K1_prior_ADE_m=None,train_K1_prior_FDE_m=None)
            gradient_audit=tracker.snapshot(model,epoch)
            csv_new(directory/f'epochs/epoch{epoch:03d}_parameter_gradients.csv.gz',gradient_audit)
            gradients=[s['gradient_L2'] for s in steps]
            curve=dict(epoch=epoch,targets_seen=seen,steps=len(steps),
                train_loss=sum(s['targets']*s['loss'] for s in steps)/seen,train_rec=sum(s['targets']*s['rec'] for s in steps)/seen,
                train_kl=sum(s['targets']*s['kl'] for s in steps)/seen,gradient_L2_mean=float(np.mean(gradients)),
                gradient_L2_P95=float(np.percentile(gradients,95)),gradient_L2_max=max(gradients),**summary,**extra,**state_info(model),
                train_compute_seconds=train_seconds,data_seconds=data_seconds,unique_targets=int((counts>0).sum()),
                repeated_visits=int((counts-np.minimum(counts,1)).sum()),repeated_target_windows=int((counts>1).sum()),
                coverage=float((counts>0).mean()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_disk_bytes=shutil.disk_usage('.').free)
            info=dict(epoch=epoch,path=None,sha256=None,bytes=None,
                K1_prior_ADE_m=summary['K1_prior_ADE_m'],K1_prior_FDE_m=summary['K1_prior_FDE_m'])
            if best is None or info['K1_prior_ADE_m']<best['K1_prior_ADE_m']:
                best=info
                best_payload=checkpoint_bytes(model,optimizer,epoch=epoch,recording=recording,condition=condition,
                    code_commit=commit,protocol=plan['protocol'],manifest_sha256=manifest['sha256'],seed=17,metrics=curve)
            if epoch in plan['checkpoint_restore_epochs']:
                path=cpdir/f'epoch{epoch:03d}.pth'
                save_checkpoint_new(path,model,optimizer,epoch=epoch,recording=recording,condition=condition,
                    code_commit=commit,protocol=plan['protocol'],manifest_sha256=manifest['sha256'],seed=17,metrics=curve)
                checksum=sha256(path)
                restored=verify_checkpoint(path,model,optimizer,batch)
                restore_checks.append(dict(epoch=epoch,checkpoint_sha256=checksum,**restored))
                inventory.append(dict(info,path=str(path),sha256=checksum,bytes=path.stat().st_size))
            curve['epoch_wall_seconds']=time.perf_counter()-begin
            csv_new(directory/f'epochs/epoch{epoch:03d}_targets.csv.gz',rows)
            csv_new(directory/f'epochs/epoch{epoch:03d}_steps.csv.gz',steps)
            json_new(directory/f'epochs/epoch{epoch:03d}.json',curve);curves.append(curve)
            event('epoch',epoch=epoch,train_loss=curve['train_loss'],K1_ADE=summary['K1_prior_ADE_m'],K1_FDE=summary['K1_prior_FDE_m'],
                minADE20=summary['minADE20_m'],minFDE20=summary['minFDE20_m'],epoch_wall_seconds=curve['epoch_wall_seconds'],
                unique_targets=curve['unique_targets'],coverage=curve['coverage'],gradient_L2_max=curve['gradient_L2_max'],
                peak_allocated_bytes=curve['peak_allocated_bytes'])
            if epoch==5:
                json_new(directory/'smoke_5epoch.json',dict(passed=True,epochs_are_part_of_30=True,extra_epochs=0,
                    all_finite=True,strict_restore_passed_at=[c['epoch'] for c in restore_checks],optimizer_steps=sum(c['steps'] for c in curves)))
        existing=next((c for c in inventory if c['epoch']==best['epoch']),None)
        if existing is None:
            path=cpdir/f"best_epoch{best['epoch']:03d}.pth";write_checkpoint_bytes_new(path,best_payload)
            best=dict(best,path=str(path),sha256=sha256(path),bytes=path.stat().st_size);inventory.append(best)
        else:best=existing
        last=next(c for c in inventory if c['epoch']==30)
        csv_new(directory/'lifetime_parameter_gradients.csv',gradient_audit)
        csv_new(directory/'training_curve.csv',curves);csv_new(directory/'sampling_curve.csv',sampling_rows)
        csv_new(directory/'checkpoint_inventory.csv',inventory);json_new(directory/'checkpoint_restore_checks.json',restore_checks)
        selected=torch.load(best['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(selected['model'],strict=True);optimizer.load_state_dict(selected['optimizer'])
        selected_check=verify_checkpoint(best['path'],model,optimizer,batch)
        best_summary,best_rows=evaluate(model,val,128);best_train=evaluate_train_K1(model,train)
        csv_new(directory/'best_paired_targets.csv.gz',best_rows)
        json_new(directory/'selection.json',dict(condition=condition,best=best,last=last,best_validation=best_summary,
            best_train_K1=best_train,selected_best_restore=selected_check,completed_epochs=len(curves),
            training_targets_per_epoch=len(train),total_optimizer_steps=sum(c['steps'] for c in curves),
            total_target_visits=sum(c['targets_seen'] for c in curves),actual_run_wall_seconds=time.perf_counter()-started,
            initial_parameter_state=initial,code_commit=commit,selection_rule=plan['model_selection'],training_completed=True,
            best_at_budget_boundary=best['epoch']==30))
        event('completed',best_epoch=best['epoch'],best_ADE=best['K1_prior_ADE_m'],best_FDE=best['K1_prior_FDE_m'],actual_run_wall_seconds=time.perf_counter()-started)
    except Exception:
        details=traceback.format_exc()
        if model is not None and optimizer is not None:
            save_checkpoint_new(cpdir/f'failure_epoch{epoch:03d}.pth',model,optimizer,epoch=epoch,traceback=details,
                batch=batch,seen=seen,code_commit=commit)
        json_new(directory/'failure.json',dict(epoch=epoch,traceback=details,training_completed=False,actual_run_wall_seconds=time.perf_counter()-started))
        log.write(json.dumps(dict(event='failed',epoch=epoch,traceback=details))+'\n');raise
    finally:
        log.close()
        with (directory/'execution.txt').open('x') as stream:stream.write((directory/'execution.jsonl').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recording',required=True,choices=['biwi_hotel','crowds_zara02'])
    parser.add_argument('--condition',required=True,choices=['Uniform','Stratified'])
    args=parser.parse_args();run(args.recording,args.condition)
