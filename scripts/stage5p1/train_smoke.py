"""Exactly five full passes, one fixed LORO development fold, original VAE."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import subprocess
import time
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.socialvae_causal.data import CausalDataset
from src.socialvae_causal.io import csv_new
from src.socialvae_causal.evaluation import evaluate
from src.socialvae_causal.training import seed_all,new_model,train_step,timed_batch,save_checkpoint_new,verify_checkpoint
from src.socialvae_causal.official import EXPECTED,SOURCE

OUT=Path('results/tables/stage5p1')
CONF=Path('configs/stage5p1')


def main():
    p=json.load(open(CONF/'protocol.json'));smoke=p['smoke'];resource=json.load(open(OUT/'resource_summary.json'))
    assert smoke['epochs']==5 and smoke['validation_recording']=='crowds_zara02'
    assert resource['finite_gate'] and json.load(open(OUT/'raw_counterfactual_summary.json'))['passed']
    assert json.load(open(OUT/'coverage_gate.json'))['passed']
    assert not (OUT/'smoke_summary.json').exists()
    seed_all(17);batch_size=resource['selected_smoke_batch_size']
    manifest=CONF/'manifests/crowds_zara02/B.json'
    t0=time.perf_counter();train=CausalDataset(p['data_root'],manifest,'train');val=CausalDataset(p['data_root'],manifest,'validation')
    prepare_seconds=time.perf_counter()-t0
    model=new_model();optimizer=torch.optim.Adam(model.parameters(),lr=smoke['learning_rate'])
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source_hashes={str(f):sha256(f) for f in sorted(Path('src/socialvae_causal').glob('*.py'))}
    source_hashes.update({str(f):sha256(f) for f in sorted(Path('scripts/stage5p1').glob('*.py'))})
    write_json_new(OUT/'smoke_config.json',dict(**smoke,actual_batch_size=batch_size,parameters=sum(p.numel() for p in model.parameters()),
        initialization_audit='seed17 fresh random; no resource/pretrained checkpoint loaded',manifest=str(manifest),manifest_sha256=sha256(manifest),
        protocol_sha256=sha256(CONF/'protocol.json'),source_commit=source_commit,source_sha256=source_hashes,
        official_source_sha256={k:sha256(SOURCE/k) for k in EXPECTED},preparation_seconds=prepare_seconds,
        train_target_windows=len(train),validation_target_windows=len(val),train_recordings=sorted(train.records),validation_recordings=sorted(val.records)))
    curve=[];all_steps=[];best=None;started=time.perf_counter()
    for epoch in range(1,6):
        order=np.random.default_rng(17+epoch).permutation(len(train));assert len(np.unique(order))==len(train)
        totals={k:0. for k in ['loss','rec','kl']};steps=[]
        torch.cuda.reset_peak_memory_stats();epoch_start=time.perf_counter()
        for step,start in enumerate(range(0,len(order),batch_size)):
            indices=order[start:start+batch_size];batch,data_seconds=timed_batch(train,indices)
            t0=time.perf_counter();stats=train_step(model,optimizer,batch);torch.cuda.synchronize();seconds=time.perf_counter()-t0
            row=dict(epoch=epoch,step=step+1,targets=len(indices),padded_neighbors=int(batch[2].shape[2]),
                data_seconds=data_seconds,train_seconds=seconds,**stats);steps.append(row);all_steps.append(row)
            for k in totals:totals[k]+=stats[k]*len(indices)
        train_epoch_seconds=time.perf_counter()-epoch_start
        peak=torch.cuda.max_memory_allocated()/2**20
        with torch.random.fork_rng(devices=[0]):
            t0=time.perf_counter();metrics,targets=evaluate(model,val,batch_size,'cuda',posterior=True)
            torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-t0
        checkpoint=Path('experiments/stage5p1/checkpoints/smoke_seed17')/f'epoch_{epoch:02d}.pth'
        save_checkpoint_new(checkpoint,model,optimizer,epoch=epoch,seed=17,source_commit=source_commit,
            manifest_sha256=sha256(manifest),protocol_sha256=sha256(CONF/'protocol.json'),initialization='random')
        restore=verify_checkpoint(checkpoint,model,optimizer,batch)
        row=dict(epoch=epoch,train_targets=len(train),optimizer_steps=len(steps),
            **{f'train_{k}':v/len(train) for k,v in totals.items()},**metrics,
            train_epoch_seconds=train_epoch_seconds,evaluation_seconds=evaluation_seconds,
            train_data_seconds=sum(s['data_seconds'] for s in steps),train_compute_seconds=sum(s['train_seconds'] for s in steps),
            train_targets_per_second=len(train)/train_epoch_seconds,
            gradient_L2_mean=float(np.mean([s['gradient_L2'] for s in steps])),gradient_L2_P95=float(np.quantile([s['gradient_L2'] for s in steps],.95)),
            peak_cuda_allocated_MiB=peak,train_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
            checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),**restore)
        curve.append(row)
        if best is None or row['K1_prior_ADE_m']<best['K1_prior_ADE_m']:best=row
        csv_new(OUT/'smoke'/f'epoch_{epoch:02d}_validation_targets.csv.gz',targets)
        write_json_new(OUT/'smoke'/f'epoch_{epoch:02d}_summary.json',row)
        print(json.dumps(row),flush=True)
    csv_new(OUT/'smoke_epoch_curves.csv',curve);csv_new(OUT/'smoke_step_curves.csv.gz',all_steps)
    write_json_new(OUT/'smoke_summary.json',dict(seed=17,epochs_completed=5,training_optimizer_steps=len(all_steps),
        total_target_exposures=sum(r['targets'] for r in all_steps),train_loss_decreased=curve[-1]['train_loss']<curve[0]['train_loss'],
        best=best,last=curve[-1],elapsed_seconds=time.perf_counter()-started,initialization='random',
        all_finite=True,all_strict_restore_passed=all(r['strict_model_restore'] for r in curve),formal_training=False,
        ETH_evaluated=False,FPC=False,not_a_paper_performance_result=True))


if __name__=='__main__':main()
