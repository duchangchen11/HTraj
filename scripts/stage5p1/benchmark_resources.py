"""Measured batch16/32/64/128 training updates, never extrapolate small batches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gc
import json
import subprocess
import time
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.socialvae_causal.data import CausalDataset
from src.socialvae_causal.io import csv_new
from src.socialvae_causal.training import seed_all,new_model,train_step,timed_batch,save_checkpoint_new,verify_checkpoint

OUT=Path('results/tables/stage5p1')
CONF=Path('configs/stage5p1')


def run_size(size,data,plan):
    seed_all(17);model=new_model();optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    order=np.random.default_rng(17).permutation(len(data));updates=[]
    torch.cuda.reset_peak_memory_stats()
    total=plan['warmup_updates']+plan['timed_updates']
    for step in range(total):
        indices=order[step*size:(step+1)*size];assert len(indices)==size
        batch,data_seconds=timed_batch(data,indices)
        torch.cuda.synchronize();t0=time.perf_counter();stats=train_step(model,optimizer,batch);torch.cuda.synchronize()
        updates.append(dict(batch_size=size,step=step+1,role='warmup' if step<plan['warmup_updates'] else 'timed',
            targets=size,padded_neighbors=batch[2].shape[2],data_seconds=data_seconds,train_seconds=time.perf_counter()-t0,**stats))
    typical_peak=torch.cuda.max_memory_allocated()/2**20
    # This is the worst stored history-context size, not a made-up dense graph.
    largest=sorted(range(len(data)),key=lambda i:(-len(data.contexts[data.indices[i][0]].ids),i))[:size]
    del batch;torch.cuda.reset_peak_memory_stats()
    batch,ds=timed_batch(data,largest);t0=time.perf_counter();stats=train_step(model,optimizer,batch);torch.cuda.synchronize()
    stress_seconds=time.perf_counter()-t0;stress_peak=torch.cuda.max_memory_allocated()/2**20
    updates.append(dict(batch_size=size,step=total+1,role='largest_context',targets=size,
        padded_neighbors=batch[2].shape[2],data_seconds=ds,train_seconds=stress_seconds,**stats))
    checkpoint=Path('experiments/stage5p1/checkpoints/resources')/f'batch{size}_after{total+1}updates.pth'
    save_checkpoint_new(checkpoint,model,optimizer,seed=17,steps=total+1,initialization='random',purpose='resource benchmark only')
    restore=verify_checkpoint(checkpoint,model,optimizer,batch)
    timed=[r for r in updates if r['role']=='timed']
    average=float(np.mean([r['train_seconds'] for r in timed]));data_average=float(np.mean([r['data_seconds'] for r in timed]))
    result=dict(batch_size=size,status='passed',parameters=sum(p.numel() for p in model.parameters()),
        optimizer_steps=total+1,warmup_updates=plan['warmup_updates'],timed_updates=plan['timed_updates'],
        mean_step_seconds=average,median_step_seconds=float(np.median([r['train_seconds'] for r in timed])),
        P95_step_seconds=float(np.quantile([r['train_seconds'] for r in timed],.95)),mean_data_seconds=data_average,
        compute_targets_per_second=size/average,end_to_end_targets_per_second=size/(average+data_average),
        typical_peak_cuda_allocated_MiB=typical_peak,largest_context_peak_cuda_allocated_MiB=stress_peak,
        largest_context_neighbors=int(batch[2].shape[2]),largest_context_step_seconds=stress_seconds,
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),**restore)
    csv_new(OUT/'resources'/f'batch{size}_steps.csv',updates)
    write_json_new(OUT/'resources'/f'batch{size}_summary.json',result)
    del model,optimizer,batch;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=json.load(open(CONF/'protocol.json'))
    assert json.load(open(OUT/'raw_counterfactual_summary.json'))['passed']
    assert json.load(open(OUT/'coverage_gate.json'))['passed']
    assert not (OUT/'resource_summary.json').exists()
    manifest=CONF/'manifests/crowds_zara02/B.json'
    t0=time.perf_counter();data=CausalDataset(p['data_root'],manifest,'train');preparation=time.perf_counter()-t0
    rows=[]
    for size in p['resource_benchmark']['batch_sizes']:
        try:result=run_size(size,data,p['resource_benchmark'])
        except torch.cuda.OutOfMemoryError as e:
            result=dict(batch_size=size,status='CUDA_out_of_memory',error=str(e));gc.collect();torch.cuda.empty_cache()
        except (FloatingPointError,ValueError,AssertionError) as e:
            result=dict(batch_size=size,status='failed_stop',error=str(e));rows.append(result);print(json.dumps(result),flush=True);break
        rows.append(result);print(json.dumps(result),flush=True)
    successful=[r['batch_size'] for r in rows if r['status']=='passed']
    write_json_new(OUT/'resource_summary.json',dict(rows=rows,selected_smoke_batch_size=max(successful) if successful else None,
        finite_gate=bool(successful) and not any(r['status']=='failed_stop' for r in rows),
        data_preparation_seconds=preparation,train_target_windows=len(data),
        manifest=str(manifest),manifest_sha256=sha256(manifest),protocol_sha256=sha256(CONF/'protocol.json'),
        script_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        GPU=torch.cuda.get_device_name(0),total_cuda_memory_MiB=torch.cuda.get_device_properties(0).total_memory/2**20,
        no_pretrained_initialization=True,ETH_evaluated=False,formal_training=False))


if __name__=='__main__':main()
