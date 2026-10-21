"""Real recorded phase costs; no invented isolated export measurement."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd,numpy as np,torch

def main():
    assert not torch.cuda.is_initialized();m=pd.read_csv(OUT/'tensor_manifest.csv');fields=['feature_seconds','sampling_seconds','kmeans_seconds','decode20_seconds','extract_seconds','cache_seconds'];stats={}
    for k in fields:stats[k]=dict(mean=float(m[k].mean()),median=float(m[k].median()),P90=float(m[k].quantile(.9)),sum=float(m[k].sum()))
    overhead=m.extract_seconds-m.total_seconds;assert (overhead>=0).all();stats['combined_passive_export_and_SHA_seconds']=dict(mean=float(overhead.mean()),median=float(overhead.median()),sum=float(overhead.sum()))
    gpu=[]
    for p in sorted(OUT.glob('GPU_card_*.csv')):
        a=pd.read_csv(p,skipinitialspace=True,comment='#');gpu.append(dict(file=p.name,samples=len(a),card_memory_peak_MiB=float(a.memory_used_MiB.max()),utilization_mean_percent=float(a.utilization_gpu_percent.mean())))
    jnew(OUT/'resource_summary.json',dict(total_GPU_process_seconds=parent_seconds(),budget_hard_seconds=7200,safe_seconds=6600,phase_timings=stats,
        pure_probability_map_D2H_seconds=None,pure_export_time_not_separately_instrumented=True,
        export_timing_scope='Observed extract_seconds minus original forward total includes passive export+SHA and synchronization overhead; not pure D2H latency',
        tensor_cache_files=998,tensor_cache_bytes=int(m.bytes.sum()),dtype='float32 maps and xy; int64 indices; no quantization',GPU_card=gpu,
        peak_allocated_bytes=int(m.peak_allocated_bytes.max()),peak_reserved_bytes=int(m.peak_reserved_bytes.max()),CPU_threads_for_GPU_worker=1,
        training_steps=0,optimizer_updates=0,OOM=False,CPU_reporting_CUDA_initialized=False,source_SHA=sha(Path(__file__))))
    print('Resource audit complete; whole process includes all1996 targets plus precheck',flush=True)
if __name__=='__main__':main()
