"""Synthetic forward/backward only. No GT file, validation prediction or update."""
from common import *
import dataclasses
import platform
import time
import numpy as np
import torch
from src.risk_dynamic_graph import GraphConfig, RiskGraphModel, build_graph


def smi():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()


def main():
    started = time.perf_counter()
    c = GraphConfig(**json.loads((CONF / 'graph.json').read_text()))
    protocol = json.loads((CONF / 'protocol.json').read_text())
    assert torch.cuda.is_available(), 'No GPU; do not invent GPU benchmark'
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rows, capacities = [], []
    for device in ('cpu','cuda'):
        for n in ([12] if device == 'cpu' else protocol['synthetic_smoke']['people']):
            batch = 1 if device == 'cpu' else 8
            gen = torch.Generator().manual_seed(1700+n)
            base = torch.randn(batch, 1, n, 2, generator=gen) * 1.5
            velocity = torch.randn(batch, 1, n, 2, generator=gen) * .6
            t = torch.arange(8)[None,:,None,None] * .4
            p = (base + t * velocity).to(device)
            mask = torch.ones(batch,8,n,dtype=torch.bool,device=device)
            ids = torch.arange(n,device=device,dtype=torch.int64)[None].expand(batch,-1)
            for mode in ('E','S','D','R'):
                torch.manual_seed(17)
                model = RiskGraphModel(mode,64,c).to(device)
                initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                times = []
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                start_mem = smi()
                for repeat in range(10):
                    model.zero_grad(set_to_none=True)
                    if device == 'cuda': torch.cuda.synchronize()
                    start = time.perf_counter()
                    out = model(p,mask,ids)
                    objective = out.square().mean()
                    objective.backward()
                    if device == 'cuda': torch.cuda.synchronize()
                    seconds = time.perf_counter()-start
                    assert torch.isfinite(out).all() and torch.isfinite(objective)
                    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
                    if repeat >= 2:
                        times.append(seconds)
                unchanged = all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
                assert unchanged  # no optimizer instance; no training
                nominal = sum(v.numel() for v in model.parameters())
                loss_connected = sum(v.numel() for v in model.parameters() if v.grad is not None)
                nonzero = sum(int(torch.count_nonzero(v.grad)) for v in model.parameters() if v.grad is not None)
                # E: all message weights disconnected; decoder social columns
                # structurally inactive even though their tensor has gradients.
                structural = nominal if mode != 'E' else nominal - sum(v.numel() for v in model.message.parameters()) - 64*64
                if n == 12 and device == 'cpu':
                    capacities.append(dict(model=mode,nominal=nominal,structural_effective_upper=structural,
                        loss_connected_tensor_numel=loss_connected,nonzero_gradient_elements_one_fixture=nonzero,
                        permanently_disconnected_message_numel=nominal-loss_connected,
                        E_decoder_inactive_elements=4096 if mode=='E' else 0,
                        note='Connected tensor size is not active element count; one synthetic gradient cannot establish long-term activity'))
                graph = build_graph(p,mask,ids,mode,c)
                rows.append(dict(device=device,model=mode,batch_scenes=batch,N=n,
                    mean_seconds=float(np.mean(times)),p50=float(np.quantile(times,.5)),p90=float(np.quantile(times,.9)),p99=float(np.quantile(times,.99)),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                    reserved_peak_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
                    whole_card_before=start_mem,whole_card_after=smi(),
                    edges=int(graph['edge_mask'].sum()),nominal_parameters=nominal,finite=True,OOM=False,
                    optimizer_updates=0,weights_bitwise_unchanged=unchanged,all_step_seconds=times))
                print(device,mode,'B',batch,'N',n,'mean',np.mean(times),flush=True)
                del model,out,objective,graph
    audit = json.loads((OUT/'data_audit.json').read_text())
    budget = []
    for fold in ('biwi_hotel','crowds_zara02'):
        train = next(r for r in audit['summary'] if r['fold']==fold and r['split']=='train')
        steps = math_ceil(train['supervised_windows']/8)*30
        for mode in ('E','S','D','R'):
            costs = [r['mean_seconds'] for r in rows if r['device']=='cuda' and r['model']==mode and r['N'] in (12,79)]
            budget.append(dict(fold=fold,model=mode,epochs=30,scene_batch=8,targets_per_epoch=train['targets'],
                steps_per_epoch=math_ceil(train['supervised_windows']/8),total_steps=steps,
                synthetic_compute_low_seconds=steps*min(costs),synthetic_compute_high_seconds=steps*max(costs),
                planning_low_seconds=steps*min(costs)*1.5,planning_high_seconds=steps*max(costs)*4,
                assumption='N12 representative / N79 all batches padded maximum; 1.5x-4x overhead for data, Adam, validation and saves; estimate, not measured full epoch'))
    write_json(OUT/'resources.json',dict(environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        platform=platform.platform(),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),torch_threads=torch.get_num_threads()),
        protocol_commit=commit(),protocol_SHA=sha(CONF/'protocol.json'),rows=rows,capacities=capacities,budget=budget,
        total_planning_GPU_hours=[sum(r[k] for r in budget)/3600 for k in ('planning_low_seconds','planning_high_seconds')],
        total_smoke_wall_seconds=time.perf_counter()-started,formal_training=False,validation_prediction=False))
    write_csv(OUT/'parameters.csv',capacities)
    write_csv(OUT/'budget_estimate.csv',budget)


def math_ceil(v):
    import math
    return math.ceil(v)


if __name__ == '__main__':
    main()
