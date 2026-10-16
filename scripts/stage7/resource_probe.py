"""Reproducible version of the one-shot N57 synthetic resource audit (no update)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time,traceback,torch
from scripts.stage7.admit import agentformer
from src.public_admission.inputs import synthetic_scene
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
meta,build,predict,loss,settings=agentformer(True)
paths=[f'configs/stage4p2/manifests/{rec}.json' for rec in ['biwi_hotel','crowds_zara02']]
n=max(len(w['agent_ids']) for p in paths for w in json.load(open(p))['splits']['train'])
result=dict(people=n,reason='Maximum source train context N in existing nonETH offline manifests; no trajectories or metrics evaluated',
 manifest_SHA={p:sha256(p) for p in paths},official_K20_training_objective=True,optimizer_updates=0,formal_training_runs=0,ETH_evaluated=False)
try:
 model=build().train();h,y=synthetic_scene(n);torch.cuda.reset_peak_memory_stats()
 objective=loss(model,h,y);objective.backward();torch.cuda.synchronize()
 result.update(passed=True,loss=float(objective.detach()),finite_loss=bool(torch.isfinite(objective)),
  gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception:result.update(passed=False,traceback=traceback.format_exc(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
result['elapsed_seconds']=time.perf_counter()-start
json_new('results/tables/stage7/AgentFormer_max_context_resource_probe.json',result)
print(json.dumps(result,indent=2))
