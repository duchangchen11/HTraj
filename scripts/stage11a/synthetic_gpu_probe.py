"""Core-only GPU forward resource lower bound, no training/no candidate metrics."""
from common import *
import importlib,time
import torch

def main():
    torch.manual_seed(17);torch.set_num_threads(1)
    sys.path.insert(0,str(CACHE/'official/ynet'))
    cfg=json.loads((OUT/'baseline_probe.json').read_text())['engineering_config']
    cfg={k:v for k,v in cfg.items() if k!='meaning'}
    module=importlib.import_module('model')
    if not torch.cuda.is_available():
        jnew(OUT/'gpu_probe.json',dict(executed=False,reason='CUDA unavailable'));return
    rows=[]
    for b,h,w in [(1,256,384),(4,256,384),(1,512,768)]:
        model=module.YNetTorch(**cfg).cuda().eval()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
        try:
            with torch.no_grad():
                x=torch.rand(b,14,h,w,device='cuda')
                torch.cuda.synchronize();t=time.perf_counter()
                feat=model.pred_features(x);goal=model.pred_goal(feat)
                for _ in range(20):
                    z=torch.rand(b,1,h,w,device='cuda')
                    down=[z]+[torch.nn.functional.avg_pool2d(z,2**i,2**i) for i in range(1,len(feat))]
                    traj=model.pred_traj([torch.cat([f,p],1) for f,p in zip(feat,down)])
                    xy=model.softargmax(traj)
                    assert torch.isfinite(xy).all()
                torch.cuda.synchronize();duration=time.perf_counter()-t
            assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
            rows.append(dict(batch=b,H=h,W=w,core_forward_20_decoder_seconds=duration,
                allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                whole_card_before_MiB=before,whole_card_after_MiB=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip(),
                finite=True,OOM=False,weights_unchanged=True))
            del x,feat,goal,z,down,traj,xy
        except torch.cuda.OutOfMemoryError as e:
            rows.append(dict(batch=b,H=h,W=w,OOM=True,error=str(e)));break
        finally:del model
    jnew(OUT/'gpu_probe.json',dict(executed=True,rows=rows,GPU=torch.cuda.get_device_name(),
        torch=torch.__version__,optimizer_updates=0,training=False,segmentation_included=False,
        author_config=False,TTST_CWS_included=False,
        note='20 synthetic conditioned decoder calls are NOT official K20 generation; forward resource lower bound only. No full-epoch/gradient measurement.'))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
