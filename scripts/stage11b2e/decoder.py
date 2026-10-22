"""Only the original goal-conditioned Path Decoder math, without TTST/Goal prediction."""
import time
import torch
import torch.nn.functional as F
from scripts.stage11b0.training_adapter import trajectory_maps

@torch.no_grad()
def decode(net,cfg,templates,history,semantic,goals_raw):
    assert not net.training and not net.semantic_segmentation.training
    assert history.shape==(1,8,2) and goals_raw.shape==(20,2)
    device=next(net.parameters()).device
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    sync();start=time.perf_counter()
    H,W=semantic.shape[-2:]
    observed=trajectory_maps(history.to(device)*cfg['resize'],H,W,templates[0])
    features=net.pred_features(torch.cat([semantic,observed],1))
    sync();feature_seconds=time.perf_counter()-start;t=time.perf_counter()
    goals=goals_raw.to(device)[:,None,None,:]*cfg['resize']
    paths=[]
    for goal in goals:
        wm=trajectory_maps(goal,H,W,templates[0])
        pyramid=[wm]+[F.avg_pool2d(wm,2**i,2**i) for i in range(1,len(features))]
        paths.append(net.softargmax(net.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)])))
    paths=torch.stack(paths)/cfg['resize']
    sync();decode_seconds=time.perf_counter()-t
    assert paths.shape==(20,1,12,2) and torch.isfinite(paths).all()
    return paths.detach().cpu(),dict(feature_seconds=feature_seconds,decode20_seconds=decode_seconds,total_seconds=time.perf_counter()-start)
