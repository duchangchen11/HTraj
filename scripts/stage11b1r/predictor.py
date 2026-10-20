"""Original neural computation, CPU categorical sampler, scoped original KMeans."""
from .common import *
from .sampler import sample_cpu,cluster_proposals
from scripts.stage11b0.runtime import state_digest
from scripts.stage11b0.training_adapter import trajectory_maps,fixed_semantics
import torch
import torch.nn.functional as F

def digest(x):return state_digest({'tensor':x})

class Predictor:
    def __init__(self,net,cfg,templates):
        assert cfg['waypoints']==[11] and cfg['use_CWS'] is False
        assert cfg['resize']==.25
        self.net,self.cfg,self.templates=net,cfg,templates

    @torch.no_grad()
    def predict(self,sample_id,history,image=None,semantic=None):
        if history.shape!=(1,8,2) or not torch.isfinite(history).all():raise ValueError('Finite history8 required')
        net,cfg=self.net,self.cfg;net.eval();device=next(net.parameters()).device
        sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
        sync();start=time.perf_counter();seg_seconds=0.
        if semantic is None:
            if image is None:raise ValueError('Image or semantic required')
            t=time.perf_counter();semantic=fixed_semantics(net,image);sync();seg_seconds=time.perf_counter()-t
        H,W=semantic.shape[-2:]
        observed=trajectory_maps(history.to(device)*cfg['resize'],H,W,self.templates[0])
        features=net.pred_features(torch.cat([semantic,observed],1))
        logits=net.pred_goal(features)[:,[11]];prob=net.sigmoid(logits/1.)
        sync();feature_seconds=time.perf_counter()-start-seg_seconds;t=time.perf_counter()
        proposals,weights,indices=sample_cpu(prob,sample_id)
        sync();sample_seconds=time.perf_counter()-t;t=time.perf_counter()
        centers=cluster_proposals(proposals[0,0],sample_id)
        goals=torch.cat([net.softargmax(logits).unsqueeze(0),centers[:,None,None,:]],0)
        sync();kmeans_seconds=time.perf_counter()-t;t=time.perf_counter();paths=[]
        for goal in goals:
            wm=trajectory_maps(goal,H,W,self.templates[0])
            pyramid=[wm]+[F.avg_pool2d(wm,2**i,2**i) for i in range(1,len(features))]
            paths.append(net.softargmax(net.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)])))
        paths=torch.stack(paths)/cfg['resize'];goals=goals[:,:,0]/cfg['resize']
        sync();decode_seconds=time.perf_counter()-t;total=time.perf_counter()-start
        assert paths.shape==(20,1,12,2) and goals.shape==(20,1,2)
        if not torch.isfinite(paths).all() or not torch.isfinite(goals).all():raise FloatingPointError('Nonfinite K20 output')
        # Hashing/serialization overhead is excluded from the operation timing.
        p,g=paths.cpu(),goals.cpu()
        return p,g,dict(total_seconds=total,segmentation_seconds=seg_seconds,
            feature_seconds=feature_seconds,sampling_seconds=sample_seconds,kmeans_seconds=kmeans_seconds,
            ttst_seconds=sample_seconds+kmeans_seconds,decode20_seconds=decode_seconds,
            probability_SHA=digest(prob),weights_SHA=digest(weights),proposal_SHA=digest(proposals),
            goal_SHA=digest(g),path_SHA=digest(p),path_shape=list(p.shape),goal_shape=list(g.shape),
            proposal_shape=list(proposals.shape),unit='original reference pixels',source_resize=cfg['resize'])
