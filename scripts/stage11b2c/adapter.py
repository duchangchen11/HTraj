"""Original neural computation, CPU categorical sampler, scoped original KMeans."""
from .common import *
from scripts.stage11b1r.sampler import sample_cpu,cluster_proposals
from scripts.stage11b0.runtime import state_digest
from scripts.stage11b0.training_adapter import trajectory_maps,fixed_semantics
import torch
import torch.nn.functional as F

def digest(x):return state_digest({'tensor':x})

class AuditPredictor:
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
        trace=dict(total_seconds=total,segmentation_seconds=seg_seconds,
            feature_seconds=feature_seconds,sampling_seconds=sample_seconds,kmeans_seconds=kmeans_seconds,
            ttst_seconds=sample_seconds+kmeans_seconds,decode20_seconds=decode_seconds,
            probability_SHA=digest(prob),weights_SHA=digest(weights),proposal_SHA=digest(proposals),
            goal_SHA=digest(g),path_SHA=digest(p),path_shape=list(p.shape),goal_shape=list(g.shape),
            proposal_shape=list(proposals.shape),unit='original reference pixels',source_resize=cfg['resize'])

        # Passive export AFTER unchanged original goal/path computation.
        # The actual normalized float32 sampling weights are returned by sample_cpu.
        flat=prob.view(1,-1)
        threshold=flat.max(dim=1)[0].unsqueeze(1).expand(-1,flat.size(1))
        mask=flat<threshold*.01
        post=flat*(~mask).int()
        assert torch.equal((post/post.sum()).cpu(),weights)
        cpu=lambda t:t.detach().cpu().contiguous().clone()
        tensors=dict(logits=cpu(logits),sigmoid=cpu(prob),threshold_weights=cpu(post.reshape_as(prob)),
            normalized_weights=cpu(weights),indices=cpu(indices),proposals_network=cpu(proposals),
            proposals_raw=cpu(proposals)/cfg['resize'],centers_network=cpu(centers),
            centers_raw=g[1:,0].clone(),softargmax_raw=g[0,0].clone(),goals=g.clone(),paths=p.clone())
        assert torch.equal(tensors['centers_network']/cfg['resize'],tensors['centers_raw'])
        trace.update(logits_SHA=digest(tensors['logits']),threshold_weights_SHA=digest(tensors['threshold_weights']),
            indices_SHA=digest(tensors['indices']),centers_SHA=digest(tensors['centers_network']),
            clustering_input_SHA=digest(tensors['proposals_network'][0,0]),history_tensor_SHA=digest(history),
            semantic_SHA=digest(semantic),canvas_height=H,canvas_width=W)
        return p,g,trace,tensors
