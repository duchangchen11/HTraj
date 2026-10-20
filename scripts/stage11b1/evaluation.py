"""Original TTST(10000) -> 19 centers + softargmax -> 20 decoded paths.

No GT argument in the inference interface. Each target is inferred separately,
with a stable identity seed, so input order cannot change its random stream.
Author MIT model/functions are read-only; segmentation alone may be cached.
"""
from .common import *
from scripts.stage11b0.runtime import original_modules, seed_all
from scripts.stage11b0.training_adapter import trajectory_maps, fixed_semantics
from scripts.stage11b0.evaluation import metrics as original_metrics
import torch
import torch.nn.functional as F

def sample_seed(sample_id):
    return int.from_bytes(hashlib.sha256(('HTraj-YNet-SDD-Development-v1:17:'+sample_id).encode()).digest()[:4], 'big')

def restored_pixels(x, resize):
    if not (resize > 0): raise ValueError('Invalid resize')
    return x / resize

def metrics(paths, truth, goals):
    r = original_metrics(paths, truth, goals)
    r['GoalFDE1'] = (goals[0]-truth[:,-1]).norm(dim=-1)
    return r

class Predictor:
    def __init__(self, net, cfg, templates):
        self.net, self.cfg, self.templates = net, cfg, templates
        assert cfg['waypoints']==[11] and cfg['use_CWS'] is False

    @torch.no_grad()
    def predict(self, sample_id, history, image=None, semantic=None):
        """history[1,8,2] RAW pixels; image or frozen segmentation, never GT."""
        if history.shape != (1,8,2) or not torch.isfinite(history).all():
            raise ValueError('Exactly one finite history8 required')
        net, cfg = self.net, self.cfg
        net.eval(); seed_all(sample_seed(sample_id))
        device = next(net.parameters()).device
        sync = lambda: torch.cuda.synchronize(device) if device.type=='cuda' else None
        sync(); start=time.perf_counter(); seg_seconds=0.
        if semantic is None:
            if image is None: raise ValueError('Image or semantic required')
            t=time.perf_counter(); semantic=fixed_semantics(net,image); sync(); seg_seconds=time.perf_counter()-t
        H,W=semantic.shape[-2:]
        observed=trajectory_maps(history.to(device)*cfg['resize'],H,W,self.templates[0])
        feats=net.pred_features(torch.cat([semantic,observed],1))
        logits=net.pred_goal(feats)[:,cfg['waypoints']]
        prob=net.sigmoid(logits/1.0)
        sync(); feature_seconds=time.perf_counter()-start-seg_seconds; t=time.perf_counter()
        # Torch2.5 rejects CUDA multinomial's internal cumsum under the global
        # deterministic-algorithms flag. Execute the unchanged author sampler
        # in a narrowly scoped compatibility exception; restore training's flag.
        # Repeated seeded K20 output is checked bitwise by the real-data gate.
        deterministic=torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            proposal=original_modules().image.sampling(prob[:,-1:],10000,replacement=True,rel_threshold=.01).permute(2,0,1,3)
        finally:
            torch.use_deterministic_algorithms(deterministic)
        # This is the unmodified author's function, including its CPU/GPU behavior.
        km=original_modules().model.evaluate.__globals__['kmeans']
        _,centers=km(X=proposal[:,0,0],num_clusters=19,distance='euclidean',device=device,
                     tqdm_flag=False,tol=.001,iter_limit=1000)
        goals=torch.cat([net.softargmax(logits[:,-1:]).unsqueeze(0),centers[:,None,None,:]],0)
        sync(); ttst_seconds=time.perf_counter()-t; t=time.perf_counter()
        paths=[]
        for goal in goals:
            wm=trajectory_maps(goal,H,W,self.templates[0])
            pyramid=[wm]+[F.avg_pool2d(wm,2**i,2**i) for i in range(1,len(feats))]
            paths.append(net.softargmax(net.pred_traj([torch.cat([f,w],1) for f,w in zip(feats,pyramid)])))
        paths=restored_pixels(torch.stack(paths),cfg['resize'])
        goals=restored_pixels(goals[:,:,0],cfg['resize'])
        sync(); decode_seconds=time.perf_counter()-t
        assert paths.shape==(20,1,12,2) and goals.shape==(20,1,2)
        assert torch.isfinite(paths).all() and torch.isfinite(goals).all()
        return paths.detach().cpu(),goals.detach().cpu(),dict(
            total_seconds=time.perf_counter()-start,segmentation_seconds=seg_seconds,
            feature_seconds=feature_seconds,ttst_kmeans_seconds=ttst_seconds,decode20_seconds=decode_seconds)

def history_record(record):
    """Project a raw trajectory record onto the only allowed predictor fields."""
    return record['sample_id'], torch.as_tensor(record['xy'][:8],dtype=torch.float32).unsqueeze(0)
