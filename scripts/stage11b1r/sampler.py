"""HTraj adapter around frozen Y-Net probability and KMeans definitions.

Author MIT sampling: utils/image_utils.py at a345ada, lines92ff.
Threshold and normalization stay on the model device in original float32.
Only categorical draws move to a local CPU Generator. No model/loss changes.
"""
from contextlib import contextmanager
import hashlib, threading
import numpy as np
import torch
from scripts.stage11b0.runtime import original_modules
from scripts.stage11b1.evaluation import sample_seed

_rng_lock=threading.RLock()

def kmeans_seed(sample_id):
    return int.from_bytes(hashlib.sha256(('HTraj-YNet-K20-v2:kmeans:'+sample_id).encode()).digest()[:4],'big')

@torch.no_grad()
def probability_weights(probability_map, rel_threshold=.01):
    """Exactly the author's threshold/max/global sum, before device transfer."""
    if probability_map.ndim!=4 or probability_map.shape[:2]!=(1,1):
        raise ValueError('One target, one goal channel per stable sample_id required')
    if probability_map.dtype!=torch.float32:raise ValueError('Original inference float32 required')
    if not torch.isfinite(probability_map).all() or (probability_map<0).any():
        raise ValueError('Invalid categorical weights')
    weights=probability_map.view(1,-1)
    if rel_threshold is not None:
        threshold=weights.max(dim=1)[0].unsqueeze(1).expand(-1,weights.size(1))
        mask=weights<threshold*rel_threshold
        weights=weights*(~mask).int()
        weights=weights/weights.sum()
    if not torch.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('No valid probability support')
    return weights

@torch.no_grad()
def sample_cpu(probability_map, sample_id):
    weights=probability_weights(probability_map)
    cpu_weights=weights.detach().to('cpu')  # No re-normalization or dtype conversion.
    generator=torch.Generator(device='cpu').manual_seed(sample_seed(sample_id))
    indices=torch.multinomial(cpu_weights,10000,replacement=True,generator=generator)
    index=indices.reshape(1,1,10000)
    xy=index.unsqueeze(3).repeat(1,1,1,2).float()
    width=probability_map.shape[-1]
    xy[:,:,:,0]=xy[:,:,:,0]%width
    xy[:,:,:,1]=torch.floor(xy[:,:,:,1]/width)
    return xy.to(probability_map.device),cpu_weights,indices

@contextmanager
def isolated_kmeans_rng(sample_id,device):
    """Scope the original np.choice and CPU empty-cluster torch.randint.

No Python random op exists in KMeans. CUDA RNG is snapshotted/restored too,
although original KMeans does not draw CUDA random numbers. Serial use only;
lock prevents concurrent adapter calls from interleaving global RNG contexts.
"""
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with _rng_lock:
        before=np.random.get_state()
        with torch.random.fork_rng(devices=devices):
            seed=kmeans_seed(sample_id)
            np.random.set_state(np.random.RandomState(seed).get_state())
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            try:yield
            finally:np.random.set_state(before)

def cluster_proposals(proposals,sample_id):
    km=original_modules().model.evaluate.__globals__['kmeans']
    with isolated_kmeans_rng(sample_id,proposals.device):
        return km(X=proposals,num_clusters=19,distance='euclidean',device=proposals.device,
                  tqdm_flag=False,tol=.001,iter_limit=1000)[1]
