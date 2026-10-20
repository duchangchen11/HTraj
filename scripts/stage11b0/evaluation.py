"""Metric equations only; tests use synthetic tensors, never real validation/test."""
import torch
def metrics(predictions,truth,goals=None):
    """predictions[K,N,T,2] and truth[N,T,2], same raw reference pixel coordinates."""
    if predictions.ndim!=4 or truth.shape!=predictions.shape[1:]:raise ValueError('K,N,T,xy shape mismatch')
    if not torch.isfinite(predictions).all() or not torch.isfinite(truth).all():raise ValueError('Nonfinite coordinates')
    dist=(predictions-truth.unsqueeze(0)).norm(dim=-1)
    ade=dist.mean(-1);fde=dist[:,:,-1]
    result=dict(ADE1=ade[0],FDE1=fde[0],minADE_K=ade.min(0).values,minFDE_K=fde.min(0).values,
                best_ADE_head=ade.argmin(0),best_FDE_head=fde.argmin(0))
    if goals is not None:
        if goals.shape!=predictions[:,:,-1,:].shape:raise ValueError('Goal shape mismatch')
        result['minGoalFDE_K']=(goals-truth[None,:,-1]).norm(dim=-1).min(0).values
    return result
