"""Only six-channel post-segmentation tensors; no histories or labels accepted."""
import torch
CONDITIONS=['S0','S1','S2','S3']
def intervene(semantic,condition,valid_height,valid_width):
    assert semantic.ndim==4 and semantic.shape[:2]==(1,6) and semantic.dtype==torch.float32
    assert torch.isfinite(semantic).all() and condition in CONDITIONS
    H,W=semantic.shape[-2:];h,w=valid_height,valid_width
    assert 0<h<=H and 0<w<=W
    out=semantic.clone()
    if condition=='S1':out.zero_()
    elif condition=='S2':out[:,:,:h,:w]=semantic[:,:,:h,:w].mean(dim=(-2,-1),keepdim=True)
    elif condition=='S3':out[:,:,:h,:w]=semantic[:,:,:h,:w].flip(-1)
    return out
