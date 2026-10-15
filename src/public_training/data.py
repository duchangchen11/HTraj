"""Complete-history public protocols; graphs accept history arrays only."""
import numpy as np
import torch

RECORDINGS=('biwi_hotel','students001','students003','crowds_zara01','crowds_zara02')


def select_window(by_frame,frames,protocol):
    assert protocol in ('A_offline_complete20','B_history_context_complete_targets')
    frames=list(map(int,frames));assert len(frames)==20 and np.all(np.diff(frames)==10)
    history=set.intersection(*(set(by_frame.get(f,{})) for f in frames[:8]))
    complete=history.intersection(*(set(by_frame.get(f,{})) for f in frames[8:]))
    ids=sorted(complete if protocol=='A_offline_complete20' else history)
    if len(ids)<2 or not complete:return None
    x=np.array([[by_frame[f][i] for i in ids] for f in frames[:8]],dtype=np.float64)
    y=np.array([[by_frame.get(f,{}).get(i,[0.,0.]) for i in ids] for f in frames[8:]],dtype=np.float64)
    # Same four-decimal positions as upstream before conversion to float32.
    x=torch.tensor(np.round(x,4),dtype=torch.float32);y=torch.tensor(np.round(y,4),dtype=torch.float32)
    mask=torch.tensor([i in complete for i in ids],dtype=torch.bool)
    return dict(observed=x,future=y,target_mask=mask,agent_ids=ids,target_ids=sorted(complete),frames=frames)


def displacement(x):
    first=torch.zeros_like(x[:1]);return torch.cat([first,x[1:]-x[:-1]],0)


def graph_from_history(observed):
    """Official inverse displacement-difference weights and normalized Laplacian."""
    rel=displacement(observed)
    # float32 subtraction, float64 norm/normalization, finally float32 graph.
    diff=(rel[:,:,None]-rel[:,None,:]).double()
    distance=diff.square().sum(-1).sqrt();w=torch.where(distance>0,1/distance.clamp_min(1e-300),0.)
    eye=torch.eye(observed.shape[1],dtype=torch.float64,device=observed.device)[None]
    w=w*(1-eye)+eye;inv=w.sum(-1).rsqrt()
    lap=eye-w*inv[:,:,None]*inv[:,None,:]
    return lap.to(observed.dtype)


def network_input(sample,device='cpu'):
    return displacement(sample['observed']).permute(2,0,1)[None].to(device),sample['graph'].to(device)


def permute_graph_input(v,a,order):
    order=torch.as_tensor(order,dtype=torch.long,device=v.device)
    return v.index_select(-1,order),a.index_select(-2,order).index_select(-1,order)


def restore_predictions(prediction,order):
    order=torch.as_tensor(order,dtype=torch.long,device=prediction.device)
    return prediction.index_select(-1,torch.argsort(order))
