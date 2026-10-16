"""History-only interfaces, with training labels attached explicitly afterwards."""
import numpy as np
import torch


def history_array(history):
    h=np.asarray(history,dtype=np.float64)
    if h.ndim!=3 or h.shape[1:]!=(8,2) or len(h)<1 or not np.isfinite(h).all():
        raise ValueError('Finite N>=1,8,2 observed positions required')
    return h


def agentformer_packet(history):
    h=history_array(history)
    return dict(pre_motion_3D=[torch.tensor(x,dtype=torch.float32) for x in h],
        pre_motion_mask=[torch.ones(8) for _ in h],
        fut_motion_3D=[torch.zeros(12,2) for _ in h],
        fut_motion_mask=[torch.zeros(12) for _ in h],heading=None)


def pecnet_packet(history,scale=1.86,radius=2.):
    h=history_array(history)
    x=(h-h[:,:1])*scale
    # Preserve the published utility's /1000 channel convention. Not a claim
    # that SDD pixel normalization is an official ETH/UCY meter protocol.
    initial=h[:,-1]/1000.
    distance=np.linalg.norm(h[:,-1,None]-h[None,:,-1],axis=-1)
    mask=(distance<=radius).astype(np.float64)
    return tuple(torch.tensor(v,dtype=torch.float64) for v in (x.reshape(len(h),16),initial,mask))


def synthetic_scene(people):
    if people<1:raise ValueError('At least one agent')
    i=np.arange(people);origin=np.stack((.35*i,.15*(i%3)),axis=-1)
    velocity=np.stack((.3+.04*(i%4),.1*np.cos(i)),axis=-1)
    past=origin[:,None]+np.arange(8)[None,:,None]*.4*velocity[:,None]
    future=origin[:,None]+np.arange(8,20)[None,:,None]*.4*velocity[:,None]
    return past,future
