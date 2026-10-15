"""HTraj-SocialVAE-Causal8x12-v1: raw-eight-frame-only inference packets.

Metadata validity is kept separately from the fixed six-dimensional interface.
No missing-velocity neighbor is passed to motion attention as a stationary one.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.preprocessing import Recording, sha256

PROTOCOL = 'HTraj-SocialVAE-Causal8x12-v1'
OFFLINE = 'SocialVAE-Official-Offline'
PADDING = np.float32(1e9)
NON_ETH = {'biwi_hotel','students001','students003','crowds_zara01','crowds_zara02'}


def array_sha(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


@dataclass
class HistoryContext:
    ids: np.ndarray
    state: np.ndarray
    packet: np.ndarray
    position_valid: np.ndarray
    velocity_valid: np.ndarray
    acceleration_valid: np.ndarray
    input_support: tuple


def history_context(recording, start):
    """Only eight get() calls on raw positions; no recording-wide extension."""
    if recording.frame_step != 10 or recording.dt != .4:
        raise ValueError('Causal8x12-v1 requires the frozen 10-tick/0.4s grid')
    frames = tuple(int(start)+10*t for t in range(8))
    raw = [recording.by_frame.get(f,{}) for f in frames]
    ids = np.asarray(sorted(set().union(*(r.keys() for r in raw))), dtype=np.int64)
    state = np.zeros((8,len(ids),6),dtype=np.float32)
    pv = np.zeros((8,len(ids)),dtype=bool)
    index = {int(i):j for j,i in enumerate(ids)}
    for t,r in enumerate(raw):
        for actor,xy in r.items():
            state[t,index[actor],:2] = xy
            pv[t,index[actor]] = True
    vv = np.zeros_like(pv)
    av = np.zeros_like(pv)
    vv[1:] = pv[1:] & pv[:-1]
    state[1:,:,2:4] = np.where(vv[1:,:,None],state[1:,:,:2]-state[:-1,:,:2],0.)
    av[2:] = vv[2:] & vv[1:-1]
    state[2:,:,4:6] = np.where(av[2:,:,None],state[2:,:,2:4]-state[1:-1,:,2:4],0.)
    # Unknown acceleration at the window boundary remains a declared boundary
    # prior, validity=False. Ego's missing first velocity slot is unused by
    # official enc(): it recomputes seven velocities from eight real positions.
    usable = vv.copy()
    usable[0] = pv[0]  # official initial hidden state uses position only here
    packet = np.where(usable[:,:,None],state,PADDING).astype(np.float32)
    return HistoryContext(ids,state,packet,pv,vv,av,frames)


def target_future(recording, start, actor):
    y = np.zeros((12,2),np.float32)
    valid = np.zeros(12,bool)
    for t in range(12):
        xy = recording.by_frame.get(int(start)+10*(8+t),{}).get(int(actor))
        if xy is not None:
            y[t], valid[t] = xy, True
    return y,valid


def example_from_context(context, recording, start, actor, require_supervised=True):
    idx = np.flatnonzero(context.ids == actor)
    if len(idx)!=1:raise ValueError('Target absent from eight-frame history')
    i = int(idx[0])
    if not context.position_valid[:,i].all():raise ValueError('Target needs complete eight-frame history')
    # Build inference data before reading labels. Labels never select context.
    others = np.arange(len(context.ids)) != i
    x = context.state[:,i].copy()
    neighbor = np.full((20,int(others.sum()),6),PADDING,np.float32)
    neighbor[:8] = context.packet[:,others]
    y, ym = target_future(recording,start,actor)
    if require_supervised and not ym.all():raise ValueError('Supervision needs complete12 labels')
    return dict(x=x,y=y,neighbor=neighbor,future_valid=ym,
        actor=int(actor),recording=recording.name,start_frame=int(start),
        context_ids=context.ids[others].copy(),position_valid=context.position_valid[:,others].copy(),
        velocity_valid=context.velocity_valid[:,others].copy(),acceleration_valid=context.acceleration_valid[:,others].copy(),
        target_velocity_valid=context.velocity_valid[:,i].copy(),target_acceleration_valid=context.acceleration_valid[:,i].copy())


def raw_example(recording,start,actor,require_supervised=True):
    return example_from_context(history_context(recording,start),recording,start,actor,require_supervised)


def context_counts(context,target_index,radius=2.):
    others = np.arange(len(context.ids)) != target_index
    visible = context.position_valid[:,others]
    relative = context.state[:,others,:2]-context.state[:,target_index:target_index+1,:2]
    near = visible & (np.linalg.norm(relative,axis=-1)<=radius)
    valid = context.velocity_valid[:,others]
    attempted = near[1:]
    blocked = attempted & ~valid[1:]
    legal = attempted & valid[1:]
    first_seen = np.where(visible.any(0),visible.argmax(0),-1)
    any_near = bool(near.any())
    available = bool(near[0].any() or legal.any())
    return dict(history_neighbors=int(visible.any(0).sum()),new_entering_neighbors=int((first_seen>0).sum()),
        single_frame_neighbors=int((visible.sum(0)==1).sum()),complete8_neighbors=int(visible.all(0).sum()),
        near_motion_opportunities=int(attempted.sum()),blocked_unknown_velocity=int(blocked.sum()),
        legal_motion_opportunities=int(legal.sum()),initial_position_neighbors=int(near[0].sum()),
        last_step_legal_neighbors=int(legal[-1].sum()),has_near_history_neighbor=any_near,
        all_history_social_context_blocked=any_near and not available)


class CausalDataset(Dataset):
    def __init__(self,raw_root,manifest,split):
        self.manifest = json.load(open(manifest)) if isinstance(manifest,(str,Path)) else manifest
        if self.manifest['protocol'] != PROTOCOL:raise ValueError('Wrong protocol')
        self.windows = self.manifest['splits'][split]
        used = {w['recording'] for w in self.windows}
        if not used <= NON_ETH:raise ValueError('ETH or unregistered recording forbidden')
        self.records = {}
        for name in sorted(used):
            source = self.manifest['source_files'][name]
            file = Path(raw_root)/source['file']
            if sha256(file)!=source['sha256']:raise ValueError('Raw file hash mismatch')
            self.records[name] = Recording.read(file,source['scene'])
        self.contexts,self.indices = [],[]
        for wi,w in enumerate(self.windows):
            r = self.records[w['recording']]
            context = history_context(r,w['start_frame'])
            self.contexts.append(context)
            for i,actor in enumerate(context.ids):
                if context.position_valid[:,i].all() and target_future(r,w['start_frame'],actor)[1].all():
                    self.indices.append((wi,i))

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        wi,i = self.indices[index]
        w,c = self.windows[wi],self.contexts[wi]
        return example_from_context(c,self.records[w['recording']],w['start_frame'],int(c.ids[i]))

    def describe(self,index):
        wi,i = self.indices[index]
        c,w = self.contexts[wi],self.windows[wi]
        return dict(target_index=index,window_id=w['sample_id'],recording=w['recording'],target_id=int(c.ids[i]),
            start_frame=w['start_frame'],last_observed_frame=w['start_frame']+70,end_frame=w['end_frame'],
            **context_counts(c,i))


def collate_examples(examples,device='cpu'):
    max_neighbors = max(1,max(e['neighbor'].shape[1] for e in examples))
    neighbor = np.full((20,len(examples),max_neighbors,6),PADDING,np.float32)
    for i,e in enumerate(examples):neighbor[:,i,:e['neighbor'].shape[1]] = e['neighbor']
    x = np.stack([e['x'] for e in examples],axis=1)
    y = np.stack([e['y'] for e in examples],axis=1)
    return tuple(torch.tensor(a,device=device,dtype=torch.float32) for a in (x,y,neighbor))
