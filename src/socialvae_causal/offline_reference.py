"""Unchanged official offline behavior, reference only, never called training."""
import inspect
import sys
import numpy as np
import torch
from .official import load_official


def official_examples(path):
    cls = load_official('data.py').Dataloader
    d = object.__new__(cls)
    d.ob_horizon,d.pred_horizon,d.horizon,d.frameskip = 8,12,20,1
    d.device,d.batch_first = torch.device('cpu'),False
    d.flip = d.rotate = d.scale = False
    d.rng = np.random.RandomState(17)
    keys = []
    source,start = inspect.getsourcelines(cls.load)
    append_line = next(start+i for i,line in enumerate(source) if 'traj.append((hist,' in line)
    def trace(frame,event,arg):
        if frame.f_code != cls.load.__code__:return None
        if event=='line' and frame.f_lineno==append_line:
            v=frame.f_locals
            actors=[int(a) for a in v['idx']]+[int(a) for a in v['neighbor_idx']]
            if len(actors)<v['agents'].shape[1]:actors.append(None)
            keys.append(dict(start_frame=int(v['time'][v['tid0']]),target_id=int(v['idx'][v['i']]),
                neighbor_ids=[a for j,a in enumerate(actors) if j!=v['i']]))
        return trace
    sys.settrace(trace)
    try:items=cls.load(d,str(path),[])
    finally:sys.settrace(None)
    assert items is not None and len(keys)==len(items)
    return items,keys
