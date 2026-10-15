"""Append-only stage artifacts and parameter fingerprints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def json_new(path, obj):
    text=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(text)


def state_info(model):
    digest=hashlib.sha256();norm=0.;maximum=0.
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        if not np.isfinite(a).all():raise FloatingPointError('Nonfinite model parameter '+name)
        digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
        norm+=float(np.square(a.astype(np.float64)).sum())
        maximum=max(maximum,float(np.abs(a).max()))
    return dict(parameter_state_sha256=digest.hexdigest(),parameter_L2=norm**.5,parameter_max_abs=maximum,
                parameters_finite=True,parameter_count=sum(p.numel() for p in model.parameters()))
