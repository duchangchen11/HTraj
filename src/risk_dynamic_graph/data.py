"""Historical-context adapter; local HTraj parsers are read-only dependencies.

No third-party model/loader implementation is copied. Qualification of targets
uses future labels separately; inference_inputs has no future-label argument.
"""
import numpy as np
import torch
from src.socialvae_causal.data import history_context


def inference_inputs(recording, start):
    c = history_context(recording, start)
    return dict(position=torch.from_numpy(c.state[None, ..., :2].copy()),
                valid=torch.from_numpy(c.position_valid[None].copy()),
                ids=torch.from_numpy(c.ids[None].copy()))


def target_supervision(recording, start, ids):
    """Labels only. Never use result to filter or build context nodes."""
    from src.socialvae_causal.data import target_future
    values = [target_future(recording, start, int(i)) for i in ids]
    y = np.stack([v[0] for v in values])
    mask = np.stack([v[1] for v in values])
    return y, mask
