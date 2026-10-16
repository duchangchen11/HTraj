"""History-only node selection, with labels kept out of the input contract."""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from src.socialvae_causal.data import target_future

PROTOCOL = 'HTraj-AgentFormer-Causal8x12-v1'


class UnlabeledContextError(ValueError):
    """The frozen joint posterior has no history-only training node type."""


@dataclass
class CausalContext:
    recording: str
    start: int
    ids: np.ndarray
    history: np.ndarray
    partial_ids: np.ndarray
    support: tuple


def build_context(recording, start):
    """Read exactly eight frames. Future existence cannot select any node."""
    if recording.frame_step != 10 or recording.dt != .4:
        raise ValueError('Expected frozen 10-tick/0.4s sampling')
    support = tuple(int(start) + 10*t for t in range(8))
    raw = [recording.by_frame.get(f, {}) for f in support]
    ids = np.array(sorted(set.intersection(*(set(r) for r in raw))), dtype=np.int64)
    partial = np.array(sorted(set.union(*(set(r) for r in raw)) - set(ids)), dtype=np.int64)
    history = np.empty((len(ids), 8, 2), dtype=np.float32)
    for i, actor in enumerate(ids):
        history[i] = np.stack([r[int(actor)] for r in raw])
    if not np.isfinite(history).all():
        raise ValueError('Nonfinite observed position')
    return CausalContext(recording.name, int(start), ids, history, partial, support)


def labels_for_context(context, recording):
    """Missing coordinates stay NaN, never a fabricated normal label."""
    y = np.full((len(context.ids), 12, 2), np.nan, dtype=np.float32)
    mask = np.zeros((len(context.ids), 12), dtype=bool)
    for i, actor in enumerate(context.ids):
        observed, valid = target_future(recording, context.start, actor)
        y[i, valid] = observed[valid]
        mask[i] = valid
    return y, mask


def inference_packet(context):
    if not len(context.ids):
        raise ValueError('No complete-eight-frame context')
    n = len(context.ids)
    return dict(pre_motion_3D=[torch.from_numpy(h.copy()) for h in context.history],
        pre_motion_mask=[torch.ones(8) for _ in range(n)],
        fut_motion_3D=[torch.full((12, 2), float('nan')) for _ in range(n)],
        fut_motion_mask=[torch.zeros(12) for _ in range(n)], heading=None)


def original_training_packet(context, recording):
    """Fail closed. No masks, posterior, losses or core modules are rewritten."""
    y, valid = labels_for_context(context, recording)
    if not valid.all():
        missing = context.ids[~valid.all(1)].tolist()
        raise UnlabeledContextError(f'Original joint posterior/KL cannot consume unlabeled context IDs {missing}')
    packet = inference_packet(context)
    packet['fut_motion_3D'] = [torch.from_numpy(v.copy()) for v in y]
    packet['fut_motion_mask'] = [torch.ones(12) for _ in y]
    return packet


def stable_epsilon(context, k, seed=17):
    """Latent epsilon coupled to stable scene/actor IDs, not batch order."""
    eps = []
    for actor in context.ids:
        identity = f'{seed}:{context.recording}:{context.start}:{int(actor)}'
        s = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
        eps.append(np.random.default_rng(s).standard_normal((k, 32)).astype(np.float32))
    return torch.from_numpy(np.stack(eps, axis=1))
