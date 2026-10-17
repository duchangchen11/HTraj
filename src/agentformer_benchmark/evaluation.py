"""Independent reference arithmetic for checking the official evaluator."""
import numpy as np


def target_metrics(prediction, truth):
    p, y = np.asarray(prediction), np.asarray(truth)
    if p.ndim != 4 or p.shape[0] != y.shape[0] or p.shape[2:] != y.shape[1:] or p.shape[1] != 20:
        raise ValueError('Expected N,20,12,2 predictions paired with N,12,2 GT')
    if p.shape[2:] != (12, 2) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid trajectory values/shape')
    error = np.linalg.norm(p-y[:, None], axis=-1)
    ade, fde = error.mean(-1), error[:, :, -1]
    return dict(minADE20=ade.min(1), minFDE20=fde.min(1),
        K1_first_ADE=ade[:, 0], K1_first_FDE=fde[:, 0],
        ADE_winner=ade.argmin(1), FDE_winner=fde.argmin(1))


def aggregate_targets(groups):
    return {name: float(np.concatenate([g[name] for g in groups]).mean())
        for name in ['minADE20', 'minFDE20', 'K1_first_ADE', 'K1_first_FDE']}
