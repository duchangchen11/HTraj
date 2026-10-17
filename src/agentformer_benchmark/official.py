from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5'
SOURCE = ROOT/'experiments/stage7/cache/AgentFormer'/COMMIT
DATA = ROOT/'data/raw/agentformer_official'/COMMIT
CACHE = ROOT/'experiments/stage7p2/cache'


def install_compatibility():
    sys.path[:0] = [str(ROOT/'experiments/stage7/cache/deps'), str(SOURCE)]
    import torch.nn.modules.linear as linear
    if not hasattr(linear, '_LinearWithBias'):
        linear._LinearWithBias = torch.nn.Linear
    if 'int' not in np.__dict__:
        np.int = int  # Removed public dtype alias used by the official loader.


def configuration(fold, stage='pre'):
    from easydict import EasyDict
    name = f'{fold}_agentformer'+('_pre' if stage == 'pre' else '')
    file = SOURCE/'cfg/eth_ucy'/fold/(name+'.yml')
    cfg = EasyDict(yaml.safe_load(file.read_text()))
    cfg.data_root_ethucy = str(DATA)
    cfg.results_root_dir = str(CACHE/'runtime/results')
    return cfg


def read_official_file(path):
    rows = np.genfromtxt(path, dtype=str, ndmin=2)
    if rows.shape[1] != 17 or not np.all(rows[:, 2] == 'Pedestrian'):
        raise ValueError('Unexpected official 17-column pedestrian format')
    a = rows[:, [0, 1, 13, 15]].astype(np.float64)
    if not np.isfinite(a).all() or not np.equal(a[:, :2], np.round(a[:, :2])).all():
        raise ValueError('Nonfinite positions or nonintegral IDs/frame grid')
    if len(np.unique(a[:, :2], axis=0)) != len(a):
        raise ValueError('Duplicate frame/actor pair')
    return a


def enumerate_windows(a):
    """Independent fast enumeration of the exact official min8/min12 filter."""
    by_frame = {}
    for f, i, x, y in a:
        by_frame.setdefault(int(f), set()).add(int(i))
    first, last = int(a[:, 0].min()), int(a[:, 0].max())
    windows = []
    for frame in range(first+7, last-12+1):
        eligible = set.intersection(*(by_frame.get(t, set()) for t in range(frame-7, frame+13)))
        if eligible:
            windows.append(dict(last_observed_frame=frame, start_frame=frame-7, end_frame=frame+12,
                people=len(eligible), target_ids=sorted(eligible)))
    return windows
