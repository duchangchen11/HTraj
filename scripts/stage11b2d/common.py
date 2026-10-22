"""Stage-local I/O with an explicit CPU-only execution boundary."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/tables/stage11b2d'
CONF = ROOT / 'configs/stage11b2d'
FIG = ROOT / 'results/figures/stage11b2d'
CACHE = ROOT / 'experiments/stage11b2d/cache'
OLD = ROOT / 'results/tables/stage11b2c'
BASE = '7ddcd1cc88433e46a2e0c519d9ae78b1b4fe6e22'
CK = '0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312'
PSHA = '2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab'
CP = ROOT / 'experiments/stage11b1c/cache/checkpoints/epoch_030.pt'
PP = ROOT / 'experiments/stage11b1c/cache/predictions/epoch030.pt'
KM = ROOT / 'experiments/stage11a/cache/official/ynet/utils/kmeans.py'

def cpu_guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES to empty'
    import torch
    assert not torch.cuda.is_initialized()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

def jnew(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def csvnew(path, rows):
    rows = list(rows)
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'xt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def tensor_sha(t):
    # Match historical state_digest({'tensor': tensor}) framing exactly.
    h=hashlib.sha256()
    h.update(b'tensor');h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
