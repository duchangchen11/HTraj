from pathlib import Path
import csv, gzip, hashlib, json, subprocess, time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/tables/stage11b1'
CONF = ROOT / 'configs/stage11b1'
CACHE = ROOT / 'experiments/stage11b1/cache'
BASE = '620532a2b70f6cc8c6a26f9b0e37ad57565763e9'
from scripts.stage11b0.common import sha, jnew, cnew, commit, RES, OFFICIAL

def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
