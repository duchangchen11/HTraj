"""Independent Stage11A+ helpers; create evidence without overwriting history."""
from pathlib import Path
import csv, gzip, hashlib, json, os, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/tables/stage11ap'
CONF = ROOT / 'configs/stage11ap'
CACHE = ROOT / 'experiments/stage11ap/cache'
BASE = '36e76a3a1e339e5e42d7ebc465385a9d67744529'
MODEL_COMMIT = 'a345ada1557f53dc2acb4c07f71d340d366980da'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def jnew(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def cnew(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'xt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
