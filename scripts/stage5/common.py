"""Bounded audit helpers; no forecasting network is implemented here."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/tables/stage5'
CACHE = ROOT / 'experiments/stage5/cache'
COMMITS = {
    'eqmotion': '5aec2e0b61c511fa93a24138dd90da59a089084b',
    'socialvae': 'eb23cbae562bac2c2dcf0d81529e6ab2ac280962',
    'sgcn': 'ec2ac7353ccba7c9691b00c0e978a0631667fa94',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def new_csv(path, rows):
    assert rows
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def git(repo, *args):
    return subprocess.check_output(['git', '--git-dir='+str(ROOT / 'experiments/stage5/repos' / (repo+'.git')), *args])


def export_blob(repo, commit, name, destination):
    """Export a pinned official blob, never replace an existing different asset."""
    path = Path(destination)
    content = git(repo, 'show', commit+':'+name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, path
    else:
        path.write_bytes(content)
    return dict(repository=repo, commit=commit, upstream_path=name,
                local_path=str(path.relative_to(ROOT)), bytes=len(content), sha256=sha(path))


def metrics(prediction, truth):
    """[K,T,B,2] -> independent per-target ADE/FDE for every candidate."""
    assert prediction.ndim == 4 and truth.ndim == 3
    assert prediction.shape[1:] == truth.shape
    error = (prediction-truth[None]).norm(dim=-1)
    return error.mean(dim=1), error[:, -1]


def check_non_eth_asset(name):
    allowed = {'models/hotel/ckpt-best', 'models/zara02/ckpt-best',
               'data/hotel/test/biwi_hotel.txt', 'data/zara02/test/crowds_zara02.txt',
               'data/zara02/train/crowds_zara01.txt'}
    if name not in allowed:
        raise ValueError('Asset is outside the fixed non-ETH allowlist: '+name)
