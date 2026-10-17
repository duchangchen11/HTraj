import hashlib
import json
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from src.agentformer_benchmark.official import ROOT, SOURCE, DATA, install_compatibility
from src.agentformer_official_evaluation.runtime import digest

OUT = ROOT/'results/tables/stage7p4/execution_v2'
CACHE = ROOT/'experiments/stage7p4/cache/execution_v2'
WORK = CACHE/'training_workspace'


def protocol():
    return json.loads((ROOT/'configs/stage7p4/protocol.json').read_text())


def verify_inputs():
    p = protocol()
    for row in p['frozen_inputs']:
        assert digest(ROOT/row['path']) == row['sha256'], row['path']
    for path, sha in p['execution_source_SHA256'].items():
        assert digest(ROOT/path) == sha, path
    return dict(files=len(p['frozen_inputs']), execution_files=len(p['execution_source_SHA256']), passed=True)


def freeze_gate():
    def git(*args):
        return subprocess.check_output(['git', *args],cwd=ROOT,text=True).strip()
    assert git('branch','--show-current') == 'feat/stage7p4-agentformer-training-feasibility'
    assert not git('status','--porcelain','--untracked-files=all'), 'Commit all inputs before execution'
    commit=git('rev-parse','HEAD')
    assert commit != protocol()['base_commit']
    paths=list(protocol()['execution_source_SHA256'])+['configs/stage7p4/protocol.json']
    paths += [r['path'] for r in protocol()['frozen_inputs'] if r['path'].startswith('configs/stage7p4/')]
    for path in paths:
        assert hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)).hexdigest()==digest(ROOT/path)
    return commit


def workspace():
    WORK.mkdir(parents=True,exist_ok=True)
    links={WORK/'cfg': ROOT/'configs/stage7p4/training',WORK/'datasets/eth_ucy':DATA}
    for dest,source in links.items():
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(source,target_is_directory=True)
        assert dest.resolve()==source.resolve()
    return WORK


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu'])
    if s['torch_cuda']:torch.cuda.set_rng_state_all(s['torch_cuda'])


def nested_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b
