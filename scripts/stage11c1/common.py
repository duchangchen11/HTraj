from pathlib import Path
import csv,gzip,hashlib,json,os,subprocess,sys,time
from scripts.stage11b0.common import sha,jnew,cnew,commit,RES,OFFICIAL
from scripts.stage11b1.common import configure_torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/stage11c1';CONF=ROOT/'configs/stage11c1'
CACHE=ROOT/'experiments/stage11c1/cache';FIG=ROOT/'results/figures/stage11c1'
BASE='6b81cf77222a3be1c89686a3b39abe9acc22e669'
INITIAL='0992e8b31410e14ca141902b201ce4e061c85fd31668c74b4e1944a2656169e9'
from scripts.stage11b1c.common import open_csv,replace_json

def parent_seconds():
    return sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('process_*.json'))

class Budget:
    def __init__(self,prior=0,clock=time.perf_counter,started=None):
        self.prior=prior;self.clock=clock;self.start=clock() if started is None else started
    def elapsed(self):return self.prior+self.clock()-self.start
    def guard(self,next_seconds=30,reserve=120):
        if next_seconds<0:raise ValueError('Nonnegative prediction required')
        if self.elapsed()+next_seconds+reserve>=16800:
            raise TimeoutError('STOPPED_BUDGET: next complete operation plus120s save/exit reserve exceeds16800s')

def shapes():return json.loads((CONF/'scene_shapes.json').read_text())['scenes']

def zero_semantic(scene,device,table=None):
    import torch
    row=(shapes() if table is None else table)[scene]
    # The data audit freezes the author resize/pad canvas, not a new image transform.
    shape=row['semantic_shape']
    out=torch.zeros(shape,dtype=torch.float32,device=device)
    assert out.shape[0:2]==(1,6) and not torch.count_nonzero(out)
    return out

def history_records(scope='validation'):
    import torch
    assert scope in {'validation','preflight'}
    name='validation_history_only.pt' if scope=='validation' else 'precheck_history_only.pt'
    p=ROOT/'experiments/stage11b2c/cache'/name
    expected={'validation':'b484f5b320908e39ad56f91baf677f000afc8328984f31e72ab98f3a0e53c495',
              'preflight':'5d9b414569a957cc504246a1089729915f3f2ee7f864ebcc935921c6f260905c'}[scope]
    assert sha(p)==expected
    a=torch.load(p,map_location='cpu',weights_only=True)
    assert set(a)=={'sample_ids','scenes','history'}
    return [dict(sample_id=s,scene=c,history=h) for s,c,h in zip(a['sample_ids'],a['scenes'],a['history'])]
