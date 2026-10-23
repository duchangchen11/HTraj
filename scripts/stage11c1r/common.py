from pathlib import Path
import csv,gzip,hashlib,json,os,subprocess,sys,time
from scripts.stage11b0.common import sha,jnew,cnew,commit,RES,OFFICIAL
from scripts.stage11b1.common import configure_torch
from scripts.stage11b1c.common import open_csv,replace_json
from scripts.stage11c1.common import INITIAL,zero_semantic,shapes,history_records
ROOT=Path(__file__).resolve().parents[2];BASE='29fe2036c034cd0701e672be69f448b401ee76dd'
OUT=ROOT/'results/tables/stage11c1r';CONF=ROOT/'configs/stage11c1r'
CACHE=ROOT/'experiments/stage11c1r/cache';FIG=ROOT/'results/figures/stage11c1r'
ORIGINAL_USED=66.67494837602135

def parent_seconds():return sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('process_*.json'))

class Budget:
    def __init__(self,prior=0,started=None,clock=time.perf_counter):
        self.prior=prior;self.clock=clock;self.start=clock() if started is None else started
    def elapsed(self):return self.prior+self.clock()-self.start
    def guard(self,next_seconds,reserve=60):
        if next_seconds<0:raise ValueError('Negative cost')
        spent=self.elapsed()
        if spent+next_seconds+reserve>=2250 or ORIGINAL_USED+spent+next_seconds+reserve>=18000:
            raise TimeoutError('STOPPED_BUDGET: no new complete stage/operation admitted')

def verify_frozen():
    f=json.loads((OUT/'protocol_frozen.json').read_text())
    for row in f['source_files']+f['input_files']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    return f

def selected_records():
    allowed=list(csv.DictReader((OUT/'diagnostic_sample_manifest.csv').open()))
    index={r['sample_id']:r for r in history_records('validation')}
    assert len(allowed)==15 and len({r['sample_id'] for r in allowed})==15
    return [index[r['sample_id']] for r in allowed]

def rng_digest():
    import random,numpy as np,torch
    n=np.random.get_state()
    value=dict(python=repr(random.getstate()),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
        torch=sha_bytes(torch.get_rng_state().numpy().tobytes()),
        cuda=[sha_bytes(t.cpu().numpy().tobytes()) for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])
    return sha_bytes(json.dumps(value,sort_keys=True).encode())

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
