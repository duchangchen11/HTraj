from pathlib import Path
import csv,gzip,hashlib,json,os,subprocess,sys,time
from scripts.stage11b2e.common import sha,digest,csvnew,jnew,CP,CK,PP,PSHA,OFFICIAL,PYTHON
ROOT=Path(__file__).resolve().parents[2]
BASE='579e4f744dd06e2ec610debf2671ddcb0b3ea716'
OUT=ROOT/'results/tables/stage11c0';CONF=ROOT/'configs/stage11c0'
CACHE=ROOT/'experiments/stage11c0/cache';FIG=ROOT/'results/figures/stage11c0'
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def cpu_only():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
def parent_seconds():return sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('process_*.json'))
def checked(record):
    import torch
    p=ROOT/record['path'];assert sha(p)==record.get('sha256',record.get('file_SHA'))
    return torch.load(p,map_location='cpu',weights_only=True)
def cache_name(sid):return hashlib.sha256(sid.encode()).hexdigest()+'.pt'
class Budget:
    def __init__(self,prior=0,clock=time.perf_counter,started=None):
        self.prior=prior;self.clock=clock;self.start=clock() if started is None else started
    def elapsed(self):return self.prior+self.clock()-self.start
    def guard(self,next_seconds=30):
        if self.elapsed()+next_seconds>=6600:raise TimeoutError('6600s cumulative safety stop')
