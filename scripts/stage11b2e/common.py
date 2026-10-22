from pathlib import Path
import csv,gzip,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/stage11b2e'
CONF=ROOT/'configs/stage11b2e'
CACHE=ROOT/'experiments/stage11b2e/cache'
FIG=ROOT/'results/figures/stage11b2e'
OLD=ROOT/'results/tables/stage11b2c'
BASE='c3e5d09354130bc51d22bfa325342ae6a0831de6'
CP=ROOT/'experiments/stage11b1c/cache/checkpoints/epoch_030.pt'
CK='0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312'
PP=ROOT/'experiments/stage11b1c/cache/predictions/epoch030.pt'
PSHA='2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab'
OFFICIAL='a345ada1557f53dc2acb4c07f71d340d366980da'
PYTHON=ROOT/'experiments/stage11ap/cache/venv/bin/python'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(t):
    h=hashlib.sha256();h.update(b'tensor');h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes());return h.hexdigest()

def jnew(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def csvnew(p,rows):
    rows=list(rows);assert rows;p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'xt',newline='') if p.suffix=='.gz' else p.open('x',newline='')) as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def cpu_only():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()

def parent_seconds():return sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('process_*.json'))

class Budget:
    def __init__(self,prior=0,clock=time.perf_counter,started=None):
        self.prior=prior;self.clock=clock;self.start=clock() if started is None else started
    def elapsed(self):return self.prior+self.clock()-self.start
    def guard(self,next_seconds=30):
        if self.elapsed()+next_seconds>=3300:raise TimeoutError('3300s safety guard; no new target admitted')
