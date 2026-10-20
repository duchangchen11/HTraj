from pathlib import Path
import csv,gzip,hashlib,json,subprocess,time,sys
from scripts.stage11b0.common import sha,jnew,cnew,commit,RES,OFFICIAL
from scripts.stage11b1.common import configure_torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/stage11b1c'
CONF=ROOT/'configs/stage11b1c'
CACHE=ROOT/'experiments/stage11b1c/cache'
FIG=ROOT/'results/figures/stage11b1c'
BASE='ff56783d5b4f14f8a81074510b4bd7226245bb56'

def replace_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    temp.replace(path)

def parent_seconds():
    return sum(json.loads(p.read_text())['wall_seconds'] for p in OUT.glob('process_*.json'))

class Budget:
    def __init__(self,prior=0,clock=time.perf_counter,started=None):
        self.prior=prior;self.clock=clock;self.start=clock() if started is None else started
    def elapsed(self):return self.prior+self.clock()-self.start
    def guard(self,next_seconds=30):
        if self.elapsed()+next_seconds>=27000:raise TimeoutError('27000s cumulative safe-stop before next operation')

def open_csv(path,fields):
    f=gzip.open(path,'xt',newline='') if path.suffix=='.gz' else path.open('x',newline='')
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();f.flush()
    return f,w
