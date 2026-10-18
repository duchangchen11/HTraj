"""Independent Stage11A evidence helpers; all writes exclusive/new-stage only."""
from pathlib import Path
import csv, gzip, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/tables/stage11a'
CONF=ROOT/'configs/stage11a'
CACHE=ROOT/'experiments/stage11a/cache'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def jnew(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def cnew(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'xt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def cread(p):
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt') as f:return list(csv.DictReader(f))

def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
