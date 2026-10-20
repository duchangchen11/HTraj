from pathlib import Path
import csv,gzip,hashlib,json,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/stage11b0'
CONF=ROOT/'configs/stage11b0'
CACHE=ROOT/'experiments/stage11b0/cache'
RES=ROOT/'experiments/stage11ap/cache/author_resources/ynet_additional_files'
BASE='fba0e46e61cf271f16cff4b03a98eb788182753c'
OFFICIAL='a345ada1557f53dc2acb4c07f71d340d366980da'
PYTHON=ROOT/'experiments/stage11ap/cache/venv/bin/python'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def jnew(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cnew(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'xt',newline='') if p.suffix=='.gz' else p.open('x',newline='')) as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
