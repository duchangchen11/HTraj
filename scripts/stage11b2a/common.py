from pathlib import Path
import csv,gzip,hashlib,json,os,subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/stage11b2a'
CONF=ROOT/'configs/stage11b2a'
REPORT=ROOT/'reports/stage11b2a'
FIG=ROOT/'results/figures/stage11b2a'
OLD=ROOT/'results/tables/stage11b1c'
BASE='e85e676af6004c495aec77387c243ddf1060d13b'
CK='0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312'
PSHA='2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab'
PP=ROOT/'experiments/stage11b1c/cache/predictions/epoch030.pt'
CP=ROOT/'experiments/stage11b1c/cache/checkpoints/epoch_030.pt'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def protocol():return readj(CONF/'analysis_protocol.json')
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def cpu_guard():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
def frozen():
    cpu_guard();import torch
    assert readj(OUT/'data_cache_audit.json')['passed']
    assert sha(PP)==PSHA
    p=torch.load(PP,map_location='cpu',weights_only=True)
    assert p['checkpoint_SHA']==CK
    from scripts.stage11b1c.data import load,manifest
    df=load('validation');groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
    rows=manifest('validation');index={r['sample_id']:r for r in rows}
    import numpy as np
    h=np.stack([groups[int(index[s]['meta_id'])][['x','y']].iloc[:8].to_numpy(dtype=float) for s in p['sample_ids']])
    y=np.stack([groups[int(index[s]['meta_id'])][['x','y']].iloc[8:].to_numpy(dtype=float) for s in p['sample_ids']])
    return p,h,y,[index[s] for s in p['sample_ids']]
