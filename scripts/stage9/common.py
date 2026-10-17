"""Stage9 controlled Re pilot. Vendor code remains frozen.
Adapted Resonance aggregation logic: Conghao Wong, cocoon2wong/Re
7358b3d4fd3fcf160c18be9425445a67ee8211ee; GPL-3.0, see LICENSE_NOTICE.md.
"""
import csv,gzip,hashlib,json,os,plistlib,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.agentformer_benchmark.official import DATA,read_official_file,enumerate_windows
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
OUT=ROOT/'results/tables/stage9'
CACHE=ROOT/'experiments/stage9/cache'
SOURCE=ROOT/'experiments/stage8/cache/Re'
BASE='d9bfed15abf32c9004b82bc41862f20f955dd8a2'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest()
def seed(s):
    import random
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def setup():
    # Metadata only; never invoke the Qpid author data loader or train/test split.
    directory=CACHE/'runtime/dataset_configs/Stage9Metadata';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'controlled_pilot.plist'
    if not path.exists():
        with path.open('xb') as f:plistlib.dump(dict(dataset='Stage9Metadata',type='meter',scale=1.,scale_vis=1.,dimension=2,anntype='coordinate',train=[],val=[],test=[]),f)
    os.chdir(CACHE/'runtime');sys.path.insert(0,str(SOURCE))
    import qpid,resonance
    return qpid,resonance

def model(kind,device='cpu'):
    qpid,resonance=setup()
    from qpid.args import Args
    seed(42)
    args=Args(['main.py','--model','re','--split','controlled_pilot','--feature_dim','128',
        '--partitions','8','--T','haar','--Tr','haar','--output_pred_steps','2_5_8_11',
        '--preprocess','100','--gpu','0','--max_agents','57','--K','1','--K_train','1'])
    structure=resonance.ResonanceStructure(args);structure.create_model();m=structure.model
    if kind!='A':
        from resonance._resonance import ResonanceLayer
        from qpid.utils import get_mask
        class PairedLayer(ResonanceLayer):
            def __init__(self,original):
                super().__init__(8,128,128,original.T_layer)
                self.load_state_dict(original.state_dict(),strict=True)
                with torch.random.fork_rng():
                    torch.manual_seed(4201)
                    if kind=='B':
                        self.joint=torch.nn.Sequential(torch.nn.Linear(128,32,bias=False),torch.nn.Tanh(),torch.nn.Linear(32,128,bias=False))
                    else:
                        self.pos=torch.nn.Linear(64,32,bias=False)
                        self.motion=torch.nn.Linear(64,32,bias=False)
                        self.cross=torch.nn.Linear(32,128,bias=False)
            def forward(self,x_ego_2d,x_nei_2d):
                # Run the exact frozen native operator to retain its original pooled representation.
                original,h=super().forward(x_ego_2d,x_nei_2d)
                p=x_nei_2d[...,-1,:];distance=torch.linalg.vector_norm(p,dim=-1)
                angle=torch.atan2(p[...,0],p[...,1])%(2*np.pi)
                valid=get_mask(x_nei_2d.sum(dim=(-1,-2)),torch.int32).bool()
                sectors=(angle/(2*np.pi/8)).to(torch.int64)
                g=self.ce(torch.stack([distance,angle],-1))
                deltas=[]
                if kind=='B':joint=self.joint(torch.cat([g,h],-1))
                else:u=self.pos(g);v=self.motion(h)
                for part in range(8):
                    mask=(valid&(sectors==part)).to(h.dtype)[...,None]
                    count=mask.sum(-2);denom=count.clamp_min(1.)
                    if kind=='B':delta=(joint*mask).sum(-2)/denom
                    else:
                        # Diagonal low-rank projection of the centered cross moment.
                        eu=(u*mask).sum(-2)/denom;ev=(v*mask).sum(-2)/denom
                        moment=(u*v*mask).sum(-2)/denom-eu*ev
                        delta=self.cross(moment)*(count>1)
                    deltas.append(delta)
                return original+torch.stack(deltas,-2),h
        m.rc=PairedLayer(m.rc)
    return m.to(device)

def loss(pred,gt):
    from qpid.training.loss.__ade import ADE_2D
    return ADE_2D(pred,gt)

def source_name(sequence):return sequence.removesuffix('_train').removesuffix('_val')

def build_data():
    files={};arrays={};frozen=json.loads((ROOT/'results/tables/stage7p2/official_data_audit.json').read_text())
    expected={x['path']:x['sha256'] for x in frozen['official_files']}
    for split in ['train','val']:
        for p in sorted((DATA/'hotel').glob('*_'+split+'.txt')):
            relative=str(p.relative_to(ROOT));assert sha(p)==expected[relative]
            arrays[p.stem]=read_official_file(p)
            files[p.stem]=dict(path=relative,sha256=sha(p),split=split)
    assert len(files)==14
    isolation=[]
    for name in sorted({source_name(k) for k in arrays}):
        a=arrays[name+'_train'];b=arrays[name+'_val']
        fa=set(a[:,0].astype(int));fb=set(b[:,0].astype(int))
        ka=set(map(tuple,a[:,:2].astype(int)));kb=set(map(tuple,b[:,:2].astype(int)))
        # Native frames retain source numbering. Fingerprint screen also catches reindexing.
        pa={tuple(row) for row in a[:,[1,2,3]]};pb={tuple(row) for row in b[:,[1,2,3]]}
        isolation.append(dict(source=name,train_min=int(a[:,0].min()),train_max=int(a[:,0].max()),val_min=int(b[:,0].min()),val_max=int(b[:,0].max()),shared_frames=len(fa&fb),shared_frame_actor=len(ka&kb),same_actor_xy=len(pa&pb)))
        if fa&fb or ka&kb:raise RuntimeError('Unexplained train/val temporal overlap '+name)
    json_new(OUT/'source_isolation.json',dict(passed=True,sources=isolation,limits='Same recording and adjacent time regions; no independent recording confirmation. Same actor/position recurrence alone is not shared frame support. No held-out HOTEL recording opened.'))
    manifest={};allrows=[]
    from qpid.utils import INIT_POSITION
    for split in ['train','val']:
        histories=[];futures=[];neighbors=[];rows=[];nwindows=0;maxn=0
        for sequence,a in arrays.items():
            if not sequence.endswith('_'+split):continue
            positions={(int(f),int(i)):np.array([x,y],dtype=np.float32) for f,i,x,y in a}
            for w in enumerate_windows(a):
                f=w['last_observed_frame'];ids=w['target_ids'];n=len(ids);maxn=max(maxn,n);nwindows+=1
                hist=np.array([[positions[t,i] for t in range(f-7,f+1)] for i in ids],dtype=np.float32)
                gt=np.array([[positions[t,i] for t in range(f+1,f+13)] for i in ids],dtype=np.float32)
                velocity=(hist[:,-1]-hist[:,0])/(7*.4)
                for index,person in enumerate(ids):
                    relative=hist[:,-1]-hist[index,-1];r=np.linalg.norm(relative,axis=-1)
                    angles=np.arctan2(relative[:,0],relative[:,1])%(2*np.pi);part=np.floor(angles/(2*np.pi/8)).astype(int)
                    pairs=0
                    for j in range(n):
                        if j==index:continue
                        for k in range(j+1,n):
                            if k!=index and part[j]==part[k] and abs(float(r[j]-r[k]))>=.5 and np.linalg.norm(velocity[j]-velocity[k])>=.3:pairs+=1
                    row=dict(split=split,sequence=sequence,source=source_name(sequence),frame=f,person_id=person,
                        history_SHA=digest(hist[index]),GT_SHA=digest(gt[index]),context_people=n,external_neighbors=n-1,
                        ambiguity=pairs>0,qualifying_pairs=pairs,speed_m_s=float(np.linalg.norm(velocity[index])))
                    rows.append(row);histories.append(hist[index]);futures.append(gt[index])
                    padded=np.full((57,8,2),INIT_POSITION,dtype=np.float32)
                    assert n<=57;padded[:n]=hist-hist[index,-1]
                    neighbors.append(padded)
        csv_new(OUT/(split+'_targets.csv.gz'),rows)
        torch.save(dict(hist=torch.from_numpy(np.array(histories)),gt=torch.from_numpy(np.array(futures)),nei=torch.from_numpy(np.array(neighbors))),CACHE/(split+'_data.pt'))
        manifest[split]=dict(windows=nwindows,targets=len(rows),max_context=maxn,manifest_SHA=sha(OUT/(split+'_targets.csv.gz')),cache_SHA=sha(CACHE/(split+'_data.pt')))
        allrows+=rows
    assert manifest['train']['targets']==29676 and manifest['val']['targets']==5203
    json_new(OUT/'data_manifest.json',dict(files=files,splits=manifest,eligibility='Official complete20 IDs as both targets and contexts; all N, ego included in native Re context, padded to57 without truncation; context positions relative to target last history. No future coordinates in model arguments.',history_grid=.4))

def dataset(split,device='cuda'):
    p=CACHE/(split+'_data.pt');manifest=json.loads((OUT/'data_manifest.json').read_text())
    assert sha(p)==manifest['splits'][split]['cache_SHA']
    return {k:v.to(device) for k,v in torch.load(p,map_location='cpu',weights_only=True).items()}

def batch(data,idx):return [data['hist'][idx],data['nei'][idx]],data['gt'][idx]
def params(m):return sum(p.numel() for p in m.parameters())
def smi():
    x=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return x.stdout.strip()
def env():return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),device_memory_bytes=torch.cuda.get_device_properties(0).total_memory)

def gpu_ledger(name,seconds):
    # Every GPU task is timed including CPU I/O while owning its CUDA context (conservative).
    p=OUT/'gpu_usage.csv';exists=p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','wall_seconds']);
        if not exists:w.writeheader()
        w.writerow(dict(task=name,wall_seconds=seconds));f.flush()
    return sum(float(r['wall_seconds']) for r in csv.DictReader(p.open()))
def used():
    p=OUT/'gpu_usage.csv'
    return sum(float(r['wall_seconds']) for r in csv.DictReader(p.open())) if p.exists() else 0.

def prepare_protection():
    files={}
    with gzip.open(ROOT/'configs/stage8/protected_files.csv.gz','rt') as f:
        for r in csv.DictReader(f):files[r['path']]=dict(path=r['path'],sha256=r['sha256'])
    for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'):
        if p:files[p]=dict(path=p,sha256=sha(ROOT/p))
    with gzip.open(ROOT/'results/tables/stage8/source_and_weight_SHA256.csv.gz','rt') as f:
        for r in csv.DictReader(f):files[r['path']]=dict(path=r['path'],sha256=r['sha256'])
    csv_new(ROOT/'configs/stage9/protected_files.csv.gz',sorted(files.values(),key=lambda r:r['path']))
    json_new(OUT/'source_versions.json',dict(repositories=[dict(path=str(p.relative_to(ROOT)),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),status=subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)) for p in [SOURCE,SOURCE/'qpid',SOURCE/'dataset_original']],base=BASE,environment=env()))

if __name__=='__main__':
    torch.set_num_threads(1);setup();prepare_protection();build_data()
