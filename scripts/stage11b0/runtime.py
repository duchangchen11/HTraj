"""Frozen MIT Y-Net import and initialization, with no predictor checkpoint input."""
from .common import *
import importlib.util,types,shutil,random
import numpy as np,torch,yaml
from scripts.stage11ap.checkpoint_io import load_author_segmentation

_modules=None
def original_modules():
    global _modules
    if _modules is not None:return _modules
    vendor=CACHE/'official/ynet'
    source=ROOT/'experiments/stage11ap/cache/official/ynet'
    if not vendor.exists():
        vendor.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,vendor,ignore=shutil.ignore_patterns('__pycache__'))
    manifest=json.loads((ROOT/'results/tables/stage11ap/frozen_source_manifest.json').read_text())
    for r in manifest['files']:assert sha(CACHE/'official'/r['path'])==r['sha256']
    # Isolate author's absolute utils/train/test imports from other project baselines.
    names=lambda n:n=='utils' or n.startswith('utils.') or n in {'model','test','train'}
    saved={k:v for k,v in sys.modules.items() if names(k)};old_path=sys.path.copy()
    try:
        for k in list(saved):sys.modules.pop(k,None)
        sys.path.insert(0,str(vendor))
        import importlib
        model=importlib.import_module('model')
        image=importlib.import_module('utils.image_utils')
        prep=importlib.import_module('utils.preprocessing')
        _modules=types.SimpleNamespace(model=model,image=image,preprocess=prep)
    finally:
        for k in list(sys.modules):
            if names(k):sys.modules.pop(k,None)
        sys.modules.update(saved);sys.path[:]=old_path
    return _modules

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def build_model(seed=17,device='cpu'):
    seed_all(seed)
    cfg=yaml.safe_load((RES/'config/sdd_trajnet.yaml').read_text())
    segpath=RES/cfg['segmentation_model_fp']
    assert sha(segpath)=='75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194'
    seg,_=load_author_segmentation(segpath)
    # Constructor initializes the entire goal/path core from scratch. Never load predictor weights.
    modules=original_modules();seed_all(seed)
    net=modules.model.YNetTorch(8,12,None,False,6,cfg['encoder_channels'],cfg['decoder_channels'],1)
    net.semantic_segmentation=seg
    for p in seg.parameters():p.requires_grad_(False)
    set_training(net)
    return net.to(device),cfg

def set_training(net):
    net.train();net.semantic_segmentation.eval()

def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def save_checkpoint(path,net,opt,progress,config):
    assert not path.exists()
    t=time.perf_counter()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),progress=progress,config=config,
                    official_commit=OFFICIAL,source_commit=commit(),scheduler=None,scheduler_reason='Original Y-Net constant Adam learning rate; no scheduler'),path)
    location=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return dict(path=location,sha256=sha(path),bytes=path.stat().st_size,save_and_hash_seconds=time.perf_counter()-t)

def load_checkpoint(path,net,opt):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    result=net.load_state_dict(saved['model'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    return saved
