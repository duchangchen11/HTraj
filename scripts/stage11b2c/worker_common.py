"""GPU helpers that never load future coordinates, labels, or error tables."""
from .common import *
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics
from .adapter import AuditPredictor,digest
import torch,os

def initialize():
    configure_torch();assert torch.cuda.is_available()
    f=json.loads((OUT/'protocol_frozen.json').read_text())
    for r in f['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(CONF/'analysis_protocol.json')==f['analysis_protocol_SHA'] and sha(CP)==CK
    net,cfg=build_model(17,'cuda:0');saved=torch.load(CP,map_location='cpu',weights_only=True)
    assert saved['official_commit']==OFFICIAL
    net.load_state_dict(saved['model'],strict=True);net.eval()
    assert state_digest(net.state_dict())==state_digest(saved['model']);del saved
    initial=state_digest(net.state_dict());templates=make_templates(cfg,'cuda:0')
    budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    return net,cfg,templates,AuditPredictor(net,cfg,templates),budget,initial

def histories(name):
    r=json.loads((OUT/'input_export.json').read_text())['artifacts'][name];p=ROOT/r['path'];assert sha(p)==r['sha256']
    pack=torch.load(p,map_location='cpu',weights_only=True);assert set(pack)=={'sample_ids','scenes','history'}
    assert pack['history'].shape==(len(pack['sample_ids']),8,2)
    return pack

def semantic(net,cfg,scene,cache,cache_to_disk=False):
    if scene not in cache:
        im=preprocess_scene(scene,cfg,'cuda:0');s=fixed_semantics(net,im).cpu();cache[scene]=s
        if cache_to_disk:
            p=CACHE/('semantic_'+scene+'.pt');assert not p.exists();torch.save(s,p)
    return cache[scene].cuda()

def same(a,b):
    assert set(a)==set(b)
    assert all(torch.equal(a[k],b[k]) for k in a),[k for k in a if not torch.equal(a[k],b[k])]

def cache_name(sid):return hashlib.sha256(sid.encode()).hexdigest()+'.pt'
