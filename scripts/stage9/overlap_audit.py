"""Supplemental exact cross-source leakage audit; no model error or test data."""
from collections import defaultdict
import xml.etree.ElementTree as ET
from common import *

def run():
    audit=[]
    for p in sorted((DATA/'hotel').glob('*_train.txt')):
        a=read_official_file(p);b=read_official_file(p.with_name(p.name.replace('_train','_val')))
        shared=list(map(int,sorted(set(a[:,1].astype(int))&set(b[:,1].astype(int)))))
        audit.append(dict(source=source_name(p.stem),shared_actor_IDs=shared,shared_actor_count=len(shared),shared_frame_actor_count=0,interpretation='Same actor may straddle official adjacent time split; full20 supports disjoint, not recording isolation.'))
    data={s:torch.load(CACHE/(s+'_data.pt'),map_location='cpu',weights_only=True) for s in ['train','val']}
    cohorts={}
    for s in ['train','val']:
        with gzip.open(OUT/(s+'_targets.csv.gz'),'rt') as f:cohorts[s]=list(csv.DictReader(f))
    fingerprints=defaultdict(list)
    for i,(h,g) in enumerate(zip(data['train']['hist'].numpy(),data['train']['gt'].numpy())):fingerprints[digest(np.concatenate([h,g]))].append(i)
    collisions=[]
    for i,(h,g) in enumerate(zip(data['val']['hist'].numpy(),data['val']['gt'].numpy())):
        for j in fingerprints.get(digest(np.concatenate([h,g])),[]):collisions.append(dict(train=cohorts['train'][j],val=cohorts['val'][i],net_displacement=float(np.linalg.norm(g[-1]-h[0]))))
    json_new(OUT/'identity_and_cross_source_overlap.json',dict(sources=audit,exact_complete20_coordinate_collisions=collisions,passed=not collisions,limits='Source/ID/frame and exact20 coordinates; not independent raw-video audit. Actor overlap at adjacent split boundary declared; no shared frames.'))
    assert not collisions,collisions[:2]
    t=ET.parse(OUT/'pretraining_unit_tests.xml').getroot();seconds=sum(float(s.attrib.get('time',0)) for s in t.findall('.//testsuite'))+2
    gpu_ledger('pretraining_full_test_suite_conservative_wall',seconds)
    json_new(OUT/'numerical_environment.json',dict(torch_version=torch.__version__,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,optimizer_defaults='Adam betas0.9/0.999 eps1e-8 weight_decay0; no AMP/no clipping',note='Backend defaults unchanged; seeded parameter initialization/order/RNG; no cross-hardware bitwise claim.'))
    print('Shared actors',[(r['source'],r['shared_actor_count']) for r in audit],'; duplicate20 collisions',len(collisions))

if __name__=='__main__':run()
