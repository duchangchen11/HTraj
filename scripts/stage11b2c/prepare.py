"""CPU-only identity audit, then physical GT-free input export."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from scripts.stage11b1c.data import load,manifest,input_records,truths
import torch,numpy as np,cv2

def main():
    assert not torch.cuda.is_initialized();assert sha(CP)==CK and sha(PP)==PSHA
    original=json.loads((OLD/'protocol_frozen.json').read_text());validation=manifest('validation')
    assert sha(ROOT/'results/tables/stage11b1/validation_target_manifest.csv')==original['data_manifest_SHA']['validation']
    old=torch.load(PP,map_location='cpu',weights_only=True);assert old['checkpoint_SHA']==CK
    assert old['paths'].shape==(20,998,12,2) and old['goals'].shape==(20,998,2)
    assert torch.isfinite(old['paths']).all() and torch.isfinite(old['goals']).all()
    index={r['sample_id']:r for r in validation};h={r['sample_id']:r for r in input_records('validation')};y=truths('validation')
    assert set(old['sample_ids'])==set(index)==set(h)==set(y) and len(index)==998
    ids=old['sample_ids'];cnew(OUT/'validation_identity_manifest.csv',[index[s] for s in ids])
    torch.save(dict(sample_ids=ids,scenes=[h[s]['scene'] for s in ids],history=torch.tensor(np.stack([h[s]['history'] for s in ids]))),CACHE/'validation_history_only.pt')
    train=input_records('train')[:2];df=load('train');by={r['sample_id']:r for r in manifest('train')};groups={int(k):g for k,g in df.groupby('metaId',sort=False)};cf=[]
    from scripts.stage11b1.evaluation import history_record
    for r in train:
        a=groups[int(by[r['sample_id']]['meta_id'])][['x','y']].to_numpy(dtype='float32').reshape(20,2).copy();a[8:]=1000000
        sid,hist=history_record(dict(sample_id=r['sample_id'],xy=a));assert np.array_equal(hist[0].numpy(),r['history']);cf.append(dict(sample_id=sid,scene=r['scene'],history=hist[0].numpy()))
    for name,rs in [('precheck_history_only',train),('counterfactual_history_only',cf)]:torch.save(dict(sample_ids=[r['sample_id'] for r in rs],scenes=[r['scene'] for r in rs],history=torch.tensor(np.stack([r['history'] for r in rs]))),CACHE/(name+'.pt'))
    scenes={}
    for scene in sorted({r['scene'] for r in validation}|{r['scene'] for r in train}):
        p=RES/'data/SDD/train'/scene/'reference.jpg';im=cv2.imread(str(p));assert im is not None
        resized=cv2.resize(im,(0,0),fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
        scenes[scene]=dict(reference_SHA=sha(p),height=im.shape[0],width=im.shape[1],resized_height=resized.shape[0],resized_width=resized.shape[1])
        expected=[r['reference_SHA'] for r in validation if r['scene']==scene]
        if expected:assert set(expected)=={sha(p)}
    jnew(OUT/'input_export.json',dict(passed=True,checkpoint_SHA=CK,original_prediction_SHA=PSHA,validation_manifest_SHA=original['data_manifest_SHA']['validation'],
        all998_history_GT_identity_verified=True,all_sample_ids=ids,scenes=scenes,precheck_sample_ids=[r['sample_id'] for r in train],
        artifacts={n:dict(path=str((CACHE/(n+'.pt')).relative_to(ROOT)),sha256=sha(CACHE/(n+'.pt'))) for n in ['validation_history_only','precheck_history_only','counterfactual_history_only']},
        original_traces_SHA=sha(OLD/'prediction_traces_epoch030.csv.gz'),GPU_input_schema=['sample_ids','scenes','history'],GPU_input_future_fields=False,raw_future_counterfactual_projection_identical=True,
        CPU_only=True,CUDA_initialized=False,test_loaded=False,source_SHA=sha(Path(__file__))))
    print('CPU identity/input export PASS; validation998,precheck2; no future in GPU input artifacts',flush=True)
if __name__=='__main__':main()
