"""CPU audit of inherited tensors, valid image extents and fixed intervention inputs."""
from .common import *
from .interventions import intervene,CONDITIONS
from scripts.stage11b0.common import RES
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import preprocess_scene
import torch,numpy as np,pandas as pd,cv2
def main():
    cpu_only();assert sha(CP)==CK and sha(PP)==PSHA
    previous=json.loads((ROOT/'configs/stage11b2e/generation_config.json').read_text())
    source=json.loads((ROOT/'results/tables/stage11b2c/input_export.json').read_text())
    hp=checked(previous['history']);old=torch.load(PP,map_location='cpu',weights_only=True)
    assert hp['sample_ids']==old['sample_ids'] and len(set(hp['sample_ids']))==998
    assert set(hp)=={'sample_ids','scenes','history'} and hp['history'].shape==(998,8,2)
    scenes=sorted(set(hp['scenes']));audits=[];rows=[];semantics={}
    for scene in scenes:
        rec=previous['semantics'][scene];s=checked(rec);assert digest(s)==rec['tensor_SHA']
        assert s.shape[:2]==(1,6) and s.dtype==torch.float32 and torch.isfinite(s).all()
        ref=source['scenes'][scene];impath=RES/'data/SDD/train'/scene/'reference.jpg';assert sha(impath)==ref['reference_SHA']
        im=cv2.imread(str(impath));valid=cv2.resize(im,(0,0),fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
        h,w=valid.shape[:2];H,W=s.shape[-2:]
        assert (h,w)==(ref['resized_height'],ref['resized_width']) and (H,W)==((h+31)//32*32,(w+31)//32*32)
        for condition in CONDITIONS:
            v=intervene(s,condition,h,w);path=CACHE/f'semantic_{condition}_{scene}.pt';assert not path.exists();torch.save(v,path)
            r=dict(scene=scene,condition=condition,path=str(path.relative_to(ROOT)),file_SHA=sha(path),tensor_SHA=digest(v),original_semantic_SHA=rec['tensor_SHA'],
                reference_SHA=ref['reference_SHA'],shape=json.dumps(list(v.shape)),dtype=str(v.dtype),valid_height=h,valid_width=w,
                canvas_height=H,canvas_width=W,padding='bottom/right to multiple32; S2/S3 untouched, S1 all canvas zero',
                formula={'S0':'identity','S1':'all six channels zero including padding','S2':'each channel valid spatial mean, padding unchanged','S3':'valid horizontal flip without interpolation, padding unchanged'}[condition])
            rows.append(r);semantics.setdefault(condition,{})[scene]=r
        sums=s.sum(dim=1);core=s[:,:,:h,:w]
        for ch in range(6):
            v=core[0,ch].double();audits.append(dict(scene=scene,channel=f'channel{ch}',shape=json.dumps(list(s.shape)),dtype=str(s.dtype),
                valid_min=float(v.min()),valid_max=float(v.max()),valid_mean=float(v.mean()),valid_variance=float(v.var(unbiased=False)),
                canvas_min=float(s[:,ch].min()),canvas_max=float(s[:,ch].max()),sum_channels_min=float(sums.min()),
                sum_channels_max=float(sums.max()),sum_channels_mean=float(sums.mean()),mean_abs_sum_minus1=float((sums-1).abs().mean()),
                finite=True,negative_values=int((v<0).sum()),values_above1=int((v>1).sum()),
                valid_height=h,valid_width=w,canvas_height=H,canvas_width=W,semantic_SHA=rec['tensor_SHA'],reference_SHA=ref['reference_SHA']))
    csvnew(OUT/'semantic_tensor_audit.csv',audits);csvnew(OUT/'intervention_manifest.csv',rows)
    # Strict checkpoint architecture/config audit on CPU; no segmentation recomputation.
    net,cfg=build_model(17,'cpu');saved=torch.load(CP,map_location='cpu',weights_only=True);net.load_state_dict(saved['model'],strict=True);net.eval()
    assert state_digest(net.state_dict())==previous['original_state_SHA'] and sum(p.numel() for p in net.parameters())==53155339
    seg=net.semantic_segmentation;details=dict(class_name=type(seg).__name__,head_repr=str(seg.segmentation_head),training=seg.training,
        state_SHA=state_digest(seg.state_dict()),segmentation_file_SHA='75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194',
        no_parameters_require_grad=not any(p.requires_grad for p in seg.parameters()),
        class_labels='UNVERIFIED; only channel0..channel5; source constructor documentation is not a checkpoint class map')
    rgb=[]
    for scene in scenes:
        im=preprocess_scene(scene,cfg,'cpu');H,W=semantics['S0'][scene]['canvas_height'],semantics['S0'][scene]['canvas_width']
        assert im.shape==(1,3,H,W)
        rgb.append(dict(scene=scene,preprocessed_RGB_SHA=digest(im),shape=list(im.shape),pipeline='cv2 BGR read; INTER_AREA x0.25; bottom/right zero border to32; author SMP resnet101/imagenet preprocess; CHW float32'))
    jnew(OUT/'segmentation_architecture.json',details);jnew(OUT/'RGB_preprocessing_audit.json',rgb)
    oldm=pd.read_csv(ROOT/'results/tables/stage11b2c/tensor_manifest.csv');assert oldm.sample_id.tolist()==hp['sample_ids']
    for r in oldm.to_dict('records'):assert sha(ROOT/r['path'])==r['sha256']
    # Fixed history-only training probes; no future data read or exported.
    trainrec=source['artifacts']['precheck_history_only'];train=checked(trainrec)
    assert set(train)=={'sample_ids','scenes','history'} and len(train['sample_ids'])==2
    cfgdata=dict(checkpoint=dict(path=str(CP.relative_to(ROOT)),sha256=CK),model_state_SHA=previous['original_state_SHA'],
        history=previous['history'],train_history=trainrec,semantics=semantics,
        original_predictions=dict(path=str(PP.relative_to(ROOT)),sha256=PSHA),
        original_tensor_records=[dict(sample_id=r['sample_id'],path=r['path'],sha256=r['sha256']) for r in oldm.to_dict('records')],
        regression_indices=[0,499,997],train_semantic_rule='bookstore_0 has no persisted semantic cache; one controlled training-only segmentation forward, require original Stage2C precheck all tensors bitwise; never rerun five validation semantics',
        train_reference=dict(path='experiments/stage11b2c/cache/precheck/rep0_target0.pt',sha256=sha(ROOT/'experiments/stage11b2c/cache/precheck/rep0_target0.pt')),
        all_conditions=CONDITIONS,GT_fields=False,budget=dict(hard=7200,safe=6600))
    jnew(CONF/'generation_config.json',cfgdata)
    jnew(OUT/'cache_audit.json',dict(passed=True,targets=998,all998_original_tensor_file_SHA=True,
        five_existing_semantics_verified=True,validation_segmentation_recomputed=False,semantic_channels=6,history_channels=8,
        resize=.25,padding_divisor=32,model_parameters=53155339,predictor_parameters=1641381,state_SHA=previous['original_state_SHA'],
        reference_and_RGB_padding_matched=True,GT_read=False,test_read=False))
    print('CPU_CACHE_AUDIT_PASS; channels audited; four inputs written; no future loaded',flush=True)
if __name__=='__main__':main()
