"""Before training: raw hashes, exact old target cohorts, fresh paired CVs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.evaluation import cv_prediction
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.evaluation import noise
from src.socialvae_baseline.groups import history_features
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

root=Path('results/tables/stage5p2');plan=json.load(open('configs/stage5p2/protocol.json'))
old_cv={(r['recording'],int(r['start_frame']),int(r['target_id']),r['model']):r for r in read_csv('results/tables/stage5p1/cv_paired_targets.csv.gz')}
print('Old CV columns:',next(iter(old_cv.values())).keys(),flush=True)
checks=[];cvrows=[];features=[];seeds=[]
for rec,manifest in plan['manifests'].items():
    assert sha256(manifest['path'])==manifest['sha256']
    for split in ['train','validation']:
        ds=CausalDataset(plan['data_root'],manifest['path'],split)
        previous=read_csv(f'results/tables/stage5p1/manifests/{rec}/B_{split}_targets.csv.gz')
        assert len(ds)==len(previous)
        for i,row in enumerate(previous):
            item=ds[i]
            assert (item['recording'],item['start_frame'],item['actor'])==(row['recording'],int(row['start_frame']),int(row['target_id']))
            assert array_sha(item['x'][:,:2])==row['history_sha256']
            assert array_sha(item['y'])==row['future_sha256']
            if split=='validation':
                wi,ti=ds.indices[i];feature=dict(recording=item['recording'],start_frame=item['start_frame'],target_id=item['actor'],
                    history_position_sha256=array_sha(item['x'][:,:2]),future_position_sha256=array_sha(item['y']),
                    **history_features(ds.contexts[wi],ti))
                features.append(feature)
                prior,ps=noise(item);posterior,qs=noise(item,True)
                seeds.append(dict(recording=item['recording'],start_frame=item['start_frame'],target_id=item['actor'],
                    prior_seed=ps,posterior_seed=qs,prior_noise_sha256=array_sha(prior.numpy()),posterior_noise_sha256=array_sha(posterior.numpy())))
                for kind in plan['CV']['models']:
                    pred=cv_prediction(item['x'][:,:2],kind)
                    errors=np.linalg.norm(pred-item['y'],axis=-1)
                    old=old_cv[(rec,item['start_frame'],item['actor'],kind)]
                    assert abs(float(errors.mean())-float(old['ADE_m']))<1e-12
                    assert abs(float(errors[-1])-float(old['FDE_m']))<1e-12
                    cvrows.append(dict(recording=rec,start_frame=item['start_frame'],target_id=item['actor'],model=kind,
                        ADE_m=float(errors.mean()),FDE_m=float(errors[-1]),future_position_sha256=feature['future_position_sha256']))
        checks.append(dict(validation_recording=rec,split=split,target_windows=len(ds),windows=len({wi for wi,i in ds.indices}),
            people=len({(ds.windows[wi]['recording'],int(ds.contexts[wi].ids[i])) for wi,i in ds.indices}),
            manifest_sha256=sha256(manifest['path']),old_target_metadata_sha256=sha256(f'results/tables/stage5p1/manifests/{rec}/B_{split}_targets.csv.gz'),
            exact_target_ID_order_and_history_and_future_SHA_verified=True,full_pass_steps=(len(ds)+127)//128,
            recordings=sorted(ds.records)))
        print(checks[-1],flush=True)
json_new(root/'cohort_verification.json',checks)
csv_new(root/'history_features.csv.gz',features)
csv_new(root/'sampling_seeds.csv.gz',seeds)
csv_new(root/'cv_paired_targets.csv.gz',cvrows)
json_new(root/'cv_metrics.json',[dict(recording=rec,model=kind,target_windows=len(rr),ADE_m=float(np.mean([r['ADE_m'] for r in rr])),
    FDE_m=float(np.mean([r['FDE_m'] for r in rr]))) for rec in plan['manifests'] for kind in plan['CV']['models']
    for rr in [[r for r in cvrows if r['recording']==rec and r['model']==kind]]])
