"""Verify complete frozen cohorts and copy immutable reference metadata."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.evaluation import cv_prediction
from src.socialvae_causal.io import csv_new,read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
plan=json.load(open('configs/stage5p4/ego_protocol.json'));checks=[]
cv={(r['recording'],int(r['start_frame']),int(r['target_id']),r['model']):r for r in read_csv('results/tables/stage5p2/cv_paired_targets.csv.gz')}
for rec,manifest in plan['manifests'].items():
    assert sha256(manifest['path'])==manifest['sha256']
    full=json.load(open(f'results/tables/stage5p2/{rec}/provenance.json'))
    assert full['manifest']['sha256']==manifest['sha256']
    for split in ['train','validation']:
        ds=CausalDataset(plan['data_root'],manifest['path'],split)
        old=read_csv(f'results/tables/stage5p1/manifests/{rec}/B_{split}_targets.csv.gz')
        assert len(ds)==len(old)==full['training_targets' if split=='train' else 'validation_targets']
        for i,ref in enumerate(old):
            item=ds[i]
            assert (item['recording'],item['start_frame'],item['actor'])==(ref['recording'],int(ref['start_frame']),int(ref['target_id']))
            assert array_sha(item['x'][:,:2])==ref['history_sha256'] and array_sha(item['y'])==ref['future_sha256']
            if split=='validation':
                for kind in ['CV-Last2','CV-OLS4','CV-OLS8']:
                    errors=np.linalg.norm(cv_prediction(item['x'][:,:2],kind)-item['y'],axis=-1)
                    r=cv[(rec,item['start_frame'],item['actor'],kind)]
                    assert abs(float(errors.mean())-float(r['ADE_m']))<1e-12 and abs(float(errors[-1])-float(r['FDE_m']))<1e-12
        checks.append(dict(recording=rec,split=split,target_windows=len(ds),manifest_sha256=manifest['sha256'],
            records=sorted(ds.records),all_target_IDs_history_GT_SHA_match_Full=True))
        if split=='train':assert rec not in ds.records
        else:assert set(ds.records)=={rec}
for name in ['history_features.csv.gz','sampling_seeds.csv.gz','cv_paired_targets.csv.gz','cv_metrics.json']:
    source=Path('results/tables/stage5p2')/name;destination=Path('results/tables/stage5p4')/name
    with destination.open('xb') as f:f.write(source.read_bytes())
    assert sha256(source)==sha256(destination)
json_new('results/tables/stage5p4/cohort_verification.json',checks)
print(json.dumps(checks,indent=2))
