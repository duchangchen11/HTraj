"""Plan five record-isolated development folds without moving old val to train."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from src.data.benchmark_audit import BenchmarkDataset,PROTOCOLS,assert_recording_isolation
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.benchmark_audit import history_features
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage3p5');CONF=Path('configs/stage3p5')


def counts(data):
    c=Counter();groups={};eligible=[]
    for start in range(0,len(data),8):
        samples=[data[i] for i in range(start,min(start+8,len(data)))];batch=collate_trajectories(samples)
        gs,features=history_features(batch)
        target=batch['future_valid'].any(1);final=batch['future_valid'][:,-1]
        c.update(windows=len(samples),supervised_windows=int(target.any(-1).sum()),ADE_targets=int(target.sum()),FDE_targets=int(final.sum()),
            supervised_points=int(batch['future_valid'].sum()),visible_targets=int(batch['observed_valid'][:,-1].sum()))
        eligible.extend(start+i for i in range(len(samples)) if target[i].any())
        for g,mask in gs.items():
            row=groups.setdefault(g,Counter());selected=target&mask
            row.update(observed_targets=int(mask.sum()),ADE_targets=int(selected.sum()),FDE_targets=int((final&mask).sum()),
                history_frames_sum=int(features['history_frames'][selected].sum()),motion_frames_sum=int(features['continuous_motion_frames'][selected].sum()))
    return dict(c),groups,eligible


def main():
    torch.set_num_threads(1);oldpath='configs/data_manifests/stage2_v1/fold_ETH.json';old=json.loads(Path(oldpath).read_text());records=sorted({w['recording'] for w in old['splits']['train']});summaries=[];group_rows=[];elig={}
    for rec in records:
        m={k:v for k,v in old.items() if k!='splits'}
        m['source_files']={k:v for k,v in old['source_files'].items() if v['scene']!='ETH'}
        m.update(stage='3.5',validation_recording=rec,origin_manifest=oldpath,origin_manifest_sha256=sha256(oldpath),
            splits=dict(train=[w for w in old['splits']['train'] if w['recording']!=rec],validation=[w for w in old['splits']['validation'] if w['recording']==rec]),
            original_validation_promoted_to_train=False,held_recording_original_train_excluded=True,independent_confirmation=False)
        assert_recording_isolation(m);path=CONF/'folds'/('val_'+rec+'.json');write_json_new(path,m)
        for protocol in PROTOCOLS:
            for split in ('train','validation'):
                data=BenchmarkDataset('data/raw/eth_ucy',path,split,protocol);c,gs,indices=counts(data);elig[(rec,protocol,split)]=indices
                summaries.append(dict(validation_recording=rec,protocol=protocol,split=split,**c,manifest=str(path),manifest_sha256=sha256(path)))
                for group,row in gs.items():group_rows.append(dict(validation_recording=rec,protocol=protocol,split=split,group=group,**row))
    # Original legacy full training pool count, not a silently changed split.
    for protocol in PROTOCOLS:
        data=BenchmarkDataset('data/raw/eth_ucy',oldpath,'train',protocol);c,gs,idx=counts(data)
        summaries.append(dict(validation_recording='legacy_all_nonETH_training',protocol=protocol,split='train',**c,manifest=oldpath,manifest_sha256=sha256(oldpath)))
    csv_new(ROOT/'split_counts.csv',summaries);csv_new(ROOT/'split_group_counts.csv',group_rows)
    rec='students003'
    write_json_new(CONF/'pilot_indices.json',dict(validation_recording=rec,manifest=str(CONF/'folds'/('val_'+rec+'.json')),
        selection='fixed before results: second UNIV recording, keep students001 training; no metric-dependent fold choice',
        protocols={protocol:{split:elig[(rec,protocol,split)] for split in ('train','validation')} for protocol in PROTOCOLS}))
    protocol=dict(stage='3.5',seed=17,epochs=30,batch_size=8,learning_rate=.001,optimizer='Adam default betas/eps; weight_decay0',gradient_clip=1.,cpu_threads=1,hidden_dim=64,
        models=['CV','PedBase','Dynamic-GNN-RN'],protocols=list(PROTOCOLS),pilot_recording=rec,pilot_indices=str(CONF/'pilot_indices.json'),
        raw_root='data/raw/eth_ucy',original_manifest=oldpath,original_manifest_sha256=sha256(oldpath),
        best_rule='minimum own-protocol validation ADE; earliest exact tie; paired FDE, last30 separate',
        data_policy='only original train on four recordings, original validation on held recording; discard held recording originaltrain and retain other old validation',
        training_from_scratch=True,ETH_evaluated=False,independent_confirmation=False,loss='original masked per-track mean L2',
        groups=dict(density='mean degree over valid continuous history steps;0/(0,1]/(1,3]/>3',moving_speed_min_mps=.2,cosine_threshold=.7,
            relationship_change='mean adjacent relative velocity difference of persistent candidate pairs>=0.5m/s; missing opportunities separate'),
        inference_controls='fixed Dynamic best no_messages and same-scene radius-constrained relation randomization; auxiliary only',
        interpretation='one recording one seed pilot; overlapping windows and prior researcher exposure; no stable/causal/five-scene claim')
    write_json_new(CONF/'protocol.json',protocol)
    olddev=json.loads(Path('configs/stage2p5/subset_1024.json').read_text())['validation']['sample_ids'];newval=json.loads((CONF/'folds'/('val_'+rec+'.json')).read_text())['splits']['validation']
    write_json_new(ROOT/'split_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        origin_manifest_sha256=sha256(oldpath),pilot_recording=rec,old_development_exact_window_overlap=sum(w['sample_id'] in olddev for w in newval),
        pilot_validation_windows=len(newval),no_old_validation_in_new_training=True,no_recording_overlap=True,ETH_evaluated=False,
        temporal_support='20 calibrated frames span190 frame-ID ticks; source-recording isolation is stronger than temporal embargo within these new folds',
        historical_exposure='all five recordings used by previous experiments; fresh models exclude held recording but this is not previously untouched confirmation data'))
    print('prepared',len(summaries),'split rows and',len(group_rows),'group counts')


if __name__=='__main__':main()
