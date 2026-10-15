"""Freeze protected files, complete-track manifests and all experiment endpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,time
from collections import defaultdict
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.public_training.data import RECORDINGS,select_window,graph_from_history
from src.public_training.model import make_model
from src.training.social_diagnostics import csv_new

CONF=Path('configs/stage4p1');ROOT=Path('results/tables/stage4p1')


def json_once(path,value):
    if path.exists():assert json.loads(path.read_text())==value,path
    else:write_json_new(path,value)


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='c0494e8aaba4bcd350d2efa75ba9a6f675d9146f'
    old=json.loads(Path('configs/stage4/protected_artifacts.json').read_text())
    paths=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))-{''}
    paths|={r['path'] for r in old['files']}
    upstream=json.loads(Path('configs/stage4/official_sources.json').read_text())
    paths|={r['local'] for source in upstream['sources'] for r in source['assets']+source['files']}
    json_once(CONF/'protected_artifacts.json',dict(base_commit='c0494e8aaba4bcd350d2efa75ba9a6f675d9146f',
        files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)],policy='all old tracked+142 checkpoints+frozen official source and assets; no old file overwritten'))
    sources={};full=defaultdict(list);all_manifest={};counts=[];t=time.perf_counter()
    oldfold=json.loads(Path('configs/stage3p5/folds/val_crowds_zara02.json').read_text())
    for rec in RECORDINGS:
        file=Path('data/raw/eth_ucy')/(rec+'.txt');a=np.loadtxt(file);assert a.shape[1]==4
        sources[rec]=dict(path=str(file),sha256=sha256(file),rows=len(a))
        assert sources[rec]['sha256']==oldfold['source_files'][rec]['sha256']
        by=defaultdict(dict)
        for frame,pid,x,y in a:by[int(frame)][int(pid)]=[x,y]
        frames=np.unique(a[:,0]).astype(int);split='validation' if rec=='crowds_zara02' else 'train'
        for protocol in ('A_offline_complete20','B_history_context_complete_targets'):
            entries=[];people=0;ctx=0
            for k in range(len(frames)-19):
                span=frames[k:k+20]
                if not np.all(np.diff(span)==10):continue
                if split=='train' and span[-1]>=oldfold['cutoff_by_recording'][rec]['train_end_exclusive']:continue
                sample=select_window(by,span,protocol)
                if sample is None:continue
                sample.update(recording=rec,window_id=rec+':'+str(span[0]),graph=graph_from_history(sample['observed']))
                entries.append(dict(window_id=sample['window_id'],recording=rec,frames=sample['frames'],agent_ids=sample['agent_ids'],target_ids=sample['target_ids']))
                people+=len(sample['target_ids']);ctx+=len(sample['agent_ids'])
                if protocol=='A_offline_complete20':full[split].append(sample)
            all_manifest[protocol,rec]=entries
            counts.append(dict(protocol=protocol,split=split,recording=rec,windows=len(entries),target_windows=people,context_windows=ctx))
    for protocol in ('A_offline_complete20','B_history_context_complete_targets'):
        manifest=dict(protocol=protocol,source_files=sources,obs_len=8,pred_len=12,dt_seconds=.4,frame_step=10,round_decimals=4,
            splits={split:[r for rec in RECORDINGS if (rec=='crowds_zara02')==(split=='validation') for r in all_manifest[protocol,rec]] for split in ('train','validation')},
            training_recordings=[r for r in RECORDINGS if r!='crowds_zara02'],validation_recording='crowds_zara02',ETH_evaluated=False,
            old_validation_regions_promoted_to_train=False,train_cutoffs={r:oldfold['cutoff_by_recording'][r] for r in RECORDINGS if r!='crowds_zara02'},
            validation_support='entire held recording; includes historically exposed old train+validation times, never training here',independent_confirmation=False,
            rule='A: all context/targets complete20 and >=2 nodes; B: context complete8 history only, >=2 context and >=1 complete12 supervised target; future values never used by graph')
        json_once(CONF/(protocol+'_manifest.json'),manifest)
    cache=Path('data/processed/stage4p1/A_windows.pt');cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        previous=torch.load(cache,weights_only=True)
        for split in full:
            assert len(previous[split])==len(full[split])
            for old_sample,new_sample in zip(previous[split],full[split]):
                for key in new_sample:
                    if torch.is_tensor(new_sample[key]):assert torch.equal(new_sample[key],old_sample[key])
                    else:assert new_sample[key]==old_sample[key]
    else:torch.save(dict(full),cache)
    csv_new(ROOT/'split_counts.csv',counts)
    parameters={kind:sum(p.numel() for p in make_model(kind).parameters()) for kind in ('Full','NoSocial')}
    protocol=dict(stage='4.1',base_commit='c0494e8aaba4bcd350d2efa75ba9a6f675d9146f',upstream_commit='333d3a57b4d2705e129b21aefefa09c79b2b9ae1',
        training_protocol='A_offline_complete20',B_training=False,seed=17,models=['Full','NoSocial'],parameters=parameters,
        observation_steps=8,prediction_steps=12,dt_seconds=.4,coordinate_unit='m',max_epochs=30,smoke_epochs=5,
        optimizer='SGD',learning_rate=.01,momentum=0.,weight_decay=0.,scheduler=None,gradient_clip=1.,accumulation_windows=64,
        training_loss='unchanged official clamped bivariate Gaussian NLL; equal-window average, last partial accumulation normalized by actual count',
        best_checkpoint='minimum validation target-weighted official NLL; earliest exact tie; never K20/test selection',
        validation_recording='crowds_zara02',train_source_recordings=[r for r in RECORDINGS if r!='crowds_zara02'],
        train_policy='all eligible original-train-region complete20 windows in four nonETH recordings; retain old val+embargo unused; no1024 subsample',
        validation_policy='all eligible complete20 windows of the whole held recording; prior researcher exposure, not independent',
        smoke_gate='all finite loss/metrics/gradients, nonzero gradient, checkpoint optimizer+RNG reload gives exactly same output; no validation improvement threshold; continue exact run from epoch5 to30 only',
        evaluation_modes=['K1_distribution_mean','K1_first_sample','K20_independent_min'],evaluation_seed=314159,
        evaluation_sampling='common fixed standard-normal bank across models/checkpoints;20 independent displacement draws; separate per-target minADE/minFDE',
        evaluation_every_epoch=True,training_metrics='train-mode NLL/grad; validation metrics all modes/NLL each epoch; full train eval at epoch5,best,last only',
        permutation=dict(checkpoint='official zara2 val_best frozen Stage4',windows=64,selection='first64 chronological eligible validation windows; no error-based choice',seed=271828,
            mode='K1 Gaussian mean only; graph and tracks permuted consistently; probe raw Gaussian outputs after each main module'),
        no_social='identity graph; all TXP-CNN spatial3x3 kernels reshaped9x1 to same allocated parameter count; nodewise training BN; original self ST-GCN/affine/view/Gaussian/NLL retained',
        no_social_confound='removes graph+node conv+cross-node normalization jointly; kernel geometry and normalization change; cannot isolate graph benefit from Full-vs-NoSocial alone',
        cache=str(cache),cache_sha256=sha256(cache),
        manifests={p:dict(path=str(CONF/(p+'_manifest.json')),sha256=sha256(CONF/(p+'_manifest.json'))) for p in ('A_offline_complete20','B_history_context_complete_targets')},
        cpu_threads=1,device='cuda',ETH_evaluated=False,pretrained_initialization=False,auto_push=False,
        endpoint='one recording, seed17, Full+NoSocial<=30 total epochs incl first5; official checkpoint permutation only; no LR/seed/fold search, no B training')
    write_json_new(CONF/'protocol.json',protocol)
    write_json_new(ROOT/'preparation.json',dict(seconds=time.perf_counter()-t,cache_sha256=sha256(cache),source_files=sources,
        script_sha256=sha256(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print(json.dumps(dict(counts=counts,parameters=parameters,seconds=time.perf_counter()-t)))


if __name__=='__main__':main()
