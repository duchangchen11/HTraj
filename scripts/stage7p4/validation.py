"""Author checkpoints on OFFICIAL validation only, never a test generator."""
import sys
import contextlib
import io
import json
import os
import time
import numpy as np
import torch
from src.agentformer_feasibility.common import *
from pathlib import Path
from src.agentformer_feasibility.diagnosis import *
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_causal.data import array_sha
from src.socialvae_baseline.artifacts import json_new
from scripts.stage7p1.runtime import ResourceMeter


def run():
    from model.model_lib import model_dict
    from utils.config import Config
    from utils.utils import prepare_seed
    from data.dataloader import data_generator
    work=CACHE/'validation_workspace'
    work.mkdir(parents=True,exist_ok=True)
    links={work/'cfg': SOURCE/'cfg',work/'datasets/eth_ucy':DATA}
    for pair in protocol()['author_pairs']:
        for kind in ['CVAE','DLow']:
            links[work/'results'/pair[kind+'_config']/'models']=Path(pair[kind+'_checkpoint']).parent
    for dest,source in links.items():
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(source,target_is_directory=True)
        assert dest.resolve()==source.resolve()
    windows=read_csv(ROOT/'configs/stage7p4/validation_windows.csv.gz')
    summary=[]
    original=Path.cwd()
    for pair in protocol()['author_pairs']:
        fold=pair['fold'];os.chdir(work)
        expected={(r['sequence'],int(r['last_observed_frame'])):int(r['people']) for r in windows if r['fold']==fold}
        expected_ids={(r['sequence'],int(r['last_observed_frame'])):set(json.loads(r['target_ids'])) for r in windows if r['fold']==fold}
        rows,times,hashes,seen=[],[],[],set()
        rawlog=CACHE/f'{fold}_validation_stdout.txt'
        print('VALIDATION ONLY',fold,len(expected),'valid windows',flush=True)
        try:
            with rawlog.open('x') as log,contextlib.redirect_stdout(log):
                cfg=Config(pair['DLow_config']);prepare_seed(cfg.seed)
                model=model_dict['dlow'](cfg)
                check=model.load_state_dict(torch.load(pair['DLow_checkpoint'],map_location='cpu',weights_only=False)['model_dict'],strict=True)
                assert not check.missing_keys and not check.unexpected_keys
                pre=torch.load(pair['CVAE_checkpoint'],map_location='cpu',weights_only=False)['model_dict']
                assert nested_equal(model.pred_model[0].state_dict(),pre)
                model.set_device(torch.device('cuda:0'));model.eval()
                generator=data_generator(cfg,log,split='val',phase='testing')
                assert all(s.endswith('_val') for s in generator.sequence_to_load)
                raw_history={s:np.genfromtxt(DATA/fold/(s+'.txt'),dtype=str)[:,[0,1,13,15]].astype(np.float64) for s in generator.sequence_to_load}
                with ResourceMeter() as meter:
                    while not generator.is_epoch_end():
                        packet=generator()
                        if packet is None:continue
                        key=packet['seq'],int(packet['frame'])
                        assert key in expected and key not in seen and len(packet['valid_id'])==expected[key]
                        assert set(map(int,packet['valid_id']))==expected_ids[key]
                        assert all(torch.all(m==1) for m in [*packet['pre_motion_mask'],*packet['fut_motion_mask']])
                        seen.add(key)
                        with torch.no_grad():
                            model.set_data(packet)
                            prediction,_=model.inference(mode='infer',sample_num=20)
                        assert prediction.shape==(len(packet['valid_id']),20,12,2)
                        p=prediction.detach().cpu().numpy()*cfg.traj_scale
                        x=torch.stack(packet['pre_motion_3D']).numpy()*cfg.traj_scale
                        y=torch.stack(packet['fut_motion_3D']).numpy()*cfg.traj_scale
                        if not np.isfinite(p).all():raise FloatingPointError('Nonfinite validation prediction')
                        result=candidate_errors(p,y);cv=cv_predictions(x)
                        speed,labels=history_labels(x)
                        ids=np.asarray(packet['valid_id'],dtype=np.int64)
                        local_neighbors,full8_people=history_neighbor_counts(raw_history[key[0]],key[1],ids)
                        directory=CACHE/'validation_predictions'/fold/key[0];directory.mkdir(parents=True,exist_ok=True)
                        path=directory/f'frame_{key[1]:06d}.npz'
                        with path.open('xb') as f:np.savez_compressed(f,history=x,truth=y,samples=p,person_ids=ids)
                        hashes.append(dict(fold=fold,sequence=key[0],frame=key[1],path=str(path.relative_to(ROOT)),sha256=digest(path),bytes=path.stat().st_size))
                        for i,actor in enumerate(ids):
                            row=dict(fold=fold,sequence=key[0],last_observed_frame=key[1],person_id=int(actor),
                                history_speed_mps=float(speed[i]),speed_group=str(labels[i]),motion_group=('stationary' if labels[i]=='stationary' else 'moving'),
                                context_people=len(ids),other_complete20_context_people=len(ids)-1,history_full8_people=full8_people,history_full8_local2m_neighbors=int(local_neighbors[i]),context_group=context_bin(int(local_neighbors[i])),
                                GT_SHA256=array_sha(y[i]),history_SHA256=array_sha(x[i]),
                                minADE20=float(result['minADE20'][i]),minFDE20=float(result['minFDE20'][i]),
                                K1_ADE=float(result['K1_ADE'][i]),K1_FDE=float(result['K1_FDE'][i]),
                                ADE_winner=int(result['ADE_winner'][i]),FDE_winner=int(result['FDE_winner'][i]),
                                endpoint_RMS_m=float(result['endpoint_RMS'][i]),endpoint_pairwise_mean_m=float(result['endpoint_pairwise_mean'][i]))
                            for name,yp in cv.items():
                                errors=np.linalg.norm(yp[i]-y[i],axis=-1)
                                row[name+'_ADE']=float(errors.mean());row[name+'_FDE']=float(errors[-1])
                            rows.append(row)
                            for t in range(12):
                                times.append(dict(fold=fold,sequence=key[0],last_observed_frame=key[1],person_id=int(actor),
                                    future_step=t+1,seconds=(t+1)*.4,
                                    DLow_ADEbest_error=float(result['ADE_best_step'][i,t]),
                                    DLow_FDEbest_error=float(result['FDE_best_step'][i,t]),
                                    DLow_first_error=float(result['first_step'][i,t]),
                                    **{name+'_error':float(np.linalg.norm(v[i,t]-y[i,t])) for name,v in cv.items()}))
                        if len(seen)%100==0:print(f'validation {fold}: {len(seen)}/{len(expected)} windows',file=sys.__stdout__,flush=True)
                assert seen==expected.keys()
            csv_new(OUT/f'{fold}_validation_targets.csv.gz',rows)
            csv_new(OUT/f'{fold}_validation_timesteps.csv.gz',times)
            csv_new(OUT/f'{fold}_validation_prediction_SHA256.csv.gz',hashes)
            group_rows=[]
            metrics=['minADE20','minFDE20','K1_ADE','K1_FDE','endpoint_RMS_m','endpoint_pairwise_mean_m']
            metrics += [name+'_'+metric for name in ['CV-Last2','CV-OLS4','CV-OLS8'] for metric in ['ADE','FDE']]
            for group_type,levels in [('all',['all']),('speed_group',['stationary','low','normal','fast']),
                    ('motion_group',['stationary','moving']),('context_group',['none','1-3','4-9','10+'])]:
                for level in levels:
                    subset=[r for r in rows if group_type=='all' or r[group_type]==level]
                    group_rows.append(dict(fold=fold,group_type=group_type,group=level,targets=len(subset),
                        **{k:float(np.mean([r[k] for r in subset])) if subset else None for k in metrics}))
            csv_new(OUT/f'{fold}_validation_groups.csv',group_rows)
            time_rows=[]
            for t in range(1,13):
                sub=[r for r in times if r['future_step']==t]
                names=[k for k in sub[0] if k.endswith('_error')]
                time_rows.append(dict(fold=fold,step=t,seconds=t*.4,targets=len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in names}))
            csv_new(OUT/f'{fold}_validation_step_summary.csv',time_rows)
            row=dict(fold=fold,split='official validation ONLY',windows=len(seen),targets=len(rows),
                source_sequences=generator.sequence_to_load,resource=meter.result,
                original_CVAE_SHA256=pair['CVAE_sha256'],original_DLow_SHA256=pair['DLow_sha256'],
                overall=group_rows[0],prediction_files=len(hashes),actual_sampling='DLow mean=True K20 learned heads',
                metrics_precision='float64 arithmetic on native float32 predictions/GT, no text rounding',
                candidates_used_for_input=False,first_head_is_not_mathematical_expectation=True,
                interpretation='K20 oracle and deterministic K1 have different budgets, no equal-budget superiority claim',
                test_split_loaded=False)
            json_new(OUT/f'{fold}_validation.json',row);summary.append(row)
            del model,pre;torch.cuda.empty_cache()
        finally:os.chdir(original)
    json_new(OUT/'validation_summary.json',dict(cases=summary,all_completed=len(summary)==2,test_metrics_recomputed=False,
        groups='History-only speed and full8 local2m neighbor counts. Offline complete20 context count is a separate descriptive field, never used to define groups. Protocol eligibility itself uses future existence.'))
    return summary
