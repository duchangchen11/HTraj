"""Freeze separate offline/causal manifests and freshly paired CV results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import Counter
import hashlib
import json
import subprocess
import time
import numpy as np
from src.data.preprocessing import Recording,sha256,write_json_new
from src.socialvae_causal.data import PROTOCOL,OFFLINE,CausalDataset,history_context,context_counts,array_sha
from src.socialvae_causal.io import csv_new
from src.socialvae_causal.offline_reference import official_examples
from src.socialvae_causal.evaluation import cv_prediction

OUT=Path('results/tables/stage5p1')
CONF=Path('configs/stage5p1')


def input_sha(x,n):
    return hashlib.sha256((array_sha(x)+array_sha(n)).encode()).hexdigest()


def aggregate(fold,protocol,split,rows,windows,seconds):
    return dict(fold=fold,protocol=protocol,split=split,manifest_windows=windows,
        supervised_windows=len({(r['recording'],r['start_frame']) for r in rows}),target_windows=len(rows),
        unique_recording_person_units=len({(r['recording'],r['target_id']) for r in rows}),
        history_neighbor_entities=sum(r['history_neighbors'] for r in rows),
        new_entering_neighbor_entities=sum(r['new_entering_neighbors'] for r in rows),
        mean_history_neighbors=sum(r['history_neighbors'] for r in rows)/len(rows),
        new_entering_neighbor_targets=sum(r['new_entering_neighbors']>0 for r in rows),
        near_motion_opportunities=sum(r['near_motion_opportunities'] for r in rows),
        unknown_velocity_near_opportunities=sum(r['blocked_unknown_velocity'] for r in rows),
        all_history_context_blocked_targets=sum(r['all_history_social_context_blocked'] for r in rows),
        build_seconds=seconds)


def main():
    p=json.load(open(CONF/'protocol.json'))
    assert not (OUT/'data_execution.json').exists()
    summaries=[];outputs=[];cv_rows=[];target_keys={};manifests={}
    original=json.load(open('configs/data_manifests/stage2_v1/fold_ETH.json'))
    old_train={(w['recording'],w['start_frame']) for w in original['splits']['train']}
    old_val={(w['recording'],w['start_frame']) for w in original['splits']['validation']}
    for fold,source in p['split_sources'].items():
        assert sha256(source['path'])==source['sha256']
        old=json.load(open(source['path']))
        train,val=old['splits']['train'],old['splits']['validation']
        assert not {w['recording'] for w in train}&{w['recording'] for w in val}
        assert {w['recording'] for w in val}=={fold}
        assert all(w['end_frame']-w['start_frame']==190 for w in train+val)
        assert all((w['recording'],w['start_frame']) in old_train and (w['recording'],w['start_frame']) not in old_val for w in train)
        assert all((w['recording'],w['start_frame']) in old_val for w in val)
        for tag,protocol in [('A',OFFLINE),('B',PROTOCOL)]:
            manifest=dict(schema_version=1,protocol=protocol,origin_manifest=source['path'],origin_manifest_sha256=source['sha256'],
                source_files=old['source_files'],splits=old['splits'],observation_frames=8,prediction_frames=12,
                frame_step=10,seconds_per_sample=.4,coordinate_unit='m',validation_recording=fold,
                original_validation_promoted_to_train=False,held_recording_original_train_excluded=True,ETH_evaluated=False,
                independent_confirmation=False,feature_contract='strict eight raw observation frames only' if tag=='B' else 'unchanged recording-wide official extend; noncausal offline reference',
                source_code_commit=p['official_commit'],full_window_support='start through start+190; only B prediction inputs use start through start+70',
                supervised_target_rule='complete 8 positions plus complete12 label positions; future used for supervision eligibility only in B')
            name=CONF/'manifests'/fold/(tag+'.json');write_json_new(name,manifest);manifests[(fold,tag)]=(name,manifest)
        name,m=manifests[(fold,'B')]
        for split in ['train','validation']:
            t0=time.perf_counter();dataset=CausalDataset(p['data_root'],name,split);rows=[]
            for i in range(len(dataset)):
                item=dataset[i];row=dataset.describe(i)
                row.update(history_sha256=array_sha(item['x'][:,:2]),future_sha256=array_sha(item['y']),
                    forecast_input_sha256=input_sha(item['x'],item['neighbor']),context_ids_json=json.dumps(item['context_ids'].tolist()))
                rows.append(row)
                if split=='validation':
                    for kind in p['CV_models']:
                        pred=cv_prediction(item['x'][:,:2],kind)
                        error=np.linalg.norm(pred-item['y'].astype(np.float64),axis=-1)
                        cv_rows.append(dict(fold=fold,protocol=PROTOCOL,model=kind,window_id=row['window_id'],recording=item['recording'],
                            start_frame=item['start_frame'],target_id=item['actor'],history_sha256=row['history_sha256'],future_sha256=row['future_sha256'],
                            ADE_m=float(error.mean()),FDE_m=float(error[-1])))
            result_path=OUT/'manifests'/fold/('B_'+split+'_targets.csv.gz');csv_new(result_path,rows)
            outputs.append(dict(path=str(result_path),sha256=sha256(result_path)))
            summaries.append(aggregate(fold,PROTOCOL,split,rows,len(dataset.windows),time.perf_counter()-t0))
            target_keys[(fold,'B',split)]={(r['recording'],r['start_frame'],r['target_id']) for r in rows}
            print(f'B {fold} {split}: {len(rows)} target windows',flush=True)
            del dataset,rows
    # Run frozen official loader exactly once per nonETH recording, then apply
    # the same allowed window lists. Future support is declared for A, not hidden.
    official_rows={key:[] for key in [(f,s) for f in p['development_recordings'] for s in ['train','validation']]}
    allowed={(f,s):{(w['recording'],w['start_frame']) for w in manifests[(f,'A')][1]['splits'][s]} for f,s in official_rows}
    sources=next(iter(manifests.values()))[1]['source_files']
    official_load_times={}
    for rec in p['recordings']:
        source=sources[rec];raw=Path(p['data_root'])/source['file'];assert sha256(raw)==source['sha256']
        recording=Recording.read(raw,source['scene']);t0=time.perf_counter()
        items,keys=official_examples(raw);official_load_times[rec]=time.perf_counter()-t0
        contexts={}
        for item,key in zip(items,keys):
            memberships=[(f,s) for f,s in allowed if (rec,key['start_frame']) in allowed[(f,s)]]
            if not memberships:continue
            start,actor=key['start_frame'],key['target_id']
            if start not in contexts:contexts[start]=history_context(recording,start)
            context=contexts[start];i=int(np.flatnonzero(context.ids==actor)[0]);h,y,n=item
            assert np.array_equal(h[:,:2],context.state[:,i,:2])
            observed=(np.abs(n[:8,:,:2]).max(-1)<1e8)
            first=np.where(observed.any(0),observed.argmax(0),-1)
            row=dict(window_id=f'{rec}:{start}',recording=rec,target_id=actor,start_frame=start,last_observed_frame=start+70,end_frame=start+190,
                **context_counts(context,i))
            row.update(history_neighbors=int(observed.any(0).sum()),new_entering_neighbors=int((first>0).sum()),
                future_only_or_dummy_slots=int((~observed.any(0)).sum()),history_sha256=array_sha(h[:,:2]),future_sha256=array_sha(y),
                forecast_input_sha256=input_sha(h,n[:8]),full_official_neighbor_tensor_sha256=array_sha(n),
                context_ids_json=json.dumps(key['neighbor_ids']))
            for membership in memberships:official_rows[membership].append(row)
        print(f'A official {rec}: {len(items)} target windows scanned',flush=True)
        del items,keys,contexts
    comparisons=[]
    for (fold,split),rows in official_rows.items():
        rows.sort(key=lambda r:(r['recording'],r['start_frame'],r['target_id']))
        result_path=OUT/'manifests'/fold/('A_'+split+'_targets.csv.gz');csv_new(result_path,rows)
        outputs.append(dict(path=str(result_path),sha256=sha256(result_path)))
        windows=manifests[(fold,'A')][1]['splits'][split]
        summaries.append(aggregate(fold,OFFLINE,split,rows,len(windows),sum(official_load_times[r] for r in {w['recording'] for w in windows})))
        target_keys[(fold,'A',split)]={(r['recording'],r['start_frame'],r['target_id']) for r in rows}
        a,b=target_keys[(fold,'A',split)],target_keys[(fold,'B',split)]
        comparisons.append(dict(fold=fold,split=split,A_targets=len(a),B_targets=len(b),common=len(a&b),only_A=len(a-b),only_B=len(b-a)))
    csv_new(OUT/'data_summary.csv',summaries);csv_new(OUT/'protocol_target_set_comparison.csv',comparisons)
    csv_new(OUT/'cv_paired_targets.csv.gz',cv_rows)
    summary=[]
    for fold in p['development_recordings']:
        for model in p['CV_models']:
            rows=[r for r in cv_rows if r['fold']==fold and r['model']==model]
            summary.append(dict(fold=fold,model=model,target_windows=len(rows),windows=len({r['window_id'] for r in rows}),
                unique_persons=len({r['target_id'] for r in rows}),ADE_m=np.mean([r['ADE_m'] for r in rows]),FDE_m=np.mean([r['FDE_m'] for r in rows])))
    csv_new(OUT/'cv_metrics.csv',summary)
    b_summaries=[r for r in summaries if r['protocol']==PROTOCOL]
    gate=[]
    for r in b_summaries:
        blocked=r['unknown_velocity_near_opportunities']/max(1,r['near_motion_opportunities'])
        lost=r['all_history_context_blocked_targets']/r['target_windows']
        gate.append(dict(fold=r['fold'],split=r['split'],blocked_near_motion_fraction=blocked,
            all_context_lost_target_fraction=lost,passed=blocked<=p['coverage_stop']['blocked_near_motion_fraction_gt'] and lost<=p['coverage_stop']['all_history_social_context_lost_target_fraction_gt']))
    write_json_new(OUT/'coverage_gate.json',dict(criteria=p['coverage_stop'],rows=gate,passed=all(r['passed'] for r in gate)))
    write_json_new(OUT/'data_execution.json',dict(script_sha256=sha256(__file__),protocol_sha256=sha256(CONF/'protocol.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        raw_sources=sources,manifests=[dict(path=str(path),sha256=sha256(path)) for path,m in manifests.values()],
        target_manifests=outputs,official_loader_seconds=official_load_times,ETH_loaded=False,
        old_validation_promoted_to_train=False,models_evaluated=['CV-Last2','CV-OLS4','CV-OLS8'],neural_metrics=False))


if __name__=='__main__':main()
