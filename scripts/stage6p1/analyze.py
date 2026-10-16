"""Paired frozen-best/last analysis; no optimizer, selection change or new run."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gzip,json,subprocess,time
from collections import defaultdict
import numpy as np
import torch
from src.data.preprocessing import sha256
from src.socialvae_causal.data import CausalDataset,collate_examples,array_sha
from src.socialvae_causal.evaluation import cv_prediction
from src.socialvae_causal.io import csv_new,read_csv
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_baseline.groups import GROUPS
from src.socialvae_attribution.model import new_model
from src.evaluation.stage6_motion import diagnose
from src.motion_sampling.policy import epoch_order,exposure,decision

ROOT=Path('results/tables/stage6p1');OUT=ROOT/'analysis'
CONDITIONS=['Uniform','Stratified'];WEIGHTS=['target-window','equal-person-ID']
FAMILIES={'K1-prior':('K1_prior_ADE_m','K1_prior_FDE_m'),
    'K1-first-sample':('K1_sample_ADE_m','K1_sample_FDE_m'),
    'K20-independent-minima':('minADE20_m','minFDE20_m')}


def key(row):return row['recording'],int(row['start_frame']),int(row.get('target_id',row.get('actor')))


def average(rows,field,weight):
    rr=[r for r in rows if r.get(field) is not None and r.get(field)!='']
    if not rr:return None
    if weight=='target-window':return float(np.mean([float(r[field]) for r in rr]))
    people=defaultdict(list)
    for r in rr:people[(r['recording'],int(r['target_id']))].append(float(r[field]))
    return float(np.mean([np.mean(v) for v in people.values()]))


def main():
    start=time.perf_counter();OUT.mkdir()
    plan=json.load(open('configs/stage6p1/protocol.json'));datasets={};pools={};checks=[];index=[]
    features={key(r):r for r in read_csv(ROOT/'history_features.csv.gz')}
    metrics=[];diagnostics=[];steps=[];mismatches=[];curve_stats=[];max_difference=0.
    packet=Path('experiments/stage6p1/cache/k1_prediction_packets.jsonl.gz');packet.parent.mkdir(parents=True,exist_ok=True)
    packets=gzip.GzipFile(filename='',mode='wb',fileobj=packet.open('xb'),mtime=0)
    for rec,manifest in plan['manifests'].items():
        assert sha256(manifest['path'])==manifest['sha256']
        ds=CausalDataset(plan['data_root'],manifest['path'],'validation');datasets[rec]=ds
        pools[rec]=read_csv(ROOT/f'pools/{rec}_validation.csv.gz')
        for condition in CONDITIONS:
            directory=ROOT/'runs'/rec/condition
            selection=json.load(open(directory/'selection.json'));provenance=json.load(open(directory/'provenance.json'))
            assert selection['training_completed'] and selection['completed_epochs']==30
            curves=read_csv(directory/'training_curve.csv');assert len(curves)==30
            best=min(curves,key=lambda r:(float(r['K1_prior_ADE_m']),int(r['epoch'])))
            assert int(best['epoch'])==selection['best']['epoch']
            labels=np.array([int(r['category_index']) for r in read_csv(ROOT/f'pools/{rec}_train.csv.gz')])
            cumulative=np.zeros(len(labels),dtype=np.int64)
            for epoch in range(1,31):
                counts,classes=exposure(labels,epoch_order(labels,condition,epoch));cumulative+=counts
                visits=read_csv(directory/f'epochs/epoch{epoch:03d}_target_visits.csv.gz')
                assert len(visits)==len(labels)
                assert np.array_equal([int(r['visits']) for r in visits],counts)
                assert np.array_equal([int(r['cumulative_visits']) for r in visits],cumulative)
                c=curves[epoch-1]
                assert int(c['targets_seen'])==len(labels) and int(c['steps'])==int(np.ceil(len(labels)/128))
            curve_stats.append(dict(recording=rec,condition=condition,best_epoch=selection['best']['epoch'],
                best_at_budget_boundary=selection['best_at_budget_boundary'],last_epoch=30,
                run_wall_seconds=selection['actual_run_wall_seconds'],parameters=provenance['initial_parameter_state']['parameter_count'],
                effective_parameters=provenance['model_effective_parameter_upper_bound'],
                total_target_visits=selection['total_target_visits'],total_optimizer_steps=selection['total_optimizer_steps'],
                peak_allocated_bytes=max(int(r['peak_allocated_bytes']) for r in curves),
                train_loss_first=float(curves[0]['train_loss']),train_loss_last=float(curves[-1]['train_loss']),
                train_loss_last10_mean=float(np.mean([float(r['train_loss']) for r in curves[-10:]])),
                validation_ADE_first=float(curves[0]['K1_prior_ADE_m']),validation_ADE_last=float(curves[-1]['K1_prior_ADE_m']),
                best_train_ADE=selection['best_train_K1']['train_K1_prior_ADE_m'],best_train_FDE=selection['best_train_K1']['train_K1_prior_FDE_m']))
            for which in ['best','last']:
                cp=selection[which];assert sha256(cp['path'])==cp['sha256']
                stored=torch.load(cp['path'],map_location='cpu',weights_only=True)
                model=new_model();model.load_state_dict(stored['model'],strict=True);model.eval()
                raw=read_csv(directory/('best_paired_targets.csv.gz' if which=='best' else 'epochs/epoch030_targets.csv.gz'))
                refs={key(r):r for r in raw};assert len(refs)==len(ds)
                with torch.no_grad():
                    for first in range(0,len(ds),128):
                        items=[ds[i] for i in range(first,min(first+128,len(ds)))];x,y,n=collate_examples(items,'cuda')
                        predictions=model(x,n)
                        for j,item in enumerate(items):
                            identity=key(item);ref=refs[identity];pool=pools[rec][first+j]
                            assert key(pool)==identity and array_sha(item['y'])==pool['future_sha256']
                            assert array_sha(item['x'][:,:2])==pool['history_sha256']
                            d,ss=diagnose(item['x'][:,:2],item['y'],predictions[:,j].cpu().numpy())
                            delta=max(abs(d['ADE_m']-float(ref['K1_prior_ADE_m'])),abs(d['FDE_m']-float(ref['K1_prior_FDE_m'])))
                            max_difference=max(max_difference,delta);assert delta<2e-6
                            common=dict(recording=rec,start_frame=identity[1],target_id=identity[2],condition=condition,selection=which,
                                history_sha256=pool['history_sha256'],future_sha256=pool['future_sha256'],groups=features[identity]['groups'])
                            metrics.append(dict(**common,**{k:float(ref[k]) for pair in FAMILIES.values() for k in pair},
                                prior_seed=int(ref['prior_seed']),posterior_seed=int(ref['posterior_seed'])))
                            diagnostics.append(dict(**common,**d,absolute_target_speed_bias_mps=abs(d['speed_bias_mps']),
                                mean_absolute_step_speed_error_mps=float(np.mean([abs(s['speed_bias_mps']) for s in ss]))))
                            steps.extend(dict(**common,**s) for s in ss)
                            packets.write((json.dumps(dict(**common,history=item['x'][:,:2].tolist(),truth=item['y'].tolist(),
                                prediction=predictions[:,j].cpu().tolist()),allow_nan=False)+'\n').encode())
                index.append(dict(recording=rec,condition=condition,selection=which,epoch=cp['epoch'],checkpoint=cp['path'],
                    checkpoint_sha256=cp['sha256'],training_code_commit=provenance['code_commit'],manifest_sha256=manifest['sha256']))
                del model,stored;torch.cuda.empty_cache()
            checks.append(dict(recording=rec,condition=condition,all_30_epoch_visits_match_frozen_sampler=True,
                source_pool_identical=True,source_labels_history_only=True,all_GT_history_IDs_verified=True,
                total_target_visits=selection['total_target_visits'],total_optimizer_steps=selection['total_optimizer_steps'],
                initial_parameter_state_sha256=selection['initial_parameter_state']['parameter_state_sha256']))
    packets.close()
    by_identity={(r['recording'],r['condition'],r['selection'],r['start_frame'],r['target_id']):r for r in metrics}
    pairs=[];overall=[];group_rows=[]
    for rec in datasets:
        for which in ['best','last']:
            ur=[r for r in metrics if r['recording']==rec and r['condition']=='Uniform' and r['selection']==which]
            for u in ur:
                s=by_identity[(rec,'Stratified',which,u['start_frame'],u['target_id'])]
                assert (u['history_sha256'],u['future_sha256'],u['prior_seed'],u['posterior_seed'])==(s['history_sha256'],s['future_sha256'],s['prior_seed'],s['posterior_seed'])
                row={k:u[k] for k in ['recording','start_frame','target_id','selection','history_sha256','future_sha256','prior_seed','posterior_seed','groups']}
                for field in [x for pair in FAMILIES.values() for x in pair]:
                    row.update({f'Uniform_{field}':u[field],f'Stratified_{field}':s[field],f'delta_{field}':s[field]-u[field]})
                pairs.append(row)
            sr=[r for r in metrics if r['recording']==rec and r['condition']=='Stratified' and r['selection']==which]
            for group in GROUPS:
                a=[r for r in ur if group in r['groups'].split('|')];b=[r for r in sr if group in r['groups'].split('|')]
                assert {key(r) for r in a}=={key(r) for r in b}
                for weighting in WEIGHTS:
                    for family,(ade,fde) in FAMILIES.items():
                        row=dict(recording=rec,selection=which,group=group,weighting=weighting,metric_family=family,
                            targets=len(a),people=len({r['target_id'] for r in a}),
                            Uniform_ADE_m=average(a,ade,weighting),Uniform_FDE_m=average(a,fde,weighting),
                            Stratified_ADE_m=average(b,ade,weighting),Stratified_FDE_m=average(b,fde,weighting))
                        row['delta_ADE_m']=row['Stratified_ADE_m']-row['Uniform_ADE_m'] if a else None
                        row['delta_FDE_m']=row['Stratified_FDE_m']-row['Uniform_FDE_m'] if a else None
                        row['small_group']=len(a)<30 or row['people']<10
                        group_rows.append(row)
                        if group=='all':overall.append(row)
    for which in ['best','last']:
        for weighting in WEIGHTS:
            for family in FAMILIES:
                sub=[r for r in overall if r['selection']==which and r['weighting']==weighting and r['metric_family']==family]
                assert len(sub)==2
                macro=dict(sub[0]);macro.update(recording='two-recording-macro',targets=sum(r['targets'] for r in sub),people=sum(r['people'] for r in sub))
                for f in ['Uniform_ADE_m','Uniform_FDE_m','Stratified_ADE_m','Stratified_FDE_m','delta_ADE_m','delta_FDE_m']:macro[f]=float(np.mean([r[f] for r in sub]))
                overall.append(macro)
    motion_groups=[];diag_fields=['ADE_m','FDE_m','predicted_mean_speed_mps','truth_mean_speed_mps','speed_bias_mps',
        'absolute_target_speed_bias_mps','mean_absolute_step_speed_error_mps','endpoint_along_error_m','endpoint_cross_error_m']
    for rec in datasets:
        for condition in CONDITIONS:
            for which in ['best','last']:
                for group in GROUPS:
                    rr=[r for r in diagnostics if r['recording']==rec and r['condition']==condition and r['selection']==which and group in r['groups'].split('|')]
                    for weighting in WEIGHTS:
                        motion_groups.append(dict(recording=rec,condition=condition,selection=which,group=group,weighting=weighting,
                            targets=len(rr),people=len({r['target_id'] for r in rr}),reliable_heading_targets=sum(r['history_heading_reliable'] for r in rr),
                            **{f:average(rr,f,weighting) for f in diag_fields}))
    horizon=[]
    for rec in datasets:
        for condition in CONDITIONS:
            for which in ['best','last']:
                for group in ['all','speed_stationary','speed_low','speed_normal','speed_fast','turning','direction_change']:
                    for step in range(1,13):
                        rr=[r for r in steps if r['recording']==rec and r['condition']==condition and r['selection']==which and r['step']==step and group in r['groups'].split('|')]
                        for weighting in WEIGHTS:
                            horizon.append(dict(recording=rec,condition=condition,selection=which,group=group,step=step,seconds=step*.4,
                                weighting=weighting,targets=len(rr),people=len({r['target_id'] for r in rr}),
                                **{f:average(rr,f,weighting) for f in ['error_m','predicted_speed_mps','truth_speed_mps','speed_bias_mps','along_error_m','cross_error_m']}))
    normal=[r for r in motion_groups if r['recording']=='biwi_hotel' and r['selection']=='best' and r['group']=='speed_normal' and r['weighting']=='target-window']
    speed={r['condition']+'_signed_bias_mps':r['speed_bias_mps'] for r in normal}
    primary=[r for r in overall if r['selection']=='best' and r['metric_family']=='K1-prior' and r['recording']!='two-recording-macro']
    stationary=[r for r in group_rows if r['selection']=='best' and r['metric_family']=='K1-prior' and r['group']=='speed_stationary']
    integrity=True
    for rec in datasets:
        a,b=[r for r in checks if r['recording']==rec]
        assert a['total_target_visits']==b['total_target_visits'] and a['total_optimizer_steps']==b['total_optimizer_steps']
        assert a['initial_parameter_state_sha256']==b['initial_parameter_state_sha256']
    result=decision(primary,speed,stationary,integrity)
    for name,rows in [('all_target_metrics.csv.gz',metrics),('paired_target_errors.csv.gz',pairs),('motion_diagnostics.csv.gz',diagnostics),
        ('per_step_motion.csv.gz',steps),('paired_overall_metrics.csv',overall),('all_history_group_metrics.csv',group_rows),
        ('all_motion_group_diagnostics.csv',motion_groups),('horizon_group_metrics.csv',horizon),('run_summary.csv',curve_stats),('checkpoint_index.csv',index)]:
        csv_new(OUT/name,rows)
    json_new(OUT/'budget_and_pairing_checks.json',checks)
    json_new(OUT/'research_decision.json',result)
    json_new(OUT/'execution.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for p in [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
            Path('src/motion_sampling/policy.py'),Path('src/evaluation/stage6_motion.py')]},
        maximum_diagnostic_error_match_difference_m=max_difference,trajectory_packet=str(packet),trajectory_packet_sha256=sha256(packet),
        ETH_evaluated=False,optimizer_updates=0,selection_changed=False,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
