"""Post-training only: all cohorts/groups, fixed selections and paired CVs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from collections import defaultdict
import numpy as np
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_baseline.groups import GROUPS
from src.data.preprocessing import sha256

ROOT=Path('results/tables/stage5p2')
RECS=['biwi_hotel','crowds_zara02']
FAMILIES={'K1-prior':('K1_prior_ADE_m','K1_prior_FDE_m'),
          'K1-first-sample':('K1_sample_ADE_m','K1_sample_FDE_m'),
          'K20-independent-minima':('minADE20_m','minFDE20_m')}


def key(row):return row['recording'],int(row['start_frame']),int(row['target_id'])


def means(rows,weighting):
    if not rows:return None,None
    if weighting=='target-window':return float(np.mean([r['ADE_m'] for r in rows])),float(np.mean([r['FDE_m'] for r in rows]))
    people=defaultdict(list)
    for r in rows:people[(r['recording'],r['target_id'])].append(r)
    return (float(np.mean([np.mean([r['ADE_m'] for r in p]) for p in people.values()])),
            float(np.mean([np.mean([r['FDE_m'] for r in p]) for p in people.values()])))


def main():
    features={key(r):r for r in read_csv(ROOT/'history_features.csv.gz')}
    cv=read_csv(ROOT/'cv_paired_targets.csv.gz')
    last2={key(r):r for r in cv if r['model']=='CV-Last2'}
    rows=[];convergence={};checkpoint_audits=[]
    for r in cv:
        rows.append(dict(recording=r['recording'],start_frame=int(r['start_frame']),target_id=int(r['target_id']),
            model=r['model'],selection='analytic',metric_family='deterministic',ADE_m=float(r['ADE_m']),FDE_m=float(r['FDE_m']),
            CV_Last2_ADE_m=float(last2[key(r)]['ADE_m']),CV_Last2_FDE_m=float(last2[key(r)]['FDE_m']),
            delta_ADE_vs_CV_Last2_m=float(r['ADE_m'])-float(last2[key(r)]['ADE_m']),
            delta_FDE_vs_CV_Last2_m=float(r['FDE_m'])-float(last2[key(r)]['FDE_m']),
            future_position_sha256=r['future_position_sha256'],groups=features[key(r)]['groups']))
    for rec in RECS:
        run=ROOT/rec;selection=json.load(open(run/'selection.json'))
        assert selection['training_completed'] and selection['completed_epochs']==100
        curves=[json.load(open(run/f'epochs/epoch{i:03d}.json')) for i in range(1,101)]
        assert all(c['targets_seen']==selection['training_targets_per_epoch'] for c in curves)
        assert min(curves,key=lambda c:(c['K1_prior_ADE_m'],c['epoch']))['epoch']==selection['best']['epoch']
        for c in read_csv(run/'checkpoint_inventory.csv'):
            assert sha256(c['path'])==c['sha256']
            checkpoint_audits.append(dict(recording=rec,epoch=int(c['epoch']),path=c['path'],sha256=c['sha256'],verified=True))
        for which in ['best','last']:
            epoch=selection[which]['epoch']
            targets=read_csv(run/f'epochs/epoch{epoch:03d}_targets.csv.gz')
            assert {key(r) for r in targets}=={k for k in features if k[0]==rec}
            for target in targets:
                k=key(target);baseline=last2[k]
                for family,(ak,fk) in FAMILIES.items():
                    a,f=float(target[ak]),float(target[fk])
                    rows.append(dict(recording=rec,start_frame=k[1],target_id=k[2],model='SocialVAE',selection=which,
                        metric_family=family,ADE_m=a,FDE_m=f,CV_Last2_ADE_m=float(baseline['ADE_m']),CV_Last2_FDE_m=float(baseline['FDE_m']),
                        delta_ADE_vs_CV_Last2_m=a-float(baseline['ADE_m']),delta_FDE_vs_CV_Last2_m=f-float(baseline['FDE_m']),
                        future_position_sha256=features[k]['future_position_sha256'],groups=features[k]['groups']))
        def average(start,end,field):return float(np.mean([c[field] for c in curves[start-1:end]]))
        fields=['train_loss','train_rec','train_kl','K1_prior_ADE_m','K1_prior_FDE_m','posterior_loss','minADE20_m','minFDE20_m']
        convergence[rec]=dict(best_epoch=selection['best']['epoch'],best_checkpoint=selection['best'],last_checkpoint=selection['last'],
            best_train_K1=selection['best_train_K1'],best_validation=selection['best_validation'],last_epoch=curves[-1],
            first_epoch=curves[0],best_in_last10=selection['best']['epoch']>=91,
            last20={f:dict(epoch81_90=average(81,90,f),epoch91_100=average(91,100,f),
                difference=average(91,100,f)-average(81,90,f),
                slope_per_epoch=float(np.polyfit(np.arange(81,101),[c[f] for c in curves[80:]],1)[0])) for f in fields},
            full_budget_completed=True,optimizer_steps=selection['total_optimizer_steps'],
            actual_run_wall_seconds=selection['actual_run_wall_seconds'],sum_training_compute_seconds=sum(c['train_compute_seconds'] for c in curves),
            peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in curves),peak_reserved_bytes=max(c['peak_reserved_bytes'] for c in curves),
            gradient_L2_max=max(c['gradient_L2_max'] for c in curves),
            all_parameters_finite=all(c['parameters_finite'] for c in curves),code_commit=selection['code_commit'])
    csv_new(ROOT/'paired_model_errors.csv.gz',rows)
    variants=sorted({(r['model'],r['selection'],r['metric_family']) for r in rows})
    summaries=[]
    for rec in RECS:
        for model,selection,family in variants:
            cohort=[r for r in rows if r['recording']==rec and (r['model'],r['selection'],r['metric_family'])==(model,selection,family)]
            assert len(cohort)==len([k for k in features if k[0]==rec])
            for group in GROUPS:
                subset=[r for r in cohort if group in r['groups'].split('|')]
                for weighting in ['target-window','equal-person-ID']:
                    a,f=means(subset,weighting)
                    base=[dict(recording=r['recording'],target_id=r['target_id'],ADE_m=r['CV_Last2_ADE_m'],FDE_m=r['CV_Last2_FDE_m']) for r in subset]
                    ba,bf=means(base,weighting)
                    summaries.append(dict(recording=rec,model=model,selection=selection,metric_family=family,group=group,weighting=weighting,
                        target_windows=len(subset),people=len({r['target_id'] for r in subset}),ADE_m=a,FDE_m=f,
                        delta_ADE_vs_CV_Last2_m=None if a is None else a-ba,delta_FDE_vs_CV_Last2_m=None if f is None else f-bf))
    csv_new(ROOT/'all_group_metrics.csv',summaries)
    csv_new(ROOT/'model_metrics.csv',[r for r in summaries if r['group']=='all'])
    macros=[]
    for model,selection,family in variants:
        for group in GROUPS:
            for weighting in ['target-window','equal-person-ID']:
                rr=[r for r in summaries if r['model']==model and r['selection']==selection and r['metric_family']==family and r['group']==group and r['weighting']==weighting]
                valid=[r for r in rr if r['ADE_m'] is not None]
                macros.append(dict(model=model,selection=selection,metric_family=family,group=group,within_recording_weighting=weighting,
                    recordings_with_targets=len(valid),two_recording_macro_available=len(valid)==2,
                    ADE_m=float(np.mean([r['ADE_m'] for r in valid])) if len(valid)==2 else None,
                    FDE_m=float(np.mean([r['FDE_m'] for r in valid])) if len(valid)==2 else None,
                    delta_ADE_vs_CV_Last2_m=float(np.mean([r['delta_ADE_vs_CV_Last2_m'] for r in valid])) if len(valid)==2 else None,
                    delta_FDE_vs_CV_Last2_m=float(np.mean([r['delta_FDE_vs_CV_Last2_m'] for r in valid])) if len(valid)==2 else None))
    csv_new(ROOT/'recording_macro_metrics.csv',macros)
    # Per-person means retain ID counts and the overlapping-window structure.
    people=defaultdict(list)
    for r in rows:people[(r['recording'],r['target_id'],r['model'],r['selection'],r['metric_family'])].append(r)
    csv_new(ROOT/'person_aggregated_errors.csv',[
        dict(recording=k[0],target_id=k[1],model=k[2],selection=k[3],metric_family=k[4],target_windows=len(rr),
            ADE_m=float(np.mean([r['ADE_m'] for r in rr])),FDE_m=float(np.mean([r['FDE_m'] for r in rr])),
            delta_ADE_vs_CV_Last2_m=float(np.mean([r['delta_ADE_vs_CV_Last2_m'] for r in rr])),
            delta_FDE_vs_CV_Last2_m=float(np.mean([r['delta_FDE_vs_CV_Last2_m'] for r in rr]))) for k,rr in sorted(people.items())])
    oracle=[]
    for rec in RECS:
        rr=[r for r in summaries if r['recording']==rec and r['group']=='all' and r['weighting']=='target-window' and r['model'].startswith('CV-')]
        best_a=min(rr,key=lambda r:r['ADE_m']);best_f=min(rr,key=lambda r:r['FDE_m'])
        oracle.append(dict(recording=rec,label='解析基线上界参考：事后逐录像选择，非可部署模型',
            minimum_ADE_model=best_a['model'],minimum_ADE_m=best_a['ADE_m'],minimum_FDE_model=best_f['model'],minimum_FDE_m=best_f['FDE_m']))
    json_new(ROOT/'analytic_oracle_reference.json',oracle)
    json_new(ROOT/'convergence_audit.json',convergence)
    csv_new(ROOT/'checkpoint_sha_verification.csv',checkpoint_audits)
    print(json.dumps({rec:dict(best_epoch=c['best_epoch'],K1_ADE=c['best_validation']['K1_prior_ADE_m'],
        K1_FDE=c['best_validation']['K1_prior_FDE_m'],minADE20=c['best_validation']['minADE20_m'],
        minFDE20=c['best_validation']['minFDE20_m']) for rec,c in convergence.items()},indent=2))


if __name__=='__main__':main()
