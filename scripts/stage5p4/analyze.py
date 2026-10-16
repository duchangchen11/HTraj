"""Read-only frozen selections, matched targets, all groups and both weightings."""
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
from scripts.stage5p3.analyze import key,means,FAMILIES
ROOT=Path('results/tables/stage5p4')
RECS=['biwi_hotel','crowds_zara02']


def main():
    models={'Full':Path('results/tables/stage5p2'),'NoSocial':Path('results/tables/stage5p3'),
            'EgoCapacity':ROOT/'ego_capacity'}
    if (ROOT/'placebo_social/biwi_hotel/selection.json').exists():
        models['PlaceboSocial']=ROOT/'placebo_social'
    out=ROOT/('analysis_final' if 'PlaceboSocial' in models else 'analysis_phase1');out.mkdir()
    features={key(r):r for r in read_csv(ROOT/'history_features.csv.gz')}
    seeds={key(r):r for r in read_csv(ROOT/'sampling_seeds.csv.gz')}
    cv=read_csv(ROOT/'cv_paired_targets.csv.gz');last2={key(r):r for r in cv if r['model']=='CV-Last2'}
    rows=[];refs=[];convergence=[]
    def add(rec,start,actor,model,selection,family,a,f,gt):
        k=(rec,int(start),int(actor));base=last2[k]
        assert gt==features[k]['future_position_sha256']
        rows.append(dict(recording=rec,start_frame=k[1],target_id=k[2],model=model,selection=selection,
            metric_family=family,ADE_m=float(a),FDE_m=float(f),CV_Last2_ADE_m=float(base['ADE_m']),CV_Last2_FDE_m=float(base['FDE_m']),
            delta_ADE_vs_CV_Last2_m=float(a)-float(base['ADE_m']),delta_FDE_vs_CV_Last2_m=float(f)-float(base['FDE_m']),
            future_position_sha256=gt,groups=features[k]['groups']))
    for r in cv:add(r['recording'],r['start_frame'],r['target_id'],r['model'],'analytic','deterministic',r['ADE_m'],r['FDE_m'],r['future_position_sha256'])
    for model,directory in models.items():
        for rec in RECS:
            d=directory/rec;selection=json.load(open(d/'selection.json'))
            assert selection['training_completed'] and selection['completed_epochs']==100
            curve=[json.load(open(d/f'epochs/epoch{e:03d}.json')) for e in range(1,101)]
            assert min(curve,key=lambda c:(c['K1_prior_ADE_m'],c['epoch']))['epoch']==selection['best']['epoch']
            for which in ['best','last']:
                checkpoint=selection[which];assert sha256(checkpoint['path'])==checkpoint['sha256']
                source=d/f"epochs/epoch{checkpoint['epoch']:03d}_targets.csv.gz"
                targets=read_csv(source)
                assert {key(r) for r in targets}=={k for k in features if k[0]==rec}
                refs.append(dict(model=model,recording=rec,selection=which,epoch=checkpoint['epoch'],source=str(source),
                    source_sha256=sha256(source),checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],
                    code_commit=selection['code_commit']))
                for r in targets:
                    k=key(r)
                    assert r['prior_seed']==seeds[k]['prior_seed'] and r['posterior_seed']==seeds[k]['posterior_seed']
                    for family,(a,f) in FAMILIES.items():
                        add(rec,k[1],k[2],model,which,family,r[a],r[f],features[k]['future_position_sha256'])
            fields=['train_loss','train_rec','train_kl','K1_prior_ADE_m','K1_prior_FDE_m','posterior_loss','minADE20_m','minFDE20_m']
            convergence.append(dict(model=model,recording=rec,best_epoch=selection['best']['epoch'],
                best_checkpoint=selection['best'],last_checkpoint=selection['last'],best_train_K1=selection['best_train_K1'],
                best_validation=selection['best_validation'],first_epoch=curve[0],last_epoch=curve[-1],
                last20={f:dict(epoch81_90=float(np.mean([c[f] for c in curve[80:90]])),
                    epoch91_100=float(np.mean([c[f] for c in curve[90:100]])),
                    slope_per_epoch=float(np.polyfit(np.arange(81,101),[c[f] for c in curve[80:]],1)[0])) for f in fields},
                full_budget_completed=True,optimizer_steps=selection['total_optimizer_steps'],
                actual_run_wall_seconds=selection['actual_run_wall_seconds'],
                sum_training_compute_seconds=sum(c['train_compute_seconds'] for c in curve),
                peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in curve),
                peak_reserved_bytes=max(c['peak_reserved_bytes'] for c in curve),
                gradient_L2_max=max(c['gradient_L2_max'] for c in curve),
                all_parameters_finite=all(c['parameters_finite'] for c in curve),code_commit=selection['code_commit']))
    csv_new(out/'model_references.csv',refs)
    csv_new(out/'paired_model_errors.csv.gz',rows)
    variants=sorted({(r['model'],r['selection'],r['metric_family']) for r in rows})
    summaries=[]
    for rec in RECS:
        for model,selection,family in variants:
            cohort=[r for r in rows if r['recording']==rec and (r['model'],r['selection'],r['metric_family'])==(model,selection,family)]
            assert len(cohort)==sum(k[0]==rec for k in features)
            for group in GROUPS:
                subset=[r for r in cohort if group in r['groups'].split('|')]
                for weighting in ['target-window','equal-person-ID']:
                    a,f=means(subset,weighting)
                    base=[dict(recording=r['recording'],target_id=r['target_id'],ADE_m=r['CV_Last2_ADE_m'],FDE_m=r['CV_Last2_FDE_m']) for r in subset]
                    ba,bf=means(base,weighting)
                    summaries.append(dict(recording=rec,model=model,selection=selection,metric_family=family,group=group,weighting=weighting,
                        target_windows=len(subset),people=len({r['target_id'] for r in subset}),ADE_m=a,FDE_m=f,
                        delta_ADE_vs_CV_Last2_m=None if a is None else a-ba,delta_FDE_vs_CV_Last2_m=None if f is None else f-bf))
    csv_new(out/'all_group_metrics.csv',summaries)
    csv_new(out/'model_metrics.csv',[r for r in summaries if r['group']=='all'])
    macros=[]
    for model,selection,family in variants:
        for group in GROUPS:
            for weighting in ['target-window','equal-person-ID']:
                rr=[r for r in summaries if (r['model'],r['selection'],r['metric_family'],r['group'],r['weighting'])==(model,selection,family,group,weighting) and r['ADE_m'] is not None]
                macros.append(dict(model=model,selection=selection,metric_family=family,group=group,within_recording_weighting=weighting,
                    recordings_with_targets=len(rr),two_recording_macro_available=len(rr)==2,
                    **{field:float(np.mean([r[field] for r in rr])) if len(rr)==2 else None for field in ['ADE_m','FDE_m','delta_ADE_vs_CV_Last2_m','delta_FDE_vs_CV_Last2_m']}))
    csv_new(out/'recording_macro_metrics.csv',macros)
    pooled=[]
    for model,selection,family in variants:
        rr=[r for r in rows if (r['model'],r['selection'],r['metric_family'])==(model,selection,family)]
        for weighting in ['target-window','equal-person-ID']:
            a,f=means(rr,weighting)
            pooled.append(dict(model=model,selection=selection,metric_family=family,weighting=weighting,
                target_windows=len(rr),people=len({(r['recording'],r['target_id']) for r in rr}),
                ADE_m=a,FDE_m=f,label='pooled across exposed recordings; NOT equal-recording macro'))
    csv_new(out/'pooled_metrics.csv',pooled)
    people=defaultdict(list)
    for r in rows:people[(r['recording'],r['target_id'],r['model'],r['selection'],r['metric_family'])].append(r)
    csv_new(out/'person_aggregated_errors.csv',[
        dict(recording=k[0],target_id=k[1],model=k[2],selection=k[3],metric_family=k[4],target_windows=len(rr),
            **{field:float(np.mean([r[field] for r in rr])) for field in ['ADE_m','FDE_m','delta_ADE_vs_CV_Last2_m','delta_FDE_vs_CV_Last2_m']}) for k,rr in sorted(people.items())])
    json_new(out/'convergence_audit.json',convergence)
    # All pairwise same-target deltas to Full, including CV and best/last distributions.
    full={(key(r),r['selection'],r['metric_family']):r for r in rows if r['model']=='Full'}
    effects=[]
    for r in rows:
        if r['model']=='Full':continue
        selection='best' if r['selection']=='analytic' else r['selection']
        family='K1-prior' if r['metric_family']=='deterministic' else r['metric_family']
        f=full[(key(r),selection,family)]
        effects.append(dict(recording=r['recording'],start_frame=r['start_frame'],target_id=r['target_id'],
            model=r['model'],selection=r['selection'],metric_family=r['metric_family'],
            model_ADE_m=r['ADE_m'],model_FDE_m=r['FDE_m'],Full_ADE_m=f['ADE_m'],Full_FDE_m=f['FDE_m'],
            model_minus_Full_ADE_m=r['ADE_m']-f['ADE_m'],model_minus_Full_FDE_m=r['FDE_m']-f['FDE_m'],groups=r['groups']))
    csv_new(out/'paired_differences_vs_Full.csv.gz',effects)
    print('Analyzed models:',list(models),'and all three CVs')

if __name__=='__main__':main()
