"""Paired Full-minus-NoSocial effects, all histories, no window-iid tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import json
import numpy as np
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.groups import GROUPS
from src.socialvae_baseline.artifacts import json_new
from scripts.stage5p3.analyze import means
ROOT=Path('results/tables/stage5p3')
rows=read_csv(ROOT/'paired_model_errors.csv.gz')
full={(r['recording'],r['start_frame'],r['target_id'],r['selection'],r['metric_family']):r for r in rows if r['model']=='Full'}
paired=[]
for r in rows:
    if r['model']!='NoSocial':continue
    key=(r['recording'],r['start_frame'],r['target_id'],r['selection'],r['metric_family'])
    f=full[key]
    assert f['future_position_sha256']==r['future_position_sha256'] and f['groups']==r['groups']
    paired.append(dict(recording=r['recording'],start_frame=int(r['start_frame']),target_id=int(r['target_id']),selection=r['selection'],metric_family=r['metric_family'],
        Full_ADE_m=float(f['ADE_m']),Full_FDE_m=float(f['FDE_m']),NoSocial_ADE_m=float(r['ADE_m']),NoSocial_FDE_m=float(r['FDE_m']),
        Full_minus_NoSocial_ADE_m=float(f['ADE_m'])-float(r['ADE_m']),Full_minus_NoSocial_FDE_m=float(f['FDE_m'])-float(r['FDE_m']),groups=r['groups']))
csv_new(ROOT/'paired_social_effects.csv.gz',paired)
summary=[]
for rec in ['biwi_hotel','crowds_zara02']:
    for which in ['best','last']:
        for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
            for group in GROUPS:
                subset=[r for r in paired if r['recording']==rec and r['selection']==which and r['metric_family']==family and group in r['groups'].split('|')]
                for weighting in ['target-window','equal-person-ID']:
                    F=[dict(recording=r['recording'],target_id=r['target_id'],ADE_m=r['Full_ADE_m'],FDE_m=r['Full_FDE_m']) for r in subset]
                    N=[dict(recording=r['recording'],target_id=r['target_id'],ADE_m=r['NoSocial_ADE_m'],FDE_m=r['NoSocial_FDE_m']) for r in subset]
                    fa,ff=means(F,weighting);na,nf=means(N,weighting)
                    summary.append(dict(recording=rec,selection=which,metric_family=family,group=group,weighting=weighting,
                        target_windows=len(subset),people=len({r['target_id'] for r in subset}),Full_ADE_m=fa,Full_FDE_m=ff,NoSocial_ADE_m=na,NoSocial_FDE_m=nf,
                        Full_minus_NoSocial_ADE_m=None if fa is None else fa-na,Full_minus_NoSocial_FDE_m=None if ff is None else ff-nf,
                        Full_relative_ADE_gain=None if na is None or na==0 else (na-fa)/na,
                        Full_relative_FDE_regression=None if nf is None or nf==0 else (ff-nf)/nf))
csv_new(ROOT/'social_effects_all_groups.csv',summary)
json_new(ROOT/'social_effects_overall.json',[r for r in summary if r['group']=='all'])
plan=json.load(open('configs/stage5p3/protocol.json'));rule=plan['stopping']
primary=[r for r in summary if r['group']=='all' and r['selection']=='best' and r['metric_family']=='K1-prior']
qualified=[];any_improved=[]
for rec in ['biwi_hotel','crowds_zara02']:
    rr=[r for r in primary if r['recording']==rec]
    if all(r['Full_relative_ADE_gain']>=rule['useful_ADE_relative_gain'] and r['Full_relative_FDE_regression']<=rule['meaningful_FDE_regression_relative'] for r in rr):qualified.append(rec)
    if any(r['Full_minus_NoSocial_ADE_m']<0 for r in rr):any_improved.append(rec)
verdict='worth_followup_capacity_unresolved' if len(qualified)==2 else 'conditional_or_weighting_conflict' if any_improved else 'stop_complex_social_modules'
json_new(ROOT/'stopping_decision.json',dict(rule=rule,qualified_recordings_both_weightings=qualified,
    any_K1_ADE_improvement_recordings=any_improved,verdict=verdict,capacity_confound_resolved=False,
    Ego_Capacity_Control_trained=False,additional_seeds=False,additional_training=False,ETH_evaluated=False))
macros=[]
for which in ['best','last']:
    for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
        for group in GROUPS:
            for weighting in ['target-window','equal-person-ID']:
                rr=[r for r in summary if r['selection']==which and r['metric_family']==family and r['group']==group and r['weighting']==weighting and r['target_windows']>0]
                fields=['Full_ADE_m','Full_FDE_m','NoSocial_ADE_m','NoSocial_FDE_m','Full_minus_NoSocial_ADE_m','Full_minus_NoSocial_FDE_m']
                macros.append(dict(selection=which,metric_family=family,group=group,within_recording_weighting=weighting,
                    recordings_with_targets=len(rr),two_recording_macro_available=len(rr)==2,
                    **{k:float(np.mean([r[k] for r in rr])) if len(rr)==2 else None for k in fields}))
csv_new(ROOT/'social_effects_recording_macro.csv',macros)
