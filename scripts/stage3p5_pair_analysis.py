"""Same-target comparisons, group consistency and temporal-support sensitivity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,subprocess
from collections import defaultdict
import numpy as np
from src.training.social_diagnostics import csv_new
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.benchmark_audit import GROUPS

ROOT=Path('results/tables/stage3p5')


def read_targets(path):
    rows=list(csv.DictReader(Path(path).open()));out={}
    for r in rows:
        key=(r['recording'],r['sample_id'],int(r['target_id']))
        assert key not in out
        for k in ('ADE_m','FDE_m','history_frames','continuous_motion_frames','mean_neighbor_degree','future_points'):
            r[k]=float(r[k]) if r[k] else None
        out[key]=r
    return out


def compare(setting,seed,models,endpoint,temporal_subset=None):
    assert set(models['CV'])==set(models['PedBase'])==set(models['Dynamic-GNN-RN'])
    paired=[];rows=[]
    for key,b in models['PedBase'].items():
        d=models['Dynamic-GNN-RN'][key];c=models['CV'][key]
        assert b['future_points']==d['future_points']==c['future_points']
        record,sample,person=key
        paired.append(dict(setting=setting,seed=seed,endpoint=endpoint,recording=record,sample_id=sample,target_id=person,
            ADE_PedBase=b['ADE_m'],ADE_Dynamic=d['ADE_m'],ADE_CV=c['ADE_m'],delta_Dynamic_minus_PedBase_ADE=d['ADE_m']-b['ADE_m'],
            delta_Dynamic_minus_CV_ADE=d['ADE_m']-c['ADE_m'],FDE_PedBase=b['FDE_m'],FDE_Dynamic=d['FDE_m'],FDE_CV=c['FDE_m'],
            delta_Dynamic_minus_PedBase_FDE=d['FDE_m']-b['FDE_m'] if b['FDE_m'] is not None else None,
            history_frames=b['history_frames'],motion_frames=b['continuous_motion_frames'],mean_degree=b['mean_neighbor_degree']))
    group_masks={g:[k for k,b in models['PedBase'].items() if b[g]=='True'] for g in GROUPS}
    group_masks.update({'recording:'+r:[k for k in models['PedBase'] if k[0]==r] for r in {k[0] for k in models['PedBase']}})
    if temporal_subset is not None:group_masks['nonoverlap_support_guard8s']=[k for k in models['PedBase'] if k[1] in temporal_subset]
    for group,keys in group_masks.items():
        if not keys:
            rows.append(dict(setting=setting,seed=seed,endpoint=endpoint,group=group,ADE_count=0,FDE_count=0));continue
        b=[models['PedBase'][k] for k in keys];d=[models['Dynamic-GNN-RN'][k] for k in keys];c=[models['CV'][k] for k in keys]
        final=[i for i,r in enumerate(b) if r['FDE_m'] is not None];delta=np.array([r['ADE_m']-s['ADE_m'] for r,s in zip(d,b)])
        row=dict(setting=setting,seed=seed,endpoint=endpoint,group=group,ADE_count=len(keys),FDE_count=len(final),unique_people=len({(k[0],k[2]) for k in keys}),
            history_frames_mean=np.mean([r['history_frames'] for r in b]),motion_frames_mean=np.mean([r['continuous_motion_frames'] for r in b]),
            degree_mean=np.mean([r['mean_neighbor_degree'] for r in b]),future_points_mean=np.mean([r['future_points'] for r in b]),
            delta_Dynamic_minus_PedBase_ADE_m=float(delta.mean()),paired_ADE_improvement_fraction=float((delta<0).mean()),paired_ADE_delta_median_m=float(np.median(delta)))
        for name,items in [('CV',c),('PedBase',b),('Dynamic',d)]:
            row[name+'_ADE_m']=float(np.mean([r['ADE_m'] for r in items]));row[name+'_FDE_m']=float(np.mean([items[i]['FDE_m'] for i in final])) if final else None
        row['delta_Dynamic_minus_CV_ADE_m']=row['Dynamic_ADE_m']-row['CV_ADE_m']
        row['delta_Dynamic_minus_PedBase_FDE_m']=row['Dynamic_FDE_m']-row['PedBase_FDE_m'] if final else None
        rows.append(row)
    return rows,paired


def temporal_sensitivity():
    m=json.loads(Path('configs/stage3p5/folds/val_students003.json').read_text());selected=[]
    for w in sorted(m['splits']['validation'],key=lambda w:(w['recording'],w['start_frame'])):
        if all(v['recording']!=w['recording'] or w['start_frame']>v['end_frame']+200 for v in selected):selected.append(w)
    write_json_new(ROOT/'nonoverlap_sensitivity.json',dict(selected_windows=selected,guard_ticks=200,guard_seconds=8,eligible_by='metadata only, no GT/metrics',
        interpretation='same development region thinning, not independent confirmation data',ETH_evaluated=False))
    return {w['sample_id'] for w in selected}


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'legacy';all_groups=[];all_pairs=[]
    if mode=='legacy':
        root=ROOT/'B_legacy_1024_96_frozen_analysis'
        for seed in [17,42,2026]:
            models={n:read_targets(root/n/f'seed_{0 if n=="CV" else seed}'/'targets_best.csv') for n in ('CV','PedBase','Dynamic-GNN-RN')}
            g,p=compare('B_legacy1024_96_frozen',seed,models,'best');all_groups.extend(g);all_pairs.extend(p)
        consistency=[]
        for group in sorted({r['group'] for r in all_groups}):
            rows=[r for r in all_groups if r['group']==group]
            if rows[0]['ADE_count']:
                consistency.append(dict(group=group,ADE_targets=rows[0]['ADE_count'],FDE_targets=rows[0]['FDE_count'],
                    mean_delta_Dynamic_minus_PedBase_ADE_m=np.mean([r['delta_Dynamic_minus_PedBase_ADE_m'] for r in rows]),
                    ADE_direction_better_seeds=sum(r['delta_Dynamic_minus_PedBase_ADE_m']<0 for r in rows),
                    FDE_direction_better_seeds=sum(r['delta_Dynamic_minus_PedBase_FDE_m']<0 for r in rows) if rows[0]['FDE_count'] else 0,
                    scope='three optimizer seeds on reused development data; no independent/generalization/causal claim'))
        csv_new(ROOT/'legacy_seed_direction_consistency.csv',consistency)
    elif mode=='pilot':
        support=temporal_sensitivity();protocols=json.loads(Path('configs/stage3p5/protocol.json').read_text())['protocols']
        cross=[]
        for view in protocols:
            for endpoint in ('best','last'):
                models={n:read_targets(ROOT/'pilot'/view/n/('targets_analytical.csv' if n=='CV' else f'targets_{endpoint}.csv')) for n in ('CV','PedBase','Dynamic-GNN-RN')}
                g,p=compare(view,17,models,endpoint,support);all_groups.extend(g);all_pairs.extend(p)
        for n in ('CV','PedBase','Dynamic-GNN-RN'):
            a=read_targets(ROOT/'pilot'/protocols[0]/n/('targets_analytical.csv' if n=='CV' else 'targets_best.csv'))
            b=read_targets(ROOT/'pilot'/protocols[1]/n/('targets_analytical.csv' if n=='CV' else 'targets_best.csv'))
            assert set(a)<=set(b)
            for k,r in a.items():
                assert r['future_points']==b[k]['future_points']==12
                cross.append(dict(model=n,recording=k[0],sample_id=k[1],target_id=k[2],A_ADE_m=r['ADE_m'],B_ADE_m=b[k]['ADE_m'],
                    delta_A_minus_B_ADE_m=r['ADE_m']-b[k]['ADE_m'],A_FDE_m=r['FDE_m'],B_FDE_m=b[k]['FDE_m'],delta_A_minus_B_FDE_m=r['FDE_m']-b[k]['FDE_m'],
                    interpretation='same complete targets; separately trained models and different history-context policy, not single-factor protocol efficacy'))
        csv_new(ROOT/'cross_protocol_common_targets.csv',cross)
    else:raise ValueError(mode)
    csv_new(ROOT/(mode+'_group_comparison.csv'),all_groups);csv_new(ROOT/(mode+'_paired_target_differences.csv'),all_pairs)
    write_json_new(ROOT/(mode+'_pair_provenance.json'),dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        mode=mode,paired_rows=len(all_pairs),group_rows=len(all_groups),grouping_future_GT=False,GT_used_for_errors_only=True,ETH_evaluated=False))


if __name__=='__main__':main()
