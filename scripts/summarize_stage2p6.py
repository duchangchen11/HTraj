"""Summarize every registered run/seed; no seed filtering or endpoint changes."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.order_verification import ROOT,PROTOCOL
from src.training.social_diagnostics import csv_new


def load_runs():
    protocol=json.loads(PROTOCOL.read_text())
    runs=list(csv.DictReader((ROOT/'frozen_baselines.csv').open()))
    runs += [json.loads(p.read_text()) for p in sorted((ROOT/'new').glob('*/*/summary.json'))]
    expected={(m,s) for m in protocol['frozen_models']+protocol['new_models'] for s in protocol['seeds']}
    assert {(r['model'],int(r['seed'])) for r in runs} == expected and len(runs)==len(expected)
    assert all(r['status']=='completed' and int(r['last_epoch'])==30 for r in runs)
    return runs


def stats(values):
    if any(v in (None,'') for v in values):
        assert all(v in (None,'') for v in values)
        return None,None
    array=np.asarray(values,dtype=float)
    return float(array.mean()),float(array.std(ddof=1))


def main():
    runs=load_runs(); lookup={(r['model'],int(r['seed'])):r for r in runs}
    csv_new(ROOT/'all_runs.csv',runs)
    models=sorted({r['model'] for r in runs})
    summary=[]
    for model in models:
        rows=[lookup[model,s] for s in [17,42,2026]]
        record=dict(model=model,seeds='17;42;2026',n=3,parameter_count=rows[0]['parameter_count'],
            trained_in_stage2p6=rows[0]['trained_in_stage2p6'],best_epochs=';'.join(str(r['best_epoch']) for r in rows),
            source_commit=rows[0]['source_commit'],data_manifest_sha256=rows[0]['data_manifest_sha256'],
            subset_manifest_sha256=rows[0]['subset_manifest_sha256'],test_evaluated=False)
        # A one-channel softmax is exactly constant. Report this obvious inert
        # scorer separately; do not pretend the subtraction measures capacity.
        inactive=8321 if model in ['TOHI-H1','TOHI-H1-Cap','Dynamic-GNN-RN'] else 0
        record['known_inactive_single_channel_scorer_parameters']=inactive
        record['nominal_minus_known_inactive_parameters']=int(rows[0]['parameter_count'])-inactive
        record['capacity_caveat']='not identifiable effective capacity; parameter allocation differs'
        for metric in ['best_ADE_m','best_FDE_m','last_ADE_m','last_FDE_m','actual_training_seconds']:
            record[metric+'_mean'],record[metric+'_std']=stats([r[metric] for r in rows])
        summary.append(record)
    csv_new(ROOT/'three_seed_summary.csv',summary)
    pairs=[('TOHI-H2','TOHI-H1'),('TOHI-H3','TOHI-H2'),('TOHI-H3','TOHI-NoOrder'),
           ('TOHI-H2','Dynamic-GNN-RN'),('TOHI-H3','Dynamic-GNN-RN'),('TOHI-NoOrder','Dynamic-GNN-RN'),
           ('TOHI-H1','PedBase'),('TOHI-H2','PedBase'),('TOHI-H3','PedBase'),('TOHI-NoOrder','PedBase'),
           ('TOHI-H1-Cap','TOHI-H1'),('TOHI-H2-Cap','TOHI-H2'),
           ('TOHI-H2','TOHI-H1-Cap'),('TOHI-H3','TOHI-H2-Cap')]
    pair_rows=[]; pair_summary=[]
    for a,b in pairs:
        record=dict(model=a,reference=b,n=3,interpretation='model minus reference; negative better; paired seeds, development only')
        actual=[]
        for seed in [17,42,2026]:
            ra,rb=lookup[a,seed],lookup[b,seed]
            row=dict(model=a,reference=b,seed=seed,model_best_epoch=ra['best_epoch'],reference_best_epoch=rb['best_epoch'])
            for metric in ['best_ADE_m','best_FDE_m','last_ADE_m','last_FDE_m']:
                row['delta_'+metric]=float(ra[metric])-float(rb[metric])
            row.update(model_source_commit=ra['source_commit'],reference_source_commit=rb['source_commit'],
                       data_manifest_sha256=ra['data_manifest_sha256'],subset_manifest_sha256=ra['subset_manifest_sha256'],test_evaluated=False)
            actual.append(row); pair_rows.append(row)
        for metric in ['best_ADE_m','best_FDE_m','last_ADE_m','last_FDE_m']:
            record['delta_'+metric+'_mean'],record['delta_'+metric+'_std']=stats([r['delta_'+metric] for r in actual])
            record[metric+'_improving_seed_count']=sum(r['delta_'+metric]<0 for r in actual)
        record['both_best_metrics_improving_seed_count']=sum(r['delta_best_ADE_m']<0 and r['delta_best_FDE_m']<0 for r in actual)
        pair_summary.append(record)
    csv_new(ROOT/'paired_differences.csv',pair_rows); csv_new(ROOT/'paired_difference_summary.csv',pair_summary)
    grouped=[]; diagnostics=[]
    def group_path(model,seed,endpoint):
        return ROOT/('frozen' if model in ['PedBase','Dynamic-GNN-RN'] else 'new')/model/f'seed_{seed}'/f'groups_{endpoint}.csv'
    for run in runs:
        name,seed=run['model'],int(run['seed'])
        if name not in ['PedBase','Dynamic-GNN-RN']:
            epochs=list(csv.DictReader((ROOT/'new'/name/f'seed_{seed}/epochs.csv').open()))
            for endpoint in ['best','last']:
                selected_epoch=int(run[endpoint+'_epoch'])
                diagnostics.append(dict(model=name,checkpoint=endpoint,**epochs[selected_epoch-1],
                    data_manifest_sha256=run['data_manifest_sha256'],subset_manifest_sha256=run['subset_manifest_sha256']))
        for endpoint in ['best','last']:
            baseline={r['group']:r for r in csv.DictReader(group_path('PedBase',seed,endpoint).open())}
            dynamic={r['group']:r for r in csv.DictReader(group_path('Dynamic-GNN-RN',seed,endpoint).open())}
            for row in csv.DictReader(group_path(name,seed,endpoint).open()):
                ped,dyn=baseline[row['group']],dynamic[row['group']]
                assert row['ade_count']==ped['ade_count']==dyn['ade_count']
                assert row['fde_count']==ped['fde_count']==dyn['fde_count']
                extra={}
                for metric in ['ADE_m','FDE_m']:
                    for ref,other in [('PedBase',ped),('Dynamic-GNN-RN',dyn)]:
                        extra['delta_'+metric+'_vs_'+ref]=float(row[metric])-float(other[metric]) if row[metric] else None
                grouped.append(dict(model=name,seed=seed,checkpoint=endpoint,selected_epoch=run[endpoint+'_epoch'],**row,
                    **extra,source_commit=run['source_commit'],data_manifest_sha256=run['data_manifest_sha256'],
                    subset_manifest_sha256=run['subset_manifest_sha256'],test_evaluated=False))
    csv_new(ROOT/'group_comparison.csv',grouped); csv_new(ROOT/'diagnostics_best_last.csv',diagnostics)
    group_summary=[]
    for model in models:
        for endpoint in ['best','last']:
            for group in sorted({r['group'] for r in grouped}):
                rows=[r for r in grouped if r['model']==model and r['checkpoint']==endpoint and r['group']==group]
                assert len(rows)==3
                record=dict(model=model,checkpoint=endpoint,group=group,n=3,
                    ade_count_per_seed=rows[0]['ade_count'],fde_count_per_seed=rows[0]['fde_count'])
                for metric in ['ADE_m','FDE_m','delta_ADE_m_vs_PedBase','delta_FDE_m_vs_PedBase',
                               'delta_ADE_m_vs_Dynamic-GNN-RN','delta_FDE_m_vs_Dynamic-GNN-RN']:
                    record[metric+'_mean'],record[metric+'_std']=stats([r[metric] for r in rows])
                group_summary.append(record)
    csv_new(ROOT/'three_seed_group_summary.csv',group_summary)
    relations=[]
    for model in ['TOHI-H2','TOHI-H3','TOHI-NoOrder','Dynamic-GNN-RN']:
        rows=list(csv.DictReader((ROOT/'relations'/f'{model}.csv').open()))
        assert len(rows)==24
        relations+=rows
    csv_new(ROOT/'relation_interventions.csv',relations)
    print('Complete:',len(runs),'registered runs;',len(grouped),'group rows;',len(relations),'inference control rows')


if __name__=='__main__':
    main()
