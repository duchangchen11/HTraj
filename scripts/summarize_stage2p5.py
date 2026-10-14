"""Pair group metrics with PedBase; aggregate all seeds without dropping runs."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.social_diagnostics import ROOT, csv_new


def main():
    grouped = []
    diagnostic_rows = []
    for phase in ('round1', 'round2'):
        summary_path = ROOT/phase/'summary.csv'
        if not summary_path.exists():
            continue
        runs = list(csv.DictReader(summary_path.open()))
        for run in runs:
            model, seed = run['model'], int(run['seed'])
            epochs = list(csv.DictReader((ROOT/phase/model/f'seed_{seed}'/'epochs.csv').open()))
            for checkpoint in ('best', 'last'):
                selected_epoch = int(run['best_epoch'] if checkpoint == 'best' else run['last_epoch'])
                diagnostic_rows.append(dict(phase=phase, model=model, checkpoint=checkpoint,
                    **epochs[selected_epoch-1], data_manifest_sha256=run['data_manifest_sha256'],
                    subset_manifest_sha256=run['subset_manifest_sha256'], protocol_sha256=run['protocol_sha256']))
                path = ROOT/phase/model/f'seed_{seed}'/f'groups_{checkpoint}.csv'
                baseline_path = ROOT/phase/'PedBase'/f'seed_{seed}'/f'groups_{checkpoint}.csv'
                actual = list(csv.DictReader(path.open()))
                baseline = {row['group']: row for row in csv.DictReader(baseline_path.open())}
                for row in actual:
                    ped = baseline[row['group']]
                    assert row['ade_count'] == ped['ade_count'] and row['fde_count'] == ped['fde_count']
                    grouped.append(dict(phase=phase, model=model, seed=seed, checkpoint=checkpoint,
                        selected_epoch=run['best_epoch'] if checkpoint == 'best' else run['last_epoch'],
                        **row, delta_ADE_vs_PedBase=float(row['ADE_m'])-float(ped['ADE_m']) if row['ADE_m'] else None,
                        delta_FDE_vs_PedBase=float(row['FDE_m'])-float(ped['FDE_m']) if row['FDE_m'] else None,
                        source_commit=run['source_commit'], subset_manifest_sha256=run['subset_manifest_sha256']))
    csv_new(ROOT/'group_comparison.csv', grouped)
    csv_new(ROOT/'diagnostics_best_last.csv', diagnostic_rows)
    summary_path = ROOT/'round2/summary.csv'
    if summary_path.exists():
        runs = list(csv.DictReader(summary_path.open()))
        models = sorted({run['model'] for run in runs})
        aggregates = []
        for model in models:
            rows = [run for run in runs if run['model'] == model]
            assert {int(row['seed']) for row in rows} == {17, 42, 2026}
            result = dict(model=model, seeds='17;42;2026', n=3, phase='round2', training_windows=1024,
                          validation_windows=96, epochs=30, optimizer_steps_per_seed=3840)
            baseline = {int(run['seed']): run for run in runs if run['model'] == 'PedBase'}
            for field in ('best_ADE_m','best_FDE_m','last_ADE_m','last_FDE_m'):
                values = np.asarray([float(row[field]) for row in rows])
                result[field+'_mean'] = float(values.mean())
                result[field+'_std'] = float(values.std(ddof=1))
                differences = np.asarray([float(row[field])-float(baseline[int(row['seed'])][field]) for row in rows])
                result[field+'_delta_vs_PedBase_mean'] = float(differences.mean())
                result[field+'_delta_vs_PedBase_std'] = float(differences.std(ddof=1))
            result['best_epochs'] = ';'.join(str(row['best_epoch']) for row in rows)
            for field in ('source_commit', 'data_manifest_sha256', 'subset_manifest_sha256', 'protocol_sha256'):
                assert len({row[field] for row in rows}) == 1
                result[field] = rows[0][field]
            result['test_evaluated'] = False
            aggregates.append(result)
        csv_new(ROOT/'round2_mean_std.csv', aggregates)
        group_aggregates = []
        for model in models:
            for checkpoint in ('best', 'last'):
                for group in sorted({r['group'] for r in grouped}):
                    rows = [r for r in grouped if r['phase']=='round2' and r['model']==model
                            and r['checkpoint']==checkpoint and r['group']==group]
                    assert {r['seed'] for r in rows} == {17,42,2026}
                    result = dict(model=model, checkpoint=checkpoint, group=group, n=3,
                                  ade_count_per_seed=rows[0]['ade_count'], fde_count_per_seed=rows[0]['fde_count'])
                    for field in ('ADE_m','FDE_m','delta_ADE_vs_PedBase','delta_FDE_vs_PedBase'):
                        values = np.asarray([float(r[field]) for r in rows])
                        result[field+'_mean'] = float(values.mean())
                        result[field+'_std'] = float(values.std(ddof=1))
                    group_aggregates.append(result)
        csv_new(ROOT/'round2_group_mean_std.csv', group_aggregates)
    print('Paired group comparisons written:', len(grouped))


if __name__ == '__main__':
    main()
