"""Synthesize only completed runs; same targets, all groups, posthoc CV errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
import subprocess
import numpy as np
from src.data.preprocessing import sha256, write_json_new
from src.training.social_diagnostics import csv_new
from src.evaluation.frozen_ego import GROUPS

ROOT = Path('results/tables/stage3p6')


def load_targets(path):
    rows = list(csv.DictReader(path.open()))
    result = {}
    for r in rows:
        key = (r['recording'], r['sample_id'], r['target_id'])
        assert key not in result and int(r['future_points']) == 12
        result[key] = r
    return result


def main():
    protocol = json.loads(Path('configs/stage3p6/protocol.json').read_text())
    decision = json.loads((ROOT/'round3_decision.json').read_text())
    summaries, epoch_rows, diagnostics = [], [], []
    for path in sorted(ROOT.glob('*/**/summary.json')):
        s = json.loads(path.read_text())
        summaries.append(dict(summary_path=str(path), **{k: v for k, v in s.items() if not isinstance(v, (dict, list))}))
        if path.with_name('epochs.csv').exists():
            for r in csv.DictReader(path.with_name('epochs.csv').open()):
                epoch_rows.append(dict(recording=s['recording'], model=s['model'], phase=s['phase'], seed=s['seed'], schedule=s['schedule'], **r))
        for d in path.parent.glob('diagnostics_*.json'):
            diagnostics.append(dict(recording=s['recording'], model=s['model'], seed=s.get('seed'), schedule=s.get('schedule'), endpoint=d.stem[12:], **json.loads(d.read_text())))
    csv_new(ROOT/'completed_run_summary.csv', summaries)
    csv_new(ROOT/'all_epoch_curves.csv', epoch_rows)
    csv_new(ROOT/'all_state_diagnostics.csv', diagnostics)
    groups, pairs, no_message_checks = [], [], []
    seeds = [17] + (protocol['conditional_seeds'] if decision['run_extra_seeds'] else [])
    for fold in protocol['folds']:
        rec = fold['recording']
        cv = load_targets(ROOT/rec/'ego/CV__analytical/seed_17/targets_analytical.csv')
        for seed in seeds:
            for endpoint in ('best', 'last'):
                models = {'CV': cv}
                for label in protocol['neural_ego_runs']:
                    name, schedule = label.split('/')
                    file = ROOT/rec/'ego'/(name+'__'+schedule)/f'seed_{seed}'/f'targets_{endpoint}.csv'
                    if file.exists(): models[label] = load_targets(file)
                for name in protocol['residual_models']:
                    file = ROOT/rec/'residual'/name/f'seed_{seed}'/('targets_analytical.csv' if name == 'Ego-Only' else f'targets_{endpoint}.csv')
                    models[name] = load_targets(file)
                assert all(set(m) == set(cv) for m in models.values())
                for key in cv:
                    r = dict(recording=rec, seed=seed, endpoint=endpoint, sample_id=key[1], target_id=key[2], future_points=12,
                        history_frames=float(cv[key]['history_frames']), mean_degree=float(cv[key]['mean_neighbor_degree']), posthoc_cv_ADE_ge1m=float(cv[key]['ADE_m']) >= 1.)
                    for name, model in models.items():
                        r[name+'_ADE_m'] = float(model[key]['ADE_m']); r[name+'_FDE_m'] = float(model[key]['FDE_m'])
                    for name in protocol['residual_models'][1:]:
                        for reference in ('Ego-Only', 'Ego-Residual'):
                            for metric in ('ADE', 'FDE'):
                                r[f'{name}_minus_{reference}_{metric}_m'] = r[f'{name}_{metric}_m']-r[f'{reference}_{metric}_m']
                    pairs.append(r)
                masks = {g: [k for k, r in cv.items() if r[g] == 'True'] for g in GROUPS}
                masks['posthoc_CV_ADE_ge1m'] = [k for k, r in cv.items() if float(r['ADE_m']) >= 1.]
                masks['posthoc_CV_ADE_lt1m'] = [k for k, r in cv.items() if float(r['ADE_m']) < 1.]
                for group, keys in masks.items():
                    for name, model in models.items():
                        row = dict(recording=rec, seed=seed, endpoint=endpoint, group=group, model=name, target_count=len(keys), unique_people=len({k[2] for k in keys}), windows=len({k[1] for k in keys}),
                            interpretation='posthoc outcome stratum, never input or selection' if group.startswith('posthoc') else 'history-only group')
                        if keys:
                            for metric in ('ADE', 'FDE'):
                                values = np.array([float(model[k][metric+'_m']) for k in keys])
                                row[metric+'_m'] = float(values.mean())
                                for ref in ('CV', 'Ego-Only', 'Ego-Residual'):
                                    delta = values-np.array([float(models[ref][k][metric+'_m']) for k in keys])
                                    row[f'delta_minus_{ref}_{metric}_m'] = float(delta.mean())
                                    row[f'paired_improved_vs_{ref}_{metric}_fraction'] = float((delta < 0).mean())
                                    row[f'paired_delta_vs_{ref}_{metric}_median_m'] = float(np.median(delta))
                            for field in ('history_frames', 'continuous_motion_frames', 'mean_neighbor_degree', 'history_velocity_change_mps', 'relative_velocity_change_mps'):
                                row[field+'_mean'] = float(np.mean([float(cv[k][field]) for k in keys]))
                        groups.append(row)
                empty = masks['no_legal_social_messages']
                for name in ('Social-Previous', 'Social-Current', 'Social-Random'):
                    for key in empty:
                        r, anchor = models[name][key], models['Ego-Only'][key]
                        assert r['ADE_m'] == anchor['ADE_m'] and r['FDE_m'] == anchor['FDE_m']
                        assert float(r['gate']) == float(r['correction_norm']) == 0
                    no_message_checks.append(dict(recording=rec, seed=seed, endpoint=endpoint, model=name, targets=len(empty), exact_per_target_error_and_zero_correction=True))
    csv_new(ROOT/'group_comparison.csv', groups); csv_new(ROOT/'paired_target_errors.csv', pairs)
    csv_new(ROOT/'no_message_exact_checks.csv', no_message_checks)
    write_json_new(ROOT/'analysis_provenance.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), script_sha256=sha256(__file__),
        predeclared_protocol_sha256=sha256('configs/stage3p6/protocol.json'), actual_seeds=seeds, completed_neural_runs=sum('best_epoch' in r for r in summaries), executed_epochs=len(epoch_rows),
        group_rows=len(groups), paired_target_rows=len(pairs), grouping_future_input=False, outcome_group_posthoc_only=True, ETH_evaluated=False, inputs=[dict(path=str(p), sha256=sha256(p)) for p in sorted(ROOT.glob('*/**/targets_*.csv'))]))
    print(json.dumps(dict(completed_neural_runs=sum('best_epoch' in r for r in summaries), executed_epochs=len(epoch_rows), groups=len(groups), paired_rows=len(pairs), seeds=seeds)))


if __name__ == '__main__':
    main()
