"""Post-training paired analysis and predeclared frozen-weight checks only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
import subprocess
import math
import torch
from src.data.preprocessing import sha256, write_json_new
from src.training.frozen_ego import data_load
from src.training.motion_generalization import ROOT, CONF, attach_groups, evaluate, save_eval, model_for
from src.training.social_diagnostics import csv_new
from src.training.smoke import move_batch
from src.models.frozen_ego import CVResidual
from src.models.relation_routing import HistoricalCV
from src.models.motion_normalization import heading_frame
from src.evaluation.benchmark_audit import target_errors


def read_csv(path): return list(csv.DictReader(Path(path).open()))
def read_json(path): return json.loads(Path(path).read_text())
def key(row): return row['sample_id'], int(row['target_id'])
def lookup(rows): return {key(r): r for r in rows}


def check_reference(p, fold):
    path = Path('results/tables/stage3p6')/fold['recording']/'ego/CV-Residual__constant/seed_17'
    c, s = read_json(path/'config.json'), read_json(path/'summary.json')
    assert c['manifest_sha256'] == fold['manifest_sha256'] and c['protocol_name'] == p['protocol_name']
    old = c['training']; assert old['source_files'] == p['source_files']
    for k in ('epochs', 'batch_size', 'hidden_dim', 'prediction_steps', 'learning_rate', 'gradient_clip'):
        assert old[k] == p[k]
    oldfold = next(f for f in old['folds'] if f['recording'] == fold['recording'])
    assert oldfold == fold and c['seed'] == 17 and c['schedule'] == 'constant'
    for f, digest in c['source_sha256'].items(): assert sha256(f) == digest, f
    for endpoint in ('best', 'last'): assert sha256(s[endpoint+'_checkpoint_path']) == s[endpoint+'_checkpoint_sha256']
    return path, s


def load_neural(name, summary, endpoint, device):
    model = CVResidual() if name == 'CVR-Unnormalized-reference' else model_for(name)
    checkpoint = summary[endpoint+'_checkpoint_path']; assert sha256(checkpoint) == summary[endpoint+'_checkpoint_sha256']
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    return model.to(device).eval()


def rotation(x, degrees):
    a = math.radians(degrees); c, s = math.cos(a), math.sin(a)
    return torch.stack([c*x[..., 0]-s*x[..., 1], s*x[..., 0]+c*x[..., 1]], -1)


@torch.no_grad()
def geometric_checks(model, name, fold, val, p, device):
    rows = []; model.eval()
    for angle in p['supplemental']['rotations_degrees']:
        totals = {g: [0., 0., 0., 0., 0] for g in ('all', 'heading_reliable', 'heading_fallback')}
        for original in val:
            b = move_batch(original, device)
            y = model(b['observed'], b['observed_valid'], b['timestamps'])
            yr = model(rotation(b['observed'], angle), b['observed_valid'], b['timestamps'])
            equiv = (yr-rotation(y, angle)).norm(dim=-1)
            ade, fde, a, f, _ = target_errors(yr, rotation(b['future'], angle), b['future_valid'])
            for group in totals:
                mask = a & b['groups'][group].to(device)
                t = totals[group]; t[0] += float(ade[mask].sum()); t[1] += float(fde[mask].sum())
                t[2] += float(equiv.mean(1)[mask].sum()); t[3] = max(t[3], float(equiv.transpose(1, 2)[mask].max()) if mask.any() else 0.)
                t[4] += int(mask.sum())
        for group, t in totals.items():
            rows.append(dict(recording=fold['recording'], model=name, endpoint='best', rotation_degrees=angle, group=group,
                targets=t[4], rotated_ADE_m=t[0]/t[4] if t[4] else None, rotated_FDE_m=t[1]/t[4] if t[4] else None,
                equivariance_mean_error_m=t[2]/t[4] if t[4] else None, equivariance_max_error_m=t[3] if t[4] else None,
                intervention='frozen checkpoint inference only; rotate observations and metric GT together; never select checkpoint'))
    return rows


def main():
    assert len(list(ROOT.glob('*/M*/seed_17/summary.json'))) == 6, 'All six fixed runs must complete first'
    torch.set_num_threads(1); device = torch.device('cuda'); p = read_json(CONF)
    summaries, groups, pairs, curves, rotation_rows, sensitivity, refs, reproducibility, stability, fixed_epochs = [], [], [], [], [], [], [], [], [], []
    for fold in p['folds']:
        train, val = data_load(p, fold); val = [attach_groups(b) for b in val]
        ref_path, ref = check_reference(p, fold)
        refs.append(dict(recording=fold['recording'], original_summary=str(ref_path/'summary.json'), summary_sha256=sha256(ref_path/'summary.json'),
            source_commit=ref['source_commit'], compatibility_checked=True, newly_trained=False,
            best_checkpoint_sha256=ref['best_checkpoint_sha256'], last_checkpoint_sha256=ref['last_checkpoint_sha256']))
        refout = ROOT/fold['recording']/'CVR-Unnormalized-reference'; refout.mkdir(exist_ok=False)
        for endpoint in ('best', 'last'):
            model = load_neural('CVR-Unnormalized-reference', ref, endpoint, device)
            r = evaluate(model, val, device, rows=True); save_eval(refout, endpoint, 'validation', r)
            assert abs(r['ADE_m']-ref[endpoint+'_ADE_m']) < 1e-6
        write_json_new(refout/'summary.json', dict(ref, model='CVR-Unnormalized-reference', newly_trained=False, reference_path=str(ref_path)))
        # Keep reference training endpoint errors from the already measured
        # strict-compatible Stage3.6 endpoints; do not retrain or overwrite.
        ref_endpoints = [r for r in read_csv('results/tables/stage3p6/endpoint_train_validation_generalization.csv')
                         if r['recording'] == fold['recording'] and r['model'] == 'CV-Residual' and r['schedule'] == 'constant']
        records = {}
        for name in ['CV', 'M0', 'M2', 'M3', 'CVR-Unnormalized-reference']:
            directory = ROOT/fold['recording']/name if name in ('CV', 'CVR-Unnormalized-reference') else ROOT/fold['recording']/name/'seed_17'
            s = read_json(directory/'summary.json')
            summary = dict(recording=fold['recording'], model=name, seed=17 if name != 'CV' else None,
                new_training=name in p['neural_models'], parameter_count=s['parameter_count'], training_seconds=s.get('training_seconds'),
                best_epoch=s.get('best_epoch'), last_epoch=s.get('last_epoch'), source_commit=s['source_commit'],
                best_checkpoint_sha256=s.get('best_checkpoint_sha256'), last_checkpoint_sha256=s.get('last_checkpoint_sha256'))
            for endpoint in ('best', 'last'):
                label = 'analytical' if name == 'CV' else endpoint
                targets = lookup(read_csv(directory/f'targets_{label}_validation.csv')); records[name, endpoint] = targets
                for split in ('train', 'validation'):
                    if name == 'CV':
                        summary[endpoint+'_'+split+'_ADE_m'] = s[split+'_ADE_m']; summary[endpoint+'_'+split+'_FDE_m'] = s[split+'_FDE_m']
                    elif name == 'CVR-Unnormalized-reference':
                        r = next(r for r in ref_endpoints if r['endpoint'] == endpoint and r['split'] == split)
                        summary[endpoint+'_'+split+'_ADE_m'] = float(r['ADE_m']); summary[endpoint+'_'+split+'_FDE_m'] = float(r['FDE_m'])
                    else:
                        for metric in ('ADE_m', 'FDE_m'): summary[endpoint+'_'+split+'_'+metric] = s[endpoint+'_'+split+'_'+metric]
                for r in read_csv(directory/f'groups_{label}_validation.csv'):
                    groups.append(dict(recording=fold['recording'], model=name, endpoint=endpoint, **r))
            summaries.append(summary)
            if name in p['neural_models']:
                c = read_csv(directory/'epochs.csv'); curves.extend(dict(recording=fold['recording'], model=name, seed=17, **r) for r in c)
                stability.append(dict(recording=fold['recording'], model=name, best_epoch=s['best_epoch'], best_in_first5=s['best_epoch'] <= 5,
                    first_online_train_loss_m=float(c[0]['train_online_ADE_loss_m']), last_online_train_loss_m=float(c[-1]['train_online_ADE_loss_m']),
                    last_minus_best_validation_ADE_m=s['last_ADE_m']-s['best_ADE_m'],
                    best_endpoint_generalization_gap_m=s['best_validation_ADE_m']-s['best_train_ADE_m'],
                    last_endpoint_generalization_gap_m=s['last_validation_ADE_m']-s['last_train_ADE_m']))
            model = HistoricalCV().to(device) if name == 'CV' else load_neural(name, s, 'best', device)
            rotation_rows.extend(geometric_checks(model, name, fold, val, p, device))
            if name in ('M2', 'M3'):
                for threshold in p['supplemental']['threshold_sensitivity_mps']:
                    model.threshold = threshold
                    r = evaluate(model, val, device)
                    changed, targets = 0, 0
                    for b in val:
                        f = heading_frame(b['observed'], b['observed_valid'], b['timestamps'], threshold)
                        target = b['future_valid'].all(1)
                        changed += int(((f['reliable'] != b['groups']['heading_reliable']) & target).sum())
                        targets += int(target.sum())
                    sensitivity.extend(dict(recording=fold['recording'], model=name, endpoint='best', inference_threshold_mps=threshold,
                        training_threshold_mps=.2, reliability_switch_targets=changed, evaluated_targets=targets,
                        used_for_selection=False, retrained=False, group_definition='fixed history strata at original .2m/s', **g) for g in r['groups'])
        # A fresh M0 run checks the frozen reference protocol's reproducibility.
        old_dir = Path('results/tables/stage3p6')/fold['recording']/'ego/PedBase__constant/seed_17'
        old_curve = read_csv(old_dir/'epochs.csv'); new_curve = read_csv(ROOT/fold['recording']/'M0/seed_17/epochs.csv')
        diffs = [abs(float(a['validation_ADE_m'])-float(b['validation_ADE_m'])) for a, b in zip(old_curve, new_curve)]
        reproducibility.append(dict(recording=fold['recording'], old_summary_sha256=sha256(old_dir/'summary.json'),
            epochs=30, maximum_validation_ADE_difference_m=max(diffs), compatible_within_1e6=max(diffs) < 1e-6))
        unnormalized_curve = read_csv(ref_path/'epochs.csv')
        for name in ('M2', 'M3'):
            candidate_curve = read_csv(ROOT/fold['recording']/name/'seed_17/epochs.csv')
            for candidate, baseline, cvres in zip(candidate_curve, new_curve, unnormalized_curve):
                fixed_epochs.append(dict(recording=fold['recording'], model=name, epoch=int(candidate['epoch']),
                    online_train_ADE_loss_m=float(candidate['train_online_ADE_loss_m']),
                    delta_online_train_loss_vs_M0_m=float(candidate['train_online_ADE_loss_m'])-float(baseline['train_online_ADE_loss_m']),
                    validation_ADE_m=float(candidate['validation_ADE_m']), validation_FDE_m=float(candidate['validation_FDE_m']),
                    delta_validation_ADE_vs_M0_m=float(candidate['validation_ADE_m'])-float(baseline['validation_ADE_m']),
                    delta_validation_FDE_vs_M0_m=float(candidate['validation_FDE_m'])-float(baseline['validation_FDE_m']),
                    delta_validation_ADE_vs_unnormalized_CVR_m=float(candidate['validation_ADE_m'])-float(cvres['validation_ADE_m']),
                    interpretation='same optimizer-update budget; online loss is not fixed-endpoint train inference'))
        for endpoint in ('best', 'last'):
            base, cv, unnorm = records['M0', endpoint], records['CV', endpoint], records['CVR-Unnormalized-reference', endpoint]
            for name in ['CV', 'M0', 'M2', 'M3', 'CVR-Unnormalized-reference']:
                assert set(records[name, endpoint]) == set(base) == set(cv)
                for k, r in records[name, endpoint].items():
                    pairs.append(dict(recording=fold['recording'], sample_id=k[0], target_id=k[1], model=name, endpoint=endpoint,
                        ADE_m=float(r['ADE_m']), FDE_m=float(r['FDE_m']),
                        M0_ADE_m=float(base[k]['ADE_m']), M0_FDE_m=float(base[k]['FDE_m']), CV_ADE_m=float(cv[k]['ADE_m']), CV_FDE_m=float(cv[k]['FDE_m']),
                        delta_ADE_vs_M0_m=float(r['ADE_m'])-float(base[k]['ADE_m']), delta_FDE_vs_M0_m=float(r['FDE_m'])-float(base[k]['FDE_m']),
                        delta_ADE_vs_CV_m=float(r['ADE_m'])-float(cv[k]['ADE_m']), delta_FDE_vs_CV_m=float(r['FDE_m'])-float(cv[k]['FDE_m']),
                        delta_ADE_vs_unnormalized_CVR_m=float(r['ADE_m'])-float(unnorm[k]['ADE_m']),
                        heading_speed_mps=float(r['heading_speed_mps']), heading_reliable=r['heading_reliable'],
                        history_speed_low=r['history_speed_low'], history_speed_medium=r['history_speed_medium'], history_speed_high=r['history_speed_high'],
                        history_CV_like=r['history_CV_like'], history_velocity_changing=r['history_velocity_changing']))
    # Explicit equivalent M1 rows have no fictitious training time or run.
    summaries.extend(dict(r, model='M1', new_training=False, training_seconds=None, equivalent_source='M0 exact alias; no extra fit') for r in list(summaries) if r['model'] == 'M0')
    group_map = {(r['recording'], r['model'], r['endpoint'], r['group']): r for r in groups}
    for r in groups:
        base = group_map[r['recording'], 'M0', r['endpoint'], r['group']]; cv = group_map[r['recording'], 'CV', r['endpoint'], r['group']]
        assert r['ADE_count'] == base['ADE_count'] == cv['ADE_count']
        for metric in ('ADE_m', 'FDE_m'):
            r['delta_'+metric+'_vs_M0'] = float(r[metric])-float(base[metric]) if r[metric] else None
            r['delta_'+metric+'_vs_CV'] = float(r[metric])-float(cv[metric]) if r[metric] else None
    decision_rows = []; eligible = []
    for name in ('M2', 'M3'):
        passed = []
        for fold in p['folds']:
            base = next(r for r in summaries if r['recording'] == fold['recording'] and r['model'] == 'M0')
            r = next(r for r in summaries if r['recording'] == fold['recording'] and r['model'] == name)
            ade, fde = r['best_validation_ADE_m'], r['best_validation_FDE_m']
            pass_ = ade <= base['best_validation_ADE_m']-max(.005, .005*base['best_validation_ADE_m']) and fde <= base['best_validation_FDE_m']
            passed.append(pass_); decision_rows.append(dict(recording=fold['recording'], model=name, pass_rule=pass_,
                delta_best_ADE_vs_M0_m=ade-base['best_validation_ADE_m'], delta_best_FDE_vs_M0_m=fde-base['best_validation_FDE_m']))
        if all(passed): eligible.append(name)
    for filename, rows in [('model_comparison.csv', summaries), ('all_epoch_curves.csv', curves), ('group_comparison.csv', groups),
        ('paired_target_errors.csv', pairs), ('rotation_checks.csv', rotation_rows), ('threshold_sensitivity.csv', sensitivity),
        ('training_stability.csv', stability), ('M0_reproducibility.csv', reproducibility), ('fixed_epoch_comparison.csv', fixed_epochs)]: csv_new(ROOT/filename, rows)
    write_json_new(ROOT/'reference_compatibility.json', dict(references=refs, code_data_evaluator_compatible=True, new_training=False))
    write_json_new(ROOT/'research_decision.json', dict(rule=p['candidate_rule'], rows=decision_rows, eligible_for_multiseed_proposal=eligible,
        extra_seeds_run=False, further_training_started=False, ETH_evaluated=False, independent_confirmation=False))
    inputs = [CONF]+list(ROOT.glob('*/M*/seed_17/summary.json'))+list(ROOT.glob('*/CV/summary.json'))
    write_json_new(ROOT/'analysis_provenance.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha256(__file__), inputs=[dict(path=str(f), sha256=sha256(f)) for f in inputs], neural_runs=6, executed_epochs=180,
        actual_seeds=[17], pair_rows=len(pairs), group_rows=len(groups), training_seconds=sum(float(r['training_seconds']) for r in summaries if r['new_training']),
        optimizer_updates=sum(read_json(f)['optimizer_steps'] for f in ROOT.glob('*/M*/seed_17/summary.json')),
        ETH_evaluated=False, supplemental_used_for_selection=False, extra_training=False))
    print(json.dumps(dict(status='analyzed', rows=len(pairs), eligible_for_proposal=eligible)))


if __name__ == '__main__': main()
