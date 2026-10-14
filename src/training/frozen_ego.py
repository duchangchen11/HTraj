"""Predeclared Stage3.6 ego, frozen-information controls, conditional seeds."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .smoke import seed_all, move_batch, synchronize
from .social_diagnostics import csv_new
from ..data.benchmark_audit import BenchmarkDataset, assert_recording_isolation
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256, write_json_new
from ..models.ped_base import PedBase
from ..models.relation_routing import HistoricalCV
from ..models.frozen_ego import CVResidual, FrozenEgoResidual
from ..evaluation.frozen_ego import history_groups, evaluate
from ..evaluation.metrics import masked_l2_loss

ROOT = Path('results/tables/stage3p6')
CONF = Path('configs/stage3p6/protocol.json')
LOCAL = Path('experiments/stage3p6')
EGO = {'PedBase': PedBase, 'CV-Residual': CVResidual}


def state_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def output_path(fold, phase, name, seed, schedule='constant'):
    label = name+'__'+schedule if phase == 'ego' else name
    suffix = Path(fold['recording'])/phase/label/f'seed_{seed}'
    return ROOT/suffix, LOCAL/suffix/'checkpoints'


def data_load(p, fold):
    m = json.loads(Path(fold['manifest']).read_text())
    assert sha256(fold['manifest']) == fold['manifest_sha256']
    assert_recording_isolation(m)
    datasets = {split: Subset(BenchmarkDataset(p['raw_root'], fold['manifest'], split, p['protocol_name']), fold['indices'][split]) for split in ('train', 'validation')}
    val = []
    for b in DataLoader(datasets['validation'], batch_size=p['batch_size'], collate_fn=collate_trajectories):
        b['groups'], b['history_features'] = history_groups(b); val.append(b)
    return datasets['train'], val


def collate_cached(samples):
    b = collate_trajectories(samples)
    B, T, N, _ = b['observed'].shape
    d = samples[0]['anchor_hidden'].shape[-1]
    b['anchor_hidden'] = torch.zeros(B, T, N, d)
    b['anchor_prediction'] = torch.zeros(B, 12, N, 2)
    for i, s in enumerate(samples):
        n = len(s['agent_ids'])
        b['anchor_hidden'][i, :, :n] = s['anchor_hidden']
        b['anchor_prediction'][i, :, :n] = s['anchor_prediction']
    return b


@torch.no_grad()
def cache_anchor(train, val, model, p, device):
    cached_train = []
    for original in DataLoader(train, batch_size=p['batch_size'], collate_fn=collate_trajectories):
        b = move_batch(original, device)
        base, encoded = model.encode_anchor(b['observed'], b['observed_valid'], b['timestamps'])
        base, hidden = base.cpu(), encoded['hidden'].cpu()
        for i in range(len(original['sample_id'])):
            n = int((original['agent_ids'][i] >= 0).sum())
            s = {k: original[k][i, :, :n] for k in ('observed', 'observed_valid', 'future', 'future_valid')}
            s.update(agent_ids=original['agent_ids'][i, :n], timestamps=original['timestamps'][i], future_timestamps=original['future_timestamps'][i], frame_ids=original['frame_ids'][i],
                scene=original['scene'][i], recording=original['recording'][i], sample_id=original['sample_id'][i],
                anchor_hidden=hidden[i, :, :n], anchor_prediction=base[i, :, :n])
            cached_train.append(s)
    cached_val = []
    for original in val:
        b = move_batch(original, device)
        base, encoded = model.encode_anchor(b['observed'], b['observed_valid'], b['timestamps'])
        cached_val.append(dict(original, anchor_prediction=base.cpu(), anchor_hidden=encoded['hidden'].cpu()))
    return cached_train, cached_val


def save_eval(output, label, result):
    csv_new(output/f'groups_{label}.csv', result['groups'])
    csv_new(output/f'targets_{label}.csv', result['targets'])
    write_json_new(output/f'diagnostics_{label}.json', result['diagnostics'])


def gradients(model):
    parts = {'all_trainable': [v for v in model.parameters() if v.requires_grad]}
    for name in ('encoder', 'decoder', 'payload', 'delta'):
        if hasattr(model, name):
            parts[name] = list(getattr(model, name).parameters())
    if hasattr(model, 'gate_logit'):
        parts['gate'] = [model.gate_logit]
    out = {}
    for name, parameters in parts.items():
        values = [v.grad.detach().square().sum() for v in parameters if v.grad is not None]
        if values:
            norm = torch.stack(values).sum().sqrt()
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient '+name)
            out[name+'_gradient_norm'] = float(norm)
    return out


def train_one(model, name, phase, schedule, seed, p, fold, train, val, source, device, anchor=None):
    output, local = output_path(fold, phase, name, seed, schedule)
    if output.exists() or local.exists():
        raise FileExistsError('Refusing to overwrite '+str(output))
    output.mkdir(parents=True); local.mkdir(parents=True)
    frozen = isinstance(model, FrozenEgoResidual)
    anchor_before = state_sha(model.anchor) if frozen else None
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=p['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=.3) if schedule == 'step10_20' else None
    loader = DataLoader(train, batch_size=p['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=collate_cached if frozen else collate_trajectories, num_workers=0)
    config = dict(model=name, phase=phase, recording=fold['recording'], seed=seed, schedule=schedule, protocol_name=p['protocol_name'], protocol_sha256=sha256(CONF),
        source_commit=source, source_sha256={str(x): sha256(x) for x in Path('src').rglob('*.py')}, manifest=fold['manifest'], manifest_sha256=fold['manifest_sha256'], raw_source_hashes=p['source_files'],
        training_windows=len(train), validation_windows=sum(len(b['sample_id']) for b in val), parameter_count=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in parameters), frozen_parameters=sum(v.numel() for v in model.parameters() if not v.requires_grad),
        anchor=anchor, anchor_parameters_sha256=anchor_before, no_future_input=True, ETH_evaluated=False, training=p,
        cache='new in-memory frozen anchor observation encodings/predictions; never old-project cache' if frozen else None,
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, GPU=torch.cuda.get_device_name(), cpu_threads=1))
    write_json_new(output/'config.json', config)
    best, best_state, best_epoch, steps, seconds, curves = None, None, None, 0, 0., []
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, p['epochs']+1):
        model.train(); lr = optimizer.param_groups[0]['lr']; loss_sum, count, grad_rows = 0., 0, []
        synchronize(device); t0 = time.perf_counter()
        for original in loader:
            b = move_batch(original, device); optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_cached(b) if frozen else model(b['observed'], b['observed_valid'], b['timestamps'])
            loss = masked_l2_loss(prediction, b['future'], b['future_valid'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward(); grad_rows.append(gradients(model))
            if frozen:
                assert all(v.grad is None for v in model.anchor.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, p['gradient_clip'], error_if_nonfinite=True); optimizer.step(); steps += 1
            n = int(b['future_valid'].any(1).sum()); count += n; loss_sum += float(loss)*n
        synchronize(device); duration = time.perf_counter()-t0; seconds += duration
        if frozen:
            assert state_sha(model.anchor) == anchor_before, 'Frozen anchor changed'
        ev = evaluate(model, val, device, cached=frozen)
        row = dict(epoch=epoch, learning_rate=lr, train_loss=loss_sum/count, validation_ADE_m=ev['ADE_m'], validation_FDE_m=ev['FDE_m'], training_seconds=duration, training_targets=count,
            anchor_unchanged=frozen if frozen else None, **ev['diagnostics'])
        for key in grad_rows[0]:
            values = [r[key] for r in grad_rows]; row[key+'_mean'] = float(np.mean(values)); row[key+'_p95'] = float(np.quantile(values, .95))
        curves.append(row)
        with (output/'epochs.csv').open('x' if epoch == 1 else 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row), lineterminator='\n')
            if epoch == 1: w.writeheader()
            w.writerow(row)
        if best is None or ev['ADE_m'] < best['ADE_m']:
            best, best_epoch = ev, epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler: scheduler.step()
        print(json.dumps(dict(recording=fold['recording'], phase=phase, model=name, schedule=schedule, seed=seed, epoch=epoch, loss=row['train_loss'], ADE=ev['ADE_m'], FDE=ev['FDE_m'], best_epoch=best_epoch, seconds=duration)), flush=True)
    last = ev; last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for endpoint, state, ep in [('best', best_state, best_epoch), ('last', last_state, p['epochs'])]:
        torch.save(dict(model=state, epoch=ep, config=config), local/f'{endpoint}.pt')
        model.load_state_dict(state)
        save_eval(output, endpoint, evaluate(model, val, device, cached=frozen, rows=True))
    summary = dict(model=name, phase=phase, recording=fold['recording'], seed=seed, schedule=schedule, status='completed', source_commit=source,
        best_epoch=best_epoch, best_ADE_m=best['ADE_m'], best_FDE_m=best['FDE_m'], last_epoch=p['epochs'], last_ADE_m=last['ADE_m'], last_FDE_m=last['FDE_m'],
        parameter_count=config['parameter_count'], trainable_parameters=config['trainable_parameters'], frozen_parameters=config['frozen_parameters'], training_windows=len(train), validation_windows=config['validation_windows'],
        optimizer_steps=steps, training_seconds=seconds, total_seconds=time.perf_counter()-start, peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        first_train_loss=curves[0]['train_loss'], last_train_loss=curves[-1]['train_loss'], tail5_relative_loss_decrease=(curves[-5]['train_loss']-curves[-1]['train_loss'])/max(curves[-5]['train_loss'], 1e-12),
        manifest_sha256=fold['manifest_sha256'], protocol_sha256=sha256(CONF), best_checkpoint_path=str(local/'best.pt'), last_checkpoint_path=str(local/'last.pt'),
        best_checkpoint_sha256=sha256(local/'best.pt'), last_checkpoint_sha256=sha256(local/'last.pt'), anchor=anchor,
        anchor_parameters_sha256_before=anchor_before, anchor_parameters_sha256_after=state_sha(model.anchor) if frozen else None, ETH_evaluated=False)
    write_json_new(output/'summary.json', summary)
    print('COMPLETED '+json.dumps(summary), flush=True)
    return summary


def analytic_cv(fold, val, source, device):
    output, _ = output_path(fold, 'ego', 'CV', 17, 'analytical')
    output.mkdir(parents=True); start = time.perf_counter()
    result = evaluate(HistoricalCV().to(device), val, device, rows=True); save_eval(output, 'analytical', result)
    write_json_new(output/'summary.json', dict(model='CV', phase='ego', seed=None, recording=fold['recording'], ADE_m=result['ADE_m'], FDE_m=result['FDE_m'], parameter_count=0,
        training_seconds=None, evaluation_seconds=time.perf_counter()-start, source_commit=source, manifest_sha256=fold['manifest_sha256'], ETH_evaluated=False))


def choose_anchors(p):
    selections = []
    for fold in p['folds']:
        candidates = []
        for label in p['neural_ego_runs']:
            name, schedule = label.split('/')
            output, _ = output_path(fold, 'ego', name, 17, schedule)
            s = json.loads((output/'summary.json').read_text()); assert s['status'] == 'completed'
            candidates.append(s)
        best = min(candidates, key=lambda r: r['best_ADE_m'])
        cv_path, _ = output_path(fold, 'ego', 'CV', 17, 'analytical')
        cv = json.loads((cv_path/'summary.json').read_text()); use_cv = cv['ADE_m'] <= best['best_ADE_m']
        selections.append(dict(recording=fold['recording'], model=best['model'], schedule=best['schedule'], checkpoint_path=best['best_checkpoint_path'], checkpoint_sha256=best['best_checkpoint_sha256'],
            cv_output=use_cv, anchor_ADE_m=cv['ADE_m'] if use_cv else best['best_ADE_m'], anchor_FDE_m=cv['FDE_m'] if use_cv else best['best_FDE_m'],
            best_neural_ADE_m=best['best_ADE_m'], candidate_summaries=candidates, seed=17, manifest_sha256=fold['manifest_sha256']))
    write_json_new(ROOT/'anchor_selection.json', dict(rule=p['anchor_rule'], selections=selections, stability='all predeclared finite runs completed', selected_before_social_training=True, ETH_evaluated=False))
    return selections


def load_anchor(selection, device):
    path = selection['checkpoint_path']; assert sha256(path) == selection['checkpoint_sha256']
    model = EGO[selection['model']]().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
    return model


def residual_runs(p, fold, selection, seed, source, device):
    train, val = data_load(p, fold)
    seed_all(seed); wrapper = FrozenEgoResidual(load_anchor(selection, device), 'Ego-Only', cv_output=selection['cv_output']).to(device)
    t0 = time.perf_counter(); train, val = cache_anchor(train, val, wrapper, p, device)
    cache_seconds = time.perf_counter()-t0
    output, _ = output_path(fold, 'residual', 'Ego-Only', seed)
    output.mkdir(parents=True); result = evaluate(wrapper, val, device, cached=True, rows=True); save_eval(output, 'analytical', result)
    assert abs(result['ADE_m']-selection['anchor_ADE_m']) < 1e-6
    write_json_new(output/'summary.json', dict(model='Ego-Only', seed=seed, recording=fold['recording'], ADE_m=result['ADE_m'], FDE_m=result['FDE_m'], anchor=selection,
        anchor_parameters_sha256=state_sha(wrapper.anchor), training_seconds=None, anchor_cache_seconds=cache_seconds, cache_persisted=False, source_commit=source, ETH_evaluated=False))
    for name in p['residual_models'][1:]:
        seed_all(seed); model = FrozenEgoResidual(load_anchor(selection, device), name, cv_output=selection['cv_output']).to(device)
        train_one(model, name, 'residual', p['residual_schedule'], seed, p, fold, train, val, source, device, selection)
        del model; torch.cuda.empty_cache()


def decide_round3(p):
    rows, eligible = [], []
    for name in p['conditional_candidate_priority']:
        passes = []
        for fold in p['folds']:
            paths = {m: output_path(fold, 'residual', m, 17)[0] for m in ('Ego-Only', 'Ego-Residual', name)}
            s = json.loads((paths[name]/'summary.json').read_text()); e = json.loads((paths['Ego-Residual']/'summary.json').read_text()); a = json.loads((paths['Ego-Only']/'summary.json').read_text())
            pass_ = s['best_ADE_m'] <= a['ADE_m']-max(.005, .005*a['ADE_m']) and s['best_ADE_m'] <= e['best_ADE_m']-max(.005, .005*e['best_ADE_m']) and s['best_FDE_m'] <= a['FDE_m'] and s['best_FDE_m'] <= e['best_FDE_m']
            passes.append(pass_); rows.append(dict(model=name, recording=fold['recording'], pass_fixed_rule=pass_, social_ADE_m=s['best_ADE_m'], social_FDE_m=s['best_FDE_m'], EgoOnly_ADE_m=a['ADE_m'], EgoResidual_ADE_m=e['best_ADE_m']))
        if all(passes): eligible.append(name)
    result = dict(rule=p['round3_rule'], qualified_models=eligible, run_extra_seeds=bool(eligible), selected_candidate=eligible[0] if eligible else None, rows=rows, ETH_evaluated=False)
    write_json_new(ROOT/'round3_decision.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['all', 'ego', 'residual'], default='all'); args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Commit protocol, source and tests before training')
    suites = ET.parse(ROOT/'pretraining_tests.xml').getroot()
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures'))
    p = json.loads(CONF.read_text()); assert p['epochs'] == 30 and not p['ETH_evaluated']
    torch.set_num_threads(1); source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(); device = torch.device('cuda')
    if args.phase in ('all', 'ego'):
        for fold in p['folds']:
            train, val = data_load(p, fold); analytic_cv(fold, val, source, device)
            for label in p['neural_ego_runs']:
                name, schedule = label.split('/'); seed_all(17); model = EGO[name]().to(device)
                train_one(model, name, 'ego', schedule, 17, p, fold, train, val, source, device)
                del model; torch.cuda.empty_cache()
        selections = choose_anchors(p)
    else:
        selections = json.loads((ROOT/'anchor_selection.json').read_text())['selections']
    if args.phase in ('all', 'residual'):
        for fold, selection in zip(p['folds'], selections): residual_runs(p, fold, selection, 17, source, device)
        decision = decide_round3(p)
        if decision['run_extra_seeds']:
            # Family/schedule/output mode frozen at17; all controls share each
            # newly trained seed-specific checkpoint, never reuse seed17 weights.
            for seed in p['conditional_seeds']:
                for fold, selection in zip(p['folds'], selections):
                    seed_all(seed); model = EGO[selection['model']]().to(device)
                    s = train_one(model, selection['model'], 'ego', selection['schedule'], seed, p, fold, *data_load(p, fold), source, device)
                    del model; torch.cuda.empty_cache()
                    candidate = dict(selection, seed=seed, checkpoint_path=s['best_checkpoint_path'], checkpoint_sha256=s['best_checkpoint_sha256'])
                    if not candidate['cv_output']:
                        candidate.update(anchor_ADE_m=s['best_ADE_m'], anchor_FDE_m=s['best_FDE_m'])
                    residual_runs(p, fold, candidate, seed, source, device)
        print('STAGE3P6_EXECUTION_COMPLETE '+json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
