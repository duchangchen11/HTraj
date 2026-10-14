"""Stage 2.5 fixed-budget development diagnostics. Never loads the ETH test split."""
from pathlib import Path
import argparse
import csv
import json
import time
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .smoke import seed_all, move_batch, synchronize
from ..data.dataset import TrajectoryDataset
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256, write_json_new
from ..models.social_diagnostics_models import DIAGNOSTIC_MODELS
from ..evaluation.metrics import metric_sums, masked_l2_loss
from ..evaluation.social_diagnostics import GROUPS, history_groups, complete_debug, extract_debug, summarize_debug

ROOT = Path('results/tables/stage2p5')
PROTOCOL = Path('configs/stage2p5/protocol.json')


def csv_new(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def cached_validation(dataset, batch_size):
    batches = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_trajectories):
        batch['groups'] = history_groups(batch)
        batches.append(batch)
    return batches


@torch.no_grad()
def evaluate(model, batches, device, intervention=None, reference_predictions=None):
    model.eval()
    sums = {group: dict(ade_sum=0., ade_count=0, fde_sum=0., fde_count=0, observed_targets=0,
                       observed_windows=0, ade_windows=0) for group in GROUPS}
    diagnostic_batches, predictions = [], []
    change_sum, change_count = 0., 0
    graph_changed, graph_total = 0, 0
    for index, original in enumerate(batches):
        batch = move_batch(original, device)
        options = {'return_debug': True}
        if intervention:
            options['intervention'] = intervention
        prediction, debug = model(batch['observed'], batch['observed_valid'], batch['timestamps'], **options)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation')
        debug = complete_debug(model, debug)
        diagnostic_batches.append(extract_debug(debug))
        predictions.append(prediction.cpu())
        for group, mask in original['groups'].items():
            valid = batch['future_valid'] & mask.to(device)[:, None]
            values = metric_sums(prediction, batch['future'], valid)
            for key, value in values.items():
                sums[group][key] += value.item()
            sums[group]['observed_targets'] += int(mask.sum())
            sums[group]['observed_windows'] += int(mask.any(-1).sum())
            sums[group]['ade_windows'] += int(valid.any((1, 2)).sum())
        if reference_predictions is not None:
            difference = (prediction.cpu()-reference_predictions[index]).norm(dim=-1)
            valid = original['observed_valid'][:, -1, None].expand(-1, prediction.shape[1], -1)
            change_sum += float(difference[valid].sum())
            change_count += int(valid.sum())
        if intervention and intervention.get('shuffle_graph'):
            num_nodes = batch['observed'].shape[0]*batch['observed'].shape[2]
            for old, new in zip(debug['original_graph_frames'], debug['graph_frames']):
                old_keys = old.target*num_nodes+old.source
                new_keys = new.target*num_nodes+new.source
                graph_changed += int((~torch.isin(new_keys, old_keys)).sum())
                graph_total += len(old_keys)
    groups = []
    for group, values in sums.items():
        groups.append(dict(group=group, **values,
            ADE_m=values['ade_sum']/values['ade_count'] if values['ade_count'] else None,
            FDE_m=values['fde_sum']/values['fde_count'] if values['fde_count'] else None))
    all_values = groups[0]
    if not all_values['ade_count'] or not all_values['fde_count']:
        raise ValueError('No supervised validation targets')
    extra = summarize_debug(diagnostic_batches)
    if reference_predictions is not None:
        extra['prediction_intervention_mean_L2_m'] = change_sum/max(change_count, 1)
    if intervention and intervention.get('shuffle_graph'):
        extra['negative_graph_changed_edge_fraction'] = graph_changed/max(graph_total, 1)
    return dict(ADE_m=all_values['ADE_m'], FDE_m=all_values['FDE_m'], groups=groups,
                diagnostics=extra, predictions=predictions)


def gradient_values(model):
    groups = dict(ego=list(model.encoder.parameters()), decoder=list(model.decoder.parameters()))
    if hasattr(model, 'fusion'):
        groups['fusion'] = list(model.fusion.parameters())
        if hasattr(model.fusion, 'delta'):
            groups['delta'] = list(model.fusion.delta.parameters())
            groups['residual_gate'] = list(model.fusion.gate.parameters())
    if hasattr(model, 'graph'):
        groups['graph'] = list(model.graph.parameters())
    if hasattr(model, 'order_state'):
        for q in range(model.order_state.max_order):
            groups[f'q{q+1}_projection'] = list(model.order_state.projections[q].parameters())
            groups[f'q{q+1}_carry_gate'] = list(model.order_state.gates[q].parameters())
    if hasattr(model, 'projection'):
        groups['dynamic_projection'] = list(model.projection.parameters())
        groups['dynamic_gru'] = list(model.social_gru.parameters())
    values = {}
    for name, parameters in groups.items():
        grads = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
        if grads:
            norm = torch.stack(grads).sum().sqrt()
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient: '+name)
            values[name+'_grad_L2'] = float(norm)
    return values


def choose_social(rows, protocol):
    ped = next(row for row in rows if row['model'] == 'PedBase')
    social = sorted([row for row in rows if not row['model'].startswith('PedBase')], key=lambda r: (r['best_ADE_m'], r['model']))
    paused = all(row['best_ADE_m'] > 1.05*ped['best_ADE_m'] and row['best_FDE_m'] >= ped['best_FDE_m'] for row in social)
    selected = []
    if not paused:
        selected.append(social[0]['model'])
        tohi = min([row for row in social if row['model'].startswith('TOHI')], key=lambda r: r['best_ADE_m'])
        if tohi['model'] != selected[0] and tohi['best_ADE_m'] <= 1.05*social[0]['best_ADE_m']:
            selected.append(tohi['model'])
        elif len(social) > 1:
            selected.append(social[1]['model'])
    return dict(round2_status='paused_by_fixed_failure_rule' if paused else 'authorized_by_fixed_rule',
                selected_social_models=selected, criterion=protocol['selection_rule'], pause_criterion=protocol['first_round_pause_rule'],
                PedBase_best_ADE_m=ped['best_ADE_m'], ranked_social=[dict(model=r['model'], best_ADE_m=r['best_ADE_m'], best_FDE_m=r['best_FDE_m']) for r in social],
                test_evaluated=False)


def train_one(name, seed, phase, protocol, train, validation_batches, subset_path, source_commit, device):
    output = ROOT/phase/name/f'seed_{seed}'
    local = Path('experiments/stage2p5')/phase/name/f'seed_{seed}'/'checkpoints'
    if output.exists() or local.exists():
        raise FileExistsError('Existing run must not be overwritten: '+str(output))
    output.mkdir(parents=True)
    local.mkdir(parents=True)
    seed_all(seed)
    model = DIAGNOSTIC_MODELS[name](hidden_dim=protocol['hidden_dim'], prediction_steps=12,
                                   max_order=3, max_neighbors=8, radius=4.).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    loader = DataLoader(train, batch_size=protocol['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate_trajectories, num_workers=0)
    config = dict(model=name, seed=seed, phase=phase, protocol=protocol, source_commit=source_commit,
                  data_manifest_sha256=sha256(protocol['data_manifest']), subset_manifest_sha256=sha256(subset_path),
                  protocol_sha256=sha256(PROTOCOL), test_evaluated=False, tracked_source_dirty=False,
                  parameter_count=sum(p.numel() for p in model.parameters()), training_windows=len(train),
                  validation_windows=sum(len(b['sample_id']) for b in validation_batches),
                  state_kind='dynamic_mixed' if name.startswith('Dynamic') else 'graded' if not name.startswith('PedBase') else 'ego_only',
                  effective_hidden_dim=model.encoder.hidden_dim, started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, hardware=torch.cuda.get_device_name(),
                                   cpu_threads=torch.get_num_threads(), scatter_bitwise_determinism=False))
    write_json_new(output/'config.json', config)
    epoch_path = output/'epochs.csv'
    all_epochs, best, best_state, best_epoch, steps = [], None, None, None, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    train_seconds = 0.
    for epoch in range(1, protocol['epochs']+1):
        model.train()
        synchronize(device)
        epoch_start = time.perf_counter()
        loss_sum, count = 0., 0
        grad_rows = []
        for original in loader:
            batch = move_batch(original, device)
            if not batch['future_valid'].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch['observed'], batch['observed_valid'], batch['timestamps'])
            loss = masked_l2_loss(prediction, batch['future'], batch['future_valid'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            grad_rows.append(gradient_values(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            targets = int(batch['future_valid'].any(1).sum())
            count += targets
            loss_sum += float(loss)*targets
            steps += 1
        synchronize(device)
        duration = time.perf_counter()-epoch_start
        train_seconds += duration
        validation = evaluate(model, validation_batches, device)
        gradients = {key: float(np.mean([row[key] for row in grad_rows if key in row]))
                     for key in set().union(*(r.keys() for r in grad_rows))}
        record = dict(epoch=epoch, train_loss=loss_sum/max(count, 1), validation_ADE_m=validation['ADE_m'],
                      validation_FDE_m=validation['FDE_m'], training_seconds=duration, source_commit=source_commit,
                      seed=seed, **validation['diagnostics'], **gradients)
        all_epochs.append(record)
        # Publish each completed epoch immediately; no fabricated missing epochs.
        if epoch == 1:
            fields = list(record)
            with epoch_path.open('x', newline='') as stream:
                csv.DictWriter(stream, fieldnames=fields, lineterminator='\n').writeheader()
        with epoch_path.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields, lineterminator='\n').writerow(record)
        if best is None or validation['ADE_m'] < best['ADE_m']:
            best, best_epoch = validation, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(dict(phase=phase, model=name, seed=seed, epoch=epoch,
                             loss=record['train_loss'], ADE=validation['ADE_m'], FDE=validation['FDE_m'],
                             best_epoch=best_epoch, seconds=round(duration, 2))), flush=True)
    last = validation
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for label, state, saved_epoch in [('best', best_state, best_epoch), ('last', last_state, protocol['epochs'])]:
        torch.save(dict(model=state, epoch=saved_epoch, config=config), local/(label+'.pt'))
    for label, result in [('best', best), ('last', last)]:
        csv_new(output/f'groups_{label}.csv', result['groups'])
    tail = [r['train_loss'] for r in all_epochs[-5:]]
    decrease = (tail[0]-tail[-1])/max(abs(tail[0]), 1e-12)
    summary = dict(model=name, seed=seed, phase=phase, status='completed', parameter_count=config['parameter_count'],
        best_epoch=best_epoch, best_ADE_m=best['ADE_m'], best_FDE_m=best['FDE_m'],
        last_epoch=protocol['epochs'], last_ADE_m=last['ADE_m'], last_FDE_m=last['FDE_m'],
        first_epoch_train_loss=all_epochs[0]['train_loss'], last_epoch_train_loss=all_epochs[-1]['train_loss'],
        tail5_relative_train_loss_decrease=decrease, convergence_flag=decrease < .02,
        optimizer_steps=steps, training_windows=len(train), validation_windows=config['validation_windows'],
        actual_training_seconds=train_seconds, total_run_seconds=time.perf_counter()-start,
        cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        cuda_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20,
        source_commit=source_commit, data_manifest_sha256=config['data_manifest_sha256'],
        subset_manifest_sha256=config['subset_manifest_sha256'], protocol_sha256=config['protocol_sha256'],
        test_evaluated=False, state_kind=config['state_kind'], best_checkpoint_sha256=sha256(local/'best.pt'),
        last_checkpoint_sha256=sha256(local/'last.pt'))
    write_json_new(output/'summary.json', summary)
    print('COMPLETED '+json.dumps(summary), flush=True)
    del model, optimizer, prediction, loss, batch
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['round1', 'round2'], required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Commit source/config before training')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['test_evaluation'] or protocol['held_out_scene'] != 'ETH' or protocol['epochs'] > 30:
        raise ValueError('Protocol changed beyond authorized scope')
    torch.set_num_threads(protocol['cpu_threads'])
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Configured GPU unavailable')
    if args.phase == 'round1':
        subset_path, models, seeds = protocol['small_subset'], protocol['round1_models'], protocol['round1_seeds']
    else:
        selection = json.loads((ROOT/'selection.json').read_text())
        if not selection['selected_social_models']:
            print('Round2 explicitly paused by the predeclared failure rule. No training executed.')
            return
        subset_path, models, seeds = protocol['large_subset'], ['PedBase', *selection['selected_social_models']], protocol['round2_seeds']
    subset = json.loads(Path(subset_path).read_text())
    if subset['data_manifest_sha256'] != sha256(protocol['data_manifest']):
        raise ValueError('Subset manifest mismatch')
    data_manifest = json.loads(Path(protocol['data_manifest']).read_text())
    if any(x['scene'] == 'ETH' for split in ('train', 'validation') for x in data_manifest['splits'][split]):
        raise ValueError('Held-out test data leaked into development splits')
    train = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'train'), subset['train']['indices'])
    validation = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'validation'), subset['validation']['indices'])
    validation_batches = cached_validation(validation, protocol['batch_size'])
    rows = []
    for name in models:
        for seed in seeds:
            rows.append(train_one(name, seed, args.phase, protocol, train, validation_batches, subset_path, source_commit, device))
    csv_new(ROOT/args.phase/'summary.csv', rows)
    if args.phase == 'round1':
        write_json_new(ROOT/'selection.json', choose_social(rows, protocol))


if __name__ == '__main__':
    main()
