"""Stage3.7 finite representation-only experiment; no archived result writes."""
import csv
import json
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import DataLoader
from .smoke import seed_all, move_batch, synchronize
from .social_diagnostics import csv_new
from .frozen_ego import data_load, gradients, state_sha
from ..data.collate import collate_trajectories
from ..data.preprocessing import sha256, write_json_new
from ..models.ped_base import PedBase
from ..models.motion_normalization import HeadingNormalized, heading_frame
from ..models.relation_routing import HistoricalCV
from ..evaluation.frozen_ego import evaluate as old_evaluate, history_groups, GROUPS as OLD_GROUPS
from ..evaluation.metrics import masked_l2_loss

ROOT = Path('results/tables/stage3p7')
CONF = Path('configs/stage3p7/protocol.json')
LOCAL = Path('experiments/stage3p7')
EXTRA = ['heading_reliable', 'heading_fallback', 'history_speed_low', 'history_speed_medium', 'history_speed_high', 'history_speed_unknown']


def model_for(name, threshold=.2):
    return PedBase() if name == 'M0' else HeadingNormalized(cv_residual=name == 'M3', threshold=threshold)


def attach_groups(batch):
    batch['groups'], batch['history_features'] = history_groups(batch)
    f = heading_frame(batch['observed'], batch['observed_valid'], batch['timestamps'])
    valid, speed, known = f['valid'][:, -1], f['heading_speed'], f['duration'] > 0
    batch['groups'].update(heading_reliable=valid & f['reliable'], heading_fallback=valid & ~f['reliable'],
        history_speed_low=valid & known & (speed < .2), history_speed_medium=valid & known & (speed >= .2) & (speed < 1),
        history_speed_high=valid & known & (speed >= 1), history_speed_unknown=valid & ~known)
    batch['history_features'].update(heading_speed_mps=speed, heading_reliable_flag=f['reliable'],
        heading_unit_x=f['unit'][..., 0], heading_unit_y=f['unit'][..., 1])
    return batch


def evaluate(model, batches, device, rows=False):
    # Preserve the archived evaluator byte-for-byte; extend only its output
    # group list temporarily for a sequential, single-process evaluation.
    import src.evaluation.frozen_ego as evaluator
    previous = evaluator.GROUPS
    try:
        evaluator.GROUPS = list(OLD_GROUPS)+EXTRA
        return old_evaluate(model, batches, device, rows=rows)
    finally:
        evaluator.GROUPS = previous


def save_eval(directory, endpoint, split, r):
    csv_new(directory/f'groups_{endpoint}_{split}.csv', r['groups'])
    if r['targets']: csv_new(directory/f'targets_{endpoint}_{split}.csv', r['targets'])
    write_json_new(directory/f'diagnostics_{endpoint}_{split}.json', r['diagnostics'])


def train_one(name, p, fold, train, val, train_eval, source, device):
    out = ROOT/fold['recording']/name/'seed_17'; local = LOCAL/fold['recording']/name/'seed_17'/'checkpoints'
    if out.exists() or local.exists(): raise FileExistsError('Refusing to overwrite '+str(out))
    out.mkdir(parents=True); local.mkdir(parents=True)
    seed_all(p['seed']); model = model_for(name, p['heading_threshold_mps']).to(device)
    initial_sha = state_sha(model)
    parameters = list(model.parameters()); optimizer = torch.optim.Adam(parameters, lr=p['learning_rate'])
    loader = DataLoader(train, batch_size=p['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(p['seed']), collate_fn=collate_trajectories, num_workers=0)
    config = dict(model=name, seed=p['seed'], recording=fold['recording'], training=p, protocol_sha256=sha256(CONF),
        source_commit=source, source_sha256={str(x): sha256(x) for x in Path('src').rglob('*.py')},
        parameter_count=sum(x.numel() for x in parameters), initial_parameters_sha256=initial_sha,
        manifest_sha256=fold['manifest_sha256'], training_windows=len(train), validation_windows=sum(len(b['sample_id']) for b in val),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, GPU=torch.cuda.get_device_name(), cpu_threads=1),
        ETH_evaluated=False, future_input=False)
    write_json_new(out/'config.json', config)
    best = None; best_state = None; best_epoch = None; seconds = 0.; steps = 0; curves = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, p['epochs']+1):
        model.train(); losses = 0.; count = 0; grads = []
        synchronize(device); start = time.perf_counter()
        for original in loader:
            b = move_batch(original, device); optimizer.zero_grad(set_to_none=True)
            prediction = model(b['observed'], b['observed_valid'], b['timestamps'])
            loss = masked_l2_loss(prediction, b['future'], b['future_valid'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); grads.append(gradients(model)); torch.nn.utils.clip_grad_norm_(parameters, p['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); steps += 1
            n = int(b['future_valid'].any(1).sum()); count += n; losses += float(loss)*n
        synchronize(device); duration = time.perf_counter()-start; seconds += duration
        r = evaluate(model, val, device)
        row = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], train_online_ADE_loss_m=losses/count,
            validation_ADE_m=r['ADE_m'], validation_FDE_m=r['FDE_m'], training_seconds=duration, training_targets=count,
            **{k: float(np.mean([g[k] for g in grads])) for k in grads[0]})
        curves.append(row)
        with (out/'epochs.csv').open('x' if epoch == 1 else 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row), lineterminator='\n')
            if epoch == 1: w.writeheader()
            w.writerow(row)
        if best is None or r['ADE_m'] < best['ADE_m']:
            best = r; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(dict(recording=fold['recording'], model=name, epoch=epoch, train_ADE=row['train_online_ADE_loss_m'], ADE=r['ADE_m'], FDE=r['FDE_m'], best_epoch=best_epoch, seconds=duration)), flush=True)
    last = r; last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    endpoints = {}
    for endpoint, state, ep in [('best', best_state, best_epoch), ('last', last_state, p['epochs'])]:
        torch.save(dict(model=state, epoch=ep, config=config), local/f'{endpoint}.pt'); model.load_state_dict(state)
        for split, batches in [('train', train_eval), ('validation', val)]:
            result = evaluate(model, batches, device, rows=split == 'validation'); save_eval(out, endpoint, split, result)
            endpoints[endpoint+'_'+split+'_ADE_m'] = result['ADE_m']; endpoints[endpoint+'_'+split+'_FDE_m'] = result['FDE_m']
    summary = dict(model=name, seed=17, recording=fold['recording'], status='completed', source_commit=source, parameter_count=config['parameter_count'],
        initial_parameters_sha256=initial_sha, best_epoch=best_epoch, best_ADE_m=best['ADE_m'], best_FDE_m=best['FDE_m'],
        last_epoch=30, last_ADE_m=last['ADE_m'], last_FDE_m=last['FDE_m'], **endpoints,
        training_seconds=seconds, total_seconds=time.perf_counter()-started, optimizer_steps=steps,
        training_windows=len(train), validation_windows=config['validation_windows'],
        protocol_sha256=sha256(CONF), manifest_sha256=fold['manifest_sha256'],
        best_checkpoint_path=str(local/'best.pt'), last_checkpoint_path=str(local/'last.pt'),
        best_checkpoint_sha256=sha256(local/'best.pt'), last_checkpoint_sha256=sha256(local/'last.pt'),
        peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20, ETH_evaluated=False)
    write_json_new(out/'summary.json', summary); print('COMPLETED '+json.dumps(summary), flush=True)


def main():
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Commit all source/config/tests before training')
    suites = ET.parse(ROOT/'pretraining_tests_final.xml').getroot()
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures'))
    p = json.loads(CONF.read_text()); assert not p['ETH_evaluated'] and p['epochs'] == 30 and p['seed'] == 17
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    torch.set_num_threads(1); device = torch.device('cuda')
    for fold in p['folds']:
        train, val = data_load(p, fold); val = [attach_groups(b) for b in val]
        train_eval = [attach_groups(b) for b in DataLoader(train, batch_size=p['batch_size'], collate_fn=collate_trajectories)]
        directory = ROOT/fold['recording']/'CV'; directory.mkdir(parents=True, exist_ok=False)
        cv_metrics = {}
        for split, batches in [('train', train_eval), ('validation', val)]:
            r = evaluate(HistoricalCV().to(device), batches, device, rows=split == 'validation'); save_eval(directory, 'analytical', split, r)
            cv_metrics.update({split+'_ADE_m': r['ADE_m'], split+'_FDE_m': r['FDE_m']})
        write_json_new(directory/'summary.json', dict(model='CV', recording=fold['recording'], **cv_metrics, parameter_count=0,
            source_commit=source, manifest_sha256=fold['manifest_sha256'], training_seconds=None, ETH_evaluated=False))
        for name in p['neural_models']: train_one(name, p, fold, train, val, train_eval, source, device)
    print('STAGE3P7_FINITE_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__': main()
