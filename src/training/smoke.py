"""Controlled tiny training budget on one development split, no test access."""
from pathlib import Path
import argparse
import csv
import json
import random
import time
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.data.dataset import TrajectoryDataset
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256, write_json_new
from src.models import MODEL_REGISTRY
from src.evaluation.metrics import metric_sums, masked_l2_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def predict(model, batch):
    # Deliberately whitelist observation inputs; future/GT mask not forwarded.
    return model(batch['observed'], batch['observed_valid'], batch['timestamps'])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = dict(ade_sum=0., ade_count=0, fde_sum=0., fde_count=0)
    for original in loader:
        batch = move_batch(original, device)
        output = predict(model, batch)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite prediction')
        values = metric_sums(output, batch['future'], batch['future_valid'])
        for name, value in values.items():
            sums[name] += value.item()
    if sums['ade_count'] == 0 or sums['fde_count'] == 0:
        raise ValueError('No supervised ADE/FDE targets')
    return dict(ade=sums['ade_sum']/sums['ade_count'], fde=sums['fde_sum']/sums['fde_count'], **sums)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage2_smoke.json')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Commit tracked source/config changes before training')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device unavailable; do not silently change protocol')
    subset_manifest = json.loads(Path(config['subset_manifest']).read_text())
    if subset_manifest['data_manifest_sha256'] != sha256(config['data_manifest']):
        raise ValueError('Subset uses a different manifest')
    manifest = json.loads(Path(config['data_manifest']).read_text())
    if manifest['held_out_scene'] != config['held_out_scene'] or config['test_evaluation']:
        raise ValueError('Invalid smoke split; test evaluation must remain disabled')
    train = Subset(TrajectoryDataset(config['raw_root'], config['data_manifest'], 'train'), subset_manifest['train']['indices'])
    validation = Subset(TrajectoryDataset(config['raw_root'], config['data_manifest'], 'validation'), subset_manifest['validation']['indices'])
    fixed_train_loader = DataLoader(train, batch_size=config['batch_size'], collate_fn=collate_trajectories)
    validation_loader = DataLoader(validation, batch_size=config['batch_size'], collate_fn=collate_trajectories)
    out_root = Path('results/tables')/args.run_id
    local_root = Path('experiments')/args.run_id
    summary_path = out_root/'summary.csv'
    if out_root.exists() or local_root.exists():
        raise FileExistsError('Run ID already exists; never overwrite experiments')
    out_root.mkdir(parents=True)
    local_root.mkdir(parents=True)
    write_json_new(local_root/'config.json', config)
    environment = dict(python=__import__('sys').version, torch=torch.__version__, numpy=np.__version__,
                       cuda=torch.version.cuda, device=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
                       gpu_scatter_caveat='CUDA index_add uses floating point atomics; seeds do not guarantee bitwise equality')
    write_json_new(out_root/'environment.json', environment)
    rows = []
    epoch_rows = []
    for name in config['models']:
        seed_all(config['seed'])
        generator = torch.Generator().manual_seed(config['seed'])
        train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator,
                                  collate_fn=collate_trajectories, num_workers=0)
        # Common encoder and decoder initialize first, before social modules.
        model = MODEL_REGISTRY[name](hidden_dim=config['hidden_dim'], prediction_steps=config['prediction_steps'],
                                     max_order=config['max_order'], max_neighbors=config['max_neighbors'], radius=config['radius']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        start = time.perf_counter()
        initial_train = evaluate(model, fixed_train_loader, device)
        initial_validation = evaluate(model, validation_loader, device)
        train_seconds, steps = 0., 0
        for epoch in range(1, config['epochs']+1):
            model.train()
            synchronize(device)
            epoch_start = time.perf_counter()
            loss_sum, count = 0., 0
            for original in train_loader:
                batch = move_batch(original, device)
                if not batch['future_valid'].any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                output = predict(model, batch)
                loss = masked_l2_loss(output, batch['future'], batch['future_valid'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss: '+name)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite gradient: '+name)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                target_count = int(batch['future_valid'].any(1).sum().item())
                loss_sum += float(loss.item())*target_count
                count += target_count
                steps += 1
            synchronize(device)
            duration = time.perf_counter()-epoch_start
            train_seconds += duration
            val = evaluate(model, validation_loader, device)
            record = dict(model=name, epoch=epoch, train_loss=loss_sum/max(count, 1),
                          validation_ADE_m=val['ade'], validation_FDE_m=val['fde'], train_seconds=duration,
                          seed=config['seed'], source_commit=commit)
            epoch_rows.append(record)
            print(json.dumps(record), flush=True)
        final_train = evaluate(model, fixed_train_loader, device)
        synchronize(device)
        total_seconds = time.perf_counter()-start
        row = dict(run_id=args.run_id, model=name, status='completed', started_at_asia_shanghai=started,
            seed=config['seed'], source_commit=commit, tracked_source_dirty=False,
            data_manifest_sha256=sha256(config['data_manifest']), subset_manifest_sha256=sha256(config['subset_manifest']),
            config_sha256=sha256(args.config), parameter_count=sum(p.numel() for p in model.parameters()),
            hidden_dim=config['hidden_dim'], epochs=config['epochs'], optimizer_steps=steps,
            training_windows=len(train), validation_windows=len(validation), loss=config['loss'],
            initial_train_loss=initial_train['ade'], final_train_loss=final_train['ade'],
            initial_validation_ADE_m=initial_validation['ade'], initial_validation_FDE_m=initial_validation['fde'],
            validation_ADE_m=val['ade'], validation_FDE_m=val['fde'],
            validation_ADE_target_count=int(val['ade_count']), validation_FDE_target_count=int(val['fde_count']),
            cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else '',
            cuda_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20 if device.type == 'cuda' else '',
            actual_training_seconds=train_seconds, total_run_seconds=total_seconds,
            hardware=environment['device'], test_evaluated=False, sampling_K=1)
        rows.append(row)
        write_json_new(out_root/(name.replace('-', '_')+'.json'), dict(summary=row, config=config, environment=environment,
                       epoch_history=[r for r in epoch_rows if r['model'] == name]))
        checkpoint = local_root/'checkpoints'
        checkpoint.mkdir(exist_ok=True)
        torch.save(dict(model=model.state_dict(), config=config, source_commit=commit), checkpoint/(name+'.pt'))
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for path, records in [(summary_path, rows), (out_root/'epochs.csv', epoch_rows)]:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    print('Smoke results: '+str(summary_path))


if __name__ == '__main__':
    main()
