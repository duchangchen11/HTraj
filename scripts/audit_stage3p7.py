"""Before-training source audit, split hashes and history-only distributions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256, write_json_new
from src.training.frozen_ego import data_load
from src.training.motion_generalization import attach_groups
from src.training.social_diagnostics import csv_new


def main():
    root = Path('results/tables/stage3p7'); p = json.loads(Path('configs/stage3p7/protocol.json').read_text())
    torch.set_num_threads(1); distribution, counts = [], []
    for info in p['source_files'].values():
        assert info['scene'] != 'ETH'
        assert sha256(Path(p['raw_root'])/info['file']) == info['sha256']
    for fold in p['folds']:
        train, val = data_load(p, fold)
        for split, batches in [('train', DataLoader(train, batch_size=8, collate_fn=collate_trajectories)), ('validation', val)]:
            fields = {}; groups = {}; directions = [0]*8; n = 0; window_count = 0
            for b in batches:
                attach_groups(b); target = b['future_valid'].all(1); n += int(target.sum()); window_count += len(b['sample_id'])
                for name, mask in b['groups'].items(): groups[name] = groups.get(name, 0)+int((target & mask).sum())
                for name, value in b['history_features'].items(): fields.setdefault(name, []).extend(value[target].double().tolist())
                f = b['history_features']; reliable = target & b['groups']['heading_reliable']
                angle = torch.atan2(f['heading_unit_y'][reliable], f['heading_unit_x'][reliable])
                bins = torch.floor((angle+torch.pi)/(torch.pi/4)).long().clamp(0, 7)
                for k in range(8): directions[k] += int((bins == k).sum())
            counts.append(dict(recording=fold['recording'], split=split, windows=window_count, target_windows=n,
                complete8x12_targets=n, partial_history_targets=0, manifest_sha256=fold['manifest_sha256'], **groups))
            for name, values in fields.items():
                a = np.asarray(values)
                distribution.append(dict(recording=fold['recording'], split=split, measure=name, target_windows=n,
                    mean=float(a.mean()), p05=float(np.quantile(a, .05)), median=float(np.median(a)), p95=float(np.quantile(a, .95))))
            for k, value in enumerate(directions):
                distribution.append(dict(recording=fold['recording'], split=split, measure=f'reliable_heading_bin_{-180+45*k}_to_{-135+45*k}_deg',
                    target_windows=n, count=value, fraction_of_reliable=value/max(sum(directions), 1)))
    csv_new(root/'split_counts.csv', counts); csv_new(root/'history_distribution.csv', distribution)
    source = [Path('src/models')/name for name in ('ego_encoder.py', 'trajectory_decoder.py', 'ped_base.py', 'motion_normalization.py')]
    write_json_new(root/'pretraining_audit.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        stage='before-training', trained_models=0, inputs=[dict(path=str(x), sha256=sha256(x)) for x in source],
        M1_equivalent_to_M0=True, encoder_absolute_coordinates=False, encoder_velocity_mps=True, encoder_acceleration_mps2=True,
        encoder_features=7, original_translation_equivariant=True, original_rotation_equivariant=False,
        speed_missing_mask_explicit=True, nominal_dt=.4, prediction_global_anchor='last observed position + direct 12-step offset',
        ETH_evaluated=False, independent_confirmation=False))
    print(json.dumps(counts))


if __name__ == '__main__': main()
