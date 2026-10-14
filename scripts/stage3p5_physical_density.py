"""Exploratory physical-density groups, separate from frozen legal graph groups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import TrajectoryDataset
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256, write_json_new
from src.training.social_diagnostics import csv_new

ROOT = Path('results/tables/stage3p5')


def main():
    torch.set_num_threads(1)
    data = TrajectoryDataset('data/raw/eth_ucy', 'configs/stage3p5/folds/val_students003.json', 'validation')
    features = {}
    for b in DataLoader(data, batch_size=8, collate_fn=collate_trajectories):
        pos, valid = b['observed'], b['observed_valid']
        n = pos.shape[2]
        distance = (pos[:, :, :, None] - pos[:, :, None, :]).norm(dim=-1)
        neighbors = valid[:, :, :, None] & valid[:, :, None, :] & (distance <= 4)
        neighbors &= ~torch.eye(n, dtype=torch.bool)[None, None]
        degree = neighbors.sum(-1)
        mean = degree.sum(1) / valid.sum(1).clamp_min(1)
        for i, j in torch.nonzero(valid[:, -1]).tolist():
            key = (b['recording'][i], b['sample_id'][i], int(b['agent_ids'][i, j]))
            assert key not in features
            features[key] = dict(raw_history_physical_degree_mean=float(mean[i, j]), cutoff_physical_neighbors=int(degree[i, -1, j]))
    groups = {
        'physical_no_neighbors_all_observed_history': lambda d: d == 0,
        'physical_low_density_mean_0to1': lambda d: 0 < d <= 1,
        'physical_medium_density_mean_1to3': lambda d: 1 < d <= 3,
        'physical_high_density_mean_over3': lambda d: d > 3,
    }
    pairs = list(csv.DictReader((ROOT / 'pilot_paired_target_differences.csv').open()))
    targets, aggregate = [], []
    for r in pairs:
        key = (r['recording'], r['sample_id'], int(r['target_id']))
        f = features[key]
        selected = [name for name, test in groups.items() if test(f['raw_history_physical_degree_mean'])]
        assert len(selected) == 1
        targets.append(dict(r, **f, physical_density_group=selected[0]))
    for setting in sorted({r['setting'] for r in targets}):
        for endpoint in ('best', 'last'):
            rows = [r for r in targets if r['setting'] == setting and r['endpoint'] == endpoint]
            counts = 0
            for group in groups:
                subset = [r for r in rows if r['physical_density_group'] == group]
                final = [r for r in subset if r['FDE_PedBase'] != '']
                counts += len(subset)
                result = dict(setting=setting, endpoint=endpoint, group=group, ADE_count=len(subset), FDE_count=len(final), unique_people=len({(r['recording'], r['target_id']) for r in subset}))
                if subset:
                    result['history_frames_mean'] = float(np.mean([float(r['history_frames']) for r in subset]))
                    for model in ('CV', 'PedBase', 'Dynamic'):
                        result[model + '_ADE_m'] = float(np.mean([float(r['ADE_' + model]) for r in subset]))
                        result[model + '_FDE_m'] = float(np.mean([float(r['FDE_' + model]) for r in final])) if final else None
                    result['delta_Dynamic_minus_PedBase_ADE_m'] = result['Dynamic_ADE_m'] - result['PedBase_ADE_m']
                    result['delta_Dynamic_minus_PedBase_FDE_m'] = result['Dynamic_FDE_m'] - result['PedBase_FDE_m'] if final else None
                aggregate.append(result)
            assert counts == len(rows)
    csv_new(ROOT / 'physical_density_paired_targets.csv', targets)
    csv_new(ROOT / 'physical_density_group_comparison.csv', aggregate)
    write_json_new(ROOT / 'physical_density_provenance.json', dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha256(__file__), input_pairs_sha256=sha256(ROOT / 'pilot_paired_target_differences.csv'),
        cutoff_distance_m=4, mean_over='raw observed-valid receiver frames, all simultaneously visible sources; no speed/cap/CPA condition',
        population='protocol-specific evaluable targets; historical physical context identical for common targets',
        analysis_status='exploratory after training, no model or checkpoint selection; frozen primary groups unchanged',
        future_GT_used_for_grouping=False, ETH_evaluated=False,
    ))
    print([r for r in aggregate if r['endpoint'] == 'best'])


if __name__ == '__main__':
    main()
