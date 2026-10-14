"""Inspect saved smoke states on the same development validation subset."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TrajectoryDataset
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256
from src.models import MODEL_REGISTRY


def main():
    torch.set_num_threads(1)
    config = json.loads(Path('configs/stage2_smoke.json').read_text())
    subset = json.loads(Path(config['subset_manifest']).read_text())
    validation = Subset(TrajectoryDataset(config['raw_root'], config['data_manifest'], 'validation'), subset['validation']['indices'])
    loader = DataLoader(validation, batch_size=8, collate_fn=collate_trajectories)
    rows = []
    for name in ('FirstOrder-GNN', 'TOHI-Net-A'):
        checkpoint = torch.load(Path('experiments/stage2_smoke_v1/checkpoints')/(name+'.pt'), map_location='cpu', weights_only=True)
        model = MODEL_REGISTRY[name](hidden_dim=config['hidden_dim'], prediction_steps=12,
                                    max_order=3, max_neighbors=8, radius=4.0).eval()
        model.load_state_dict(checkpoint['model'])
        orders = model.order_state.max_order
        states = [[] for _ in range(orders)]
        supports = [[] for _ in range(orders)]
        weights = [[] for _ in range(orders)]
        ego_norms = []
        with torch.no_grad():
            for batch in loader:
                _, debug = model(batch['observed'], batch['observed_valid'], batch['timestamps'], return_debug=True)
                valid = batch['observed_valid'][:, -1]
                ego_norms.extend(debug['hidden'][:, -1].norm(dim=-1)[valid].tolist())
                for q in range(orders):
                    states[q].extend(debug['order_states'][:, -1, :, q].norm(dim=-1)[valid].tolist())
                    supports[q].extend(debug['order_support'][:, -1, :, q][valid].tolist())
                    weights[q].extend(debug['order_weights'][:, :, q][valid].tolist())
        for q in range(orders):
            active = np.asarray(supports[q], bool)
            norm = np.asarray(states[q])
            beta = np.asarray(weights[q])
            rows.append(dict(model=name, social_order=q+1, cutoff_people=len(active),
                             support_fraction=float(active.mean()),
                             mean_state_L2=float(norm.mean()), state_L2_p95=float(np.quantile(norm, .95)),
                             mean_state_L2_if_supported=float(norm[active].mean()) if active.any() else 0.,
                             mean_fusion_weight=float(beta.mean()), mean_ego_L2=float(np.mean(ego_norms)),
                             source_commit=checkpoint['source_commit'], subset_manifest_sha256=sha256(config['subset_manifest']),
                             future_gt_used=False, checkpoint_sha256=sha256(Path('experiments/stage2_smoke_v1/checkpoints')/(name+'.pt'))))
    output = Path('results/tables/stage2_smoke_v1/state_diagnostics.csv')
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
