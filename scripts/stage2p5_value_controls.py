"""Separate zero social VALUE from the no-path ego fallback. No retraining."""
from pathlib import Path
import json
import sys
import hashlib
import torch
from torch.utils.data import Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TrajectoryDataset
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS
from src.training.social_diagnostics import ROOT, PROTOCOL, cached_validation, csv_new
from src.evaluation.metrics import metric_sums
from src.data.preprocessing import sha256


def main():
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    subset = json.loads(Path(protocol['small_subset']).read_text())
    data = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'validation'), subset['validation']['indices'])
    batches = cached_validation(data, protocol['batch_size'])
    rows = []
    for name in ['TOHI-A0','TOHI-A-R','TOHI-A-N','TOHI-A-RN']:
        path = Path('experiments/stage2p5/round1')/name/'seed_17/checkpoints/best.pt'
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        model = DIAGNOSTIC_MODELS[name](hidden_dim=64).eval()
        model.load_state_dict(checkpoint['model'])
        for keep_support in (True, False):
            totals = dict(ade_sum=0., ade_count=0, fde_sum=0., fde_count=0)
            changes, predictions = 0., 0
            with torch.no_grad():
                for batch in batches:
                    full, debug = model(batch['observed'], batch['observed_valid'], batch['timestamps'], return_debug=True)
                    support = debug['order_support'][:, -1] if keep_support else torch.zeros_like(debug['order_support'][:, -1])
                    context, _, _ = model.fusion(debug['hidden'][:, -1], torch.zeros_like(debug['order_states'][:, -1]), support)
                    altered = model.decoder(context, debug['positions'][:, -1], debug['valid'][:, -1])
                    for key, value in metric_sums(altered, batch['future'], batch['future_valid']).items():
                        totals[key] += value.item()
                    valid = batch['observed_valid'][:, -1, None].expand(-1,12,-1)
                    changes += float((altered-full).norm(dim=-1)[valid].sum())
                    predictions += int(valid.sum())
            rows.append(dict(model=name, case='zero_values_keep_support' if keep_support else 'zero_values_zero_support',
                status='executed', inference_device='cpu', ADE_m=totals['ade_sum']/totals['ade_count'],
                FDE_m=totals['fde_sum']/totals['fde_count'], prediction_mean_change_m=changes/predictions,
                raw_core_modified=False, mask_kept=keep_support, test_evaluated=False,
                model_training_commit=checkpoint['config']['source_commit'], checkpoint_sha256=sha256(path),
                inference_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                data_manifest_sha256=sha256(protocol['data_manifest']), subset_manifest_sha256=sha256(protocol['small_subset'])))
            print(json.dumps(rows[-1]), flush=True)
    csv_new(ROOT/'value_vs_support_controls.csv', rows)


if __name__ == '__main__':
    main()
