"""All selected models/seeds: fixed-checkpoint CPU interventions, no retraining."""
from pathlib import Path
import csv
import json
import sys
import torch
from torch.utils.data import Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage2p5_attributions import CASES
from src.data.dataset import TrajectoryDataset
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS
from src.training.social_diagnostics import ROOT, PROTOCOL, cached_validation, evaluate, csv_new
from src.data.preprocessing import sha256


def main():
    protocol = json.loads(PROTOCOL.read_text())
    selection = json.loads((ROOT/'selection.json').read_text())
    complete = list(csv.DictReader((ROOT/'round2/summary.csv').open()))
    expected = {(name, seed) for name in ['PedBase', *selection['selected_social_models']]
                for seed in [17, 42, 2026]}
    assert {(row['model'], int(row['seed'])) for row in complete} == expected
    torch.set_num_threads(1)
    subset = json.loads(Path(protocol['large_subset']).read_text())
    data = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'validation'),
                  subset['validation']['indices'])
    batches = cached_validation(data, protocol['batch_size'])
    rows = []
    for name in selection['selected_social_models']:
        for seed in [17, 42, 2026]:
            path = Path('experiments/stage2p5/round2')/name/f'seed_{seed}/checkpoints/best.pt'
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            model = DIAGNOSTIC_MODELS[name](hidden_dim=64, max_order=3, max_neighbors=8, radius=4.).eval()
            model.load_state_dict(checkpoint['model'])
            full = evaluate(model, batches, torch.device('cpu'))
            for label, intervention in CASES.items():
                reason = ''
                if label in ('disable_residual', 'fixed_gate_0.05') and not model.fusion.residual:
                    reason = 'model_has_no_residual'
                if label == 'disable_normalization' and not model.fusion.normalization:
                    reason = 'model_has_no_normalization'
                if name == 'Dynamic-GNN-RN' and label in ('only_order1', 'orders1_2', 'orders1_2_3'):
                    reason = 'one_mixed_dynamic_channel_is_not_social_order'
                result = evaluate(model, batches, torch.device('cpu'), intervention if not reason else {}, full['predictions'])
                row = dict(model=name, seed=seed, phase='round2', checkpoint='best',
                    best_epoch=checkpoint['epoch'], case=label, applicability='active' if not reason else 'not_applicable_noop',
                    noop_reason=reason, status='executed', inference_device='cpu',
                    ADE_m=result['ADE_m'], FDE_m=result['FDE_m'],
                    delta_ADE_vs_full=result['ADE_m']-full['ADE_m'],
                    delta_FDE_vs_full=result['FDE_m']-full['FDE_m'], **result['diagnostics'],
                    source_commit=checkpoint['config']['source_commit'],
                    inference_runner_sha256=sha256(Path(__file__)),
                    data_manifest_sha256=sha256(protocol['data_manifest']),
                    subset_manifest_sha256=sha256(protocol['large_subset']),
                    checkpoint_sha256=sha256(path), test_evaluated=False,
                    interpretation='same best checkpoint CPU inference intervention; not retrained ablation')
                rows.append(row)
                print(json.dumps({key: row[key] for key in ['model', 'seed', 'case', 'ADE_m', 'delta_ADE_vs_full']}), flush=True)
            del model
    csv_new(ROOT/'final_candidate_interventions.csv', rows)


if __name__ == '__main__':
    main()
