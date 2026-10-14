"""Run CPU inference interventions while GPU training continues independently."""
from pathlib import Path
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
    torch.set_num_threads(1)
    subset = json.loads(Path(protocol['small_subset']).read_text())
    data = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'validation'), subset['validation']['indices'])
    batches = cached_validation(data, protocol['batch_size'])
    rows = []
    for name in ['TOHI-A0','TOHI-A-R','TOHI-A-N','TOHI-A-RN']:
        path = Path('experiments/stage2p5/round1')/name/'seed_17/checkpoints/best.pt'
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        model = DIAGNOSTIC_MODELS[name](hidden_dim=64, max_order=3, max_neighbors=8, radius=4.).eval()
        model.load_state_dict(checkpoint['model'])
        full = evaluate(model, batches, torch.device('cpu'))
        for label, intervention in CASES.items():
            active = not ((label in ('disable_residual','fixed_gate_0.05') and not model.fusion.residual)
                          or (label=='disable_normalization' and not model.fusion.normalization))
            result = evaluate(model, batches, torch.device('cpu'), intervention if active else {}, full['predictions'])
            row = dict(model=name, seed=17, phase='round1', checkpoint='best', best_epoch=checkpoint['epoch'], case=label,
                applicability='active' if active else 'not_applicable_noop', status='executed', inference_device='cpu',
                ADE_m=result['ADE_m'], FDE_m=result['FDE_m'], delta_ADE_vs_full=result['ADE_m']-full['ADE_m'],
                delta_FDE_vs_full=result['FDE_m']-full['FDE_m'], **result['diagnostics'],
                source_commit=checkpoint['config']['source_commit'], data_manifest_sha256=sha256(protocol['data_manifest']),
                subset_manifest_sha256=sha256(protocol['small_subset']), checkpoint_sha256=sha256(path), test_evaluated=False,
                interpretation='same best checkpoint CPU inference intervention; not retrained ablation')
            rows.append(row)
            print(json.dumps({key: row[key] for key in ['model','case','ADE_m','FDE_m','delta_ADE_vs_full','prediction_intervention_mean_L2_m']}), flush=True)
        del model
    csv_new(ROOT/'mechanism_interventions.csv', rows)


if __name__ == '__main__':
    main()
