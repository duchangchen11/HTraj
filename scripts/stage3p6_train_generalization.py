"""Post-training endpoint train/val diagnostics, no optimization or selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import subprocess
import torch
from torch.utils.data import DataLoader, Subset
from src.data.benchmark_audit import BenchmarkDataset
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256, write_json_new
from src.evaluation.metrics import metric_sums
from src.models.ped_base import PedBase
from src.models.frozen_ego import CVResidual
from src.models.relation_routing import HistoricalCV
from src.training.social_diagnostics import csv_new


@torch.no_grad()
def metrics(model, dataset):
    model.eval(); totals = dict(ade_sum=0., ade_count=0, fde_sum=0., fde_count=0)
    for b in DataLoader(dataset, batch_size=8, collate_fn=collate_trajectories):
        prediction = model(b['observed'], b['observed_valid'], b['timestamps'])
        for key, value in metric_sums(prediction, b['future'], b['future_valid']).items():
            totals[key] += value.item()
    return dict(ADE_m=totals['ade_sum']/totals['ade_count'], FDE_m=totals['fde_sum']/totals['fde_count'], targets=totals['ade_count'])


def main():
    root = Path('results/tables/stage3p6')
    assert (root/'round3_decision.json').exists(), 'Only after all primary runs finish'
    torch.set_num_threads(1); p = json.loads(Path('configs/stage3p6/protocol.json').read_text()); rows = []
    for fold in p['folds']:
        data = {split: Subset(BenchmarkDataset(p['raw_root'], fold['manifest'], split, p['protocol_name']), fold['indices'][split]) for split in ('train', 'validation')}
        cv = {split: metrics(HistoricalCV(), d) for split, d in data.items()}
        for split, r in cv.items():
            rows.append(dict(recording=fold['recording'], model='CV', schedule='analytical', endpoint='analytical', split=split, **r, checkpoint_sha256=None, device='CPU'))
        for label in p['neural_ego_runs']:
            name, schedule = label.split('/')
            directory = root/fold['recording']/'ego'/(name+'__'+schedule)/'seed_17'
            s = json.loads((directory/'summary.json').read_text())
            for endpoint in ('best', 'last'):
                checkpoint = s[endpoint+'_checkpoint_path']; assert sha256(checkpoint) == s[endpoint+'_checkpoint_sha256']
                model = (PedBase if name == 'PedBase' else CVResidual)()
                model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
                for split, d in data.items():
                    r = metrics(model, d)
                    if split == 'validation':
                        assert abs(r['ADE_m']-s[endpoint+'_ADE_m']) < 1e-6
                    rows.append(dict(recording=fold['recording'], model=name, schedule=schedule, endpoint=endpoint, split=split, **r,
                        delta_ADE_vs_same_split_CV_m=r['ADE_m']-cv[split]['ADE_m'], checkpoint_sha256=s[endpoint+'_checkpoint_sha256'], device='CPU'))
    csv_new(root/'endpoint_train_validation_generalization.csv', rows)
    write_json_new(root/'train_generalization_provenance.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), script_sha256=sha256(__file__),
        purpose='post-training same endpoint train/val inference; different from online minibatch training loss', used_for_selection=False, new_training=False, ETH_evaluated=False, rows=len(rows)))
    print(json.dumps(dict(rows=len(rows), new_training=False, ETH_evaluated=False)))


if __name__ == '__main__':
    main()
