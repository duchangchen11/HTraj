"""Describe observation length inside the FIXED groups; do not redefine them."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TrajectoryDataset
from src.training.social_diagnostics import PROTOCOL, ROOT, cached_validation, csv_new
from src.data.preprocessing import sha256

torch.set_num_threads(1)
protocol = json.loads(PROTOCOL.read_text())
subset = json.loads(Path(protocol['small_subset']).read_text())
data = Subset(TrajectoryDataset(protocol['raw_root'], protocol['data_manifest'], 'validation'), subset['validation']['indices'])
batches = cached_validation(data, protocol['batch_size'])
lengths = {}
for batch in batches:
    history_points = batch['observed_valid'].sum(1)
    # Future validity defines only which population has an ADE, never grouping.
    evaluable = batch['future_valid'].any(1)
    for group, mask in batch['groups'].items():
        lengths.setdefault(group, []).extend(history_points[mask & evaluable].tolist())
rows = []
for group, values in lengths.items():
    array = np.asarray(values)
    rows.append(dict(group=group, ADE_evaluable_targets=len(array), mean_observed_points=float(array.mean()) if len(array) else None,
                     one_observation_fraction=float((array==1).mean()) if len(array) else None,
                     full_eight_observations_fraction=float((array==8).mean()) if len(array) else None,
                     group_definition_changed=False, test_evaluated=False,
                     data_manifest_sha256=sha256(protocol['data_manifest']), subset_manifest_sha256=sha256(protocol['small_subset'])))
csv_new(ROOT/'history_group_composition.csv', rows)
print(json.dumps(rows, indent=2))
