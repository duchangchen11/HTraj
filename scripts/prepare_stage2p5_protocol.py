from pathlib import Path
import json
import numpy as np
from src.data.preprocessing import sha256, write_json_new

root = Path('configs/stage2p5')
small = json.loads(Path('configs/smoke_subsets/stage2_smoke_v1.json').read_text())
manifest = json.loads(Path('configs/data_manifests/stage2_v1/fold_ETH.json').read_text())
existing = set(small['train']['indices'])
remaining = sorted(set(range(len(manifest['splits']['train'])))-existing)
extra = np.random.default_rng(1725).choice(remaining, 1024-len(existing), replace=False).tolist()
large_indices = sorted([*existing, *extra])
large = dict(seed_for_subset=1725, data_manifest_sha256=small['data_manifest_sha256'], test_used=False,
             train=dict(indices=large_indices, sample_ids=[manifest['splits']['train'][i]['sample_id'] for i in large_indices]),
             validation=small['validation'])
write_json_new(root/'subset_1024.json', large)
protocol = dict(schema_version=1, held_out_scene='ETH', test_evaluation=False,
    data_manifest='configs/data_manifests/stage2_v1/fold_ETH.json', raw_root='data/raw/eth_ucy',
    small_subset='configs/smoke_subsets/stage2_smoke_v1.json', large_subset='configs/stage2p5/subset_1024.json',
    epochs=30, batch_size=8, learning_rate=.001, gradient_clip=1., cpu_threads=1,
    hidden_dim=64, prediction_steps=12, max_order=3, max_neighbors=8, radius=4., initial_residual_gate=.05,
    loss='masked_per_track_mean_L2', metrics='K1 ADE/FDE meters; fixed final-horizon FDE mask',
    round1_models=['PedBase','FirstOrder-GNN','Dynamic-GNN','TOHI-A0','TOHI-A-R','TOHI-A-N','TOHI-A-RN',
                   'FirstOrder-GNN-RN','Dynamic-GNN-RN','PedBase-Wide'], round1_seeds=[17],
    round2_seeds=[17,42,2026], maximum_selected_social_models=2,
    best_epoch_rule='minimum validation ADE, earliest epoch on exact tie; report paired FDE of that epoch',
    first_round_pause_rule='all social models best ADE > 1.05*PedBase best ADE and no FDE improvement',
    selection_rule='lowest best-ADE social; add lowest TOHI if different and within 5% of best social, else second social',
    group_rule='cutoff-visible people; average incoming degree over eligible contiguous observation steps: 0,(0,1],(1,3],>3; order support at cutoff',
    convergence_rule='relative train loss decrease in final five epochs < 2%; diagnostic only, not proof of optimum',
    shuffle_rule='resample same incoming count within same radius-valid candidate set, no self/cross-scene, fixed seed17025',
    attribution_rule='best checkpoint inference interventions; not retrained order ablations; include TOHI-A-RN reference',
    graph_core_changes=False, additional_loss=False, augmentation=False, decoder_changes=False)
write_json_new(root/'protocol.json', protocol)
print('Fixed protocol and nested 1024-window subset created; validation IDs unchanged.')
