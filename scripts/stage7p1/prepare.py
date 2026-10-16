"""Protect historical artifacts and preregister bounded causal admission."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert base == '546c256755139edd59b1ffe09c184bc05260a1e8'
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'feat/stage7p1-agentformer-causal-admission'
paths = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
old = json.load(open('configs/stage7/protected_artifacts.json'))
for r in old['files']:
    assert sha256(r['path']) == r['sha256'], r['path']
    paths.add(r['path'])
for p in Path('experiments/stage7/cache').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
        paths.add(str(p))
json_new('configs/stage7p1/protected_artifacts.json', dict(base_commit=base,
    files=[dict(path=p, sha256=sha256(p)) for p in sorted(paths)]))
manifests = {r: dict(path=f'configs/stage5p1/manifests/{r}/B.json',
    sha256=sha256(f'configs/stage5p1/manifests/{r}/B.json')) for r in ['biwi_hotel', 'crowds_zara02']}
json_new('configs/stage7p1/admission_protocol.json', dict(base_commit=base,
    official_commit='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',
    protocol_A='HTraj-AgentFormer-A_offline_complete20-v1',
    protocol_B='HTraj-AgentFormer-Causal8x12-v1', manifests=manifests,
    context_rule='Intersection of raw actor IDs in the eight historical frames; sorted IDs; no radius or count cap',
    target_rule='Exact Stage5.1 frozen complete8/complete12 cohort; labels kept separately from context selection',
    observations=8, predictions=12, seconds_per_sample=.4, coordinate_unit='m',
    model_config='cfg/eth_ucy/hotel/hotel_agentformer_pre.yml', seed=17,
    core_or_loss_modifications_allowed=False, formal_epochs=0, optimizer_updates=0,
    real_case_selection='Per recording: first N1, first N2, first N>=3 fully labeled if available, first mixed-label, first partial-history presence, maximum N; deduplicate',
    maximum_real_cases_per_recording=6,
    raw_counterfactuals=['future_positions', 'neighbor_future_deleted', 'neighbor_future_shortened', 'future_only_actor', 'non_target_future_changed'],
    raw_counterfactual_windows='First mixed-label validation window per recording, otherwise first N>=2 validation window',
    inference_noise='K1 epsilon zero at learned prior; K20 stable SHA256(recording,start,actor,seed)-seeded epsilon; same noise for all raw variants',
    inference_interface='NaN future sentinel and zero validity; infer only; never a valid training packet',
    true_original_loss='Allowed only if ALL causal context nodes have real complete12 labels; otherwise fail closed',
    negative_probe='One mixed-label real window per recording; compare two explicitly artificial masked placeholder completions and NaN sentinel; diagnostics ONLY, no update',
    resource_probe='Maximum complete8 context across frozen train and validation supports; no truncation; original backward only if all labels available',
    numerical_tolerance=dict(atol=1e-5, rtol=1e-5),
    semantic_stop='If posterior, KL, normalization or output semantics must change for unlabeled context: NO-GO; do not implement replacement training objective',
    ETH_evaluated=False, automatic_push=False))
print('Historical artifacts protected:', len(paths), flush=True)
