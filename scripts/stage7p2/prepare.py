"""Freeze benchmark transition before any standard-protocol smoke."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert base == '81a2e239e8ee79c07d19e319401c0f96c0b5e7ec'
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'feat/stage7p2-standard-benchmark-alignment'
old = json.load(open('configs/stage7p1/protected_artifacts.json'))
protected = {r['path']: r['sha256'] for r in old['files']}
new = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
new |= {str(p) for p in Path('experiments/stage7p1/cache').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for p in sorted(new - protected.keys()):
    protected[p] = sha256(p)
json_new('configs/stage7p2/protected_artifacts.json', dict(base_commit=base,
    inherited_expected_SHA_verified_at_delivery=True,
    files=[dict(path=p, sha256=v) for p, v in sorted(protected.items())]))
json_new('configs/stage7p2/benchmark_protocol.json', dict(base_commit=base,
    protocol='HTraj-AgentFormer-OfficialETHUCY8x12-K20-v1',
    user_authorized_protocol_transition=True,
    previous_strict_history_neighbor_protocol='Preserved, not imposed on official AgentFormer and never ranked together',
    ETH_rule='Replaced by official five-fold split; ETH test scores never used for tuning; no ETH model evaluation in this preparation stage',
    official_repository='https://github.com/Khrylx/AgentFormer',
    official_commit='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',
    expected_recordings=['biwi_eth','biwi_hotel','crowds_zara01','crowds_zara02','crowds_zara03','students001','students003','uni_examples'],
    folds=['eth','hotel','univ','zara1','zara2'], obs_len=8, pred_len=12,
    dt_seconds=.4, coordinate_unit='m', sample_k=20,
    context_rule='Unmodified official preprocessor, min_past_frames=8 and min_future_frames=12',
    metrics='Per target-window minADE@20 and minFDE@20 independently; target-weighted within fold; unweighted five-fold mean',
    inference='CVAE prior or full CVAE+DLow infer; never reconstruction; samples only',
    model_core_changes=False, original_losses=True,
    smoke_seed=17, smoke_folds=['hotel','zara2'],
    smoke_case_rule='First single, first multiple, largest complete20 context from official training pool, deduplicated; never truncate context',
    smoke_budget='At most six scene cases and one Adam update per stage (CVAE and DLow) per case; no epoch',
    maximum_smoke_optimizer_updates=12,
    highest_density_OOM='Persist failure; no deleting actors or retuning core; declare resource gate failed unless a separately frozen non-truncating execution plan is validated',
    pretrained_download_attempts='Author Drive link and one alternate download endpoint; Baidu landing metadata only, no credential requests',
    strict_checkpoint_loading=True, formal_epochs=0, automatic_push=False))
print('Protected', len(protected), 'historical files; standard protocol transition frozen')
