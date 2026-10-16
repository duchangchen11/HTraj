"""Freeze previous stages and a descriptive, non-training diagnostic contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess
from src.data.preprocessing import sha256
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='6d6c741520ce09a5a72e375ffa7a17d4206b7d97'
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/stage6-research-reassessment'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage5p5/protected_artifacts.json'))['files']}
for p in Path('results/tables/stage5p5/runs').glob('*/*/*/checkpoint_inventory.csv'):
 paths|={r['path'] for r in read_csv(p)}
json_new('configs/stage6/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
old=json.load(open('configs/stage5p5/protocol.json'))
json_new('configs/stage6/diagnostic_protocol.json',dict(
 base_commit=base,protocol=old['protocol'],manifests=old['manifests'],data_root=old['data_root'],
 models=['Full','PlaceboSocial'],seeds=[17,42,2026],selection='Reuse each Stage5.5 indexed best checkpoint, never reselect',
 scope='Post-hoc descriptive diagnosis on exposed development recordings, no new training or ETH evaluation',
 dt_seconds=.4,obs_len=8,pred_len=12,evaluation_batch_size=128,training=False,
 predictions='Recompute deterministic K1 from strict-loaded frozen weights; check every saved ADE/FDE. No posterior or GT model input.',
 historical_groups='Stage5.2 frozen history groups, copied by stable target ID; new orientation bins are historical only',
 thresholds=dict(history_stationary_speed_mps=.05,reliable_heading_net_displacement_m=.2,
  reliable_heading_speed_mps=.2,heading_bins=8,turn_error_deg=30.,
  sustained_future_speed_mps=.4,sustained_late_to_early_ratio=.75,
  predicted_deceleration_ratio=.7,motion_flag_mps=.2,future_stationary_speed_mps=.05,
  displacement_ratio_low=.5,displacement_ratio_high=2.,displacement_comparison_min_m=.5),
 warning='Future-defined error tags are diagnosis outcomes only; never input features, training groups or gates. Tags overlap. Thresholds are descriptive, not registered statistical confirmation.',
 inference_match_tolerance_m=2e-6,
 public_models=dict(SGCN='5153702d8ae32a2b0586b093d4e094aee08b0f9b',additional_maximum=1),
 public_function_test='Only synthetic forward/checkpoint/ordering checks. No official scene scores, no optimizer steps.',
 raw_packets='Keep exact trajectory arrays in ignored experiments/stage6/cache; commit only derived diagnostic scalars and hashes',
 ETH_evaluated=False,automatic_push=False))
print('Protected prior artifacts:',len(paths))
