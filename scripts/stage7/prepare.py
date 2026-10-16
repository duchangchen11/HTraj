"""Seal previous results; no model training or ETH evaluation in Stage7."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess
from src.data.preprocessing import sha256
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='e7706706e7ea3d1e4146a16821f862a194ef7bc0'
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/stage7-paper-baseline-decision'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage6p1/protected_artifacts.json'))['files']}
for index in Path('results/tables/stage6p1/runs').glob('*/*/checkpoint_inventory.csv'):
 paths|={r['path'] for r in read_csv(index)}
paths.add(json.load(open('results/tables/stage6p1/analysis/execution.json'))['trajectory_packet'])
json_new('configs/stage7/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
json_new('configs/stage7/admission_protocol.json',dict(base_commit=base,stage=7,new_training_runs=0,optimizer_updates=0,
 ETH_evaluated=False,max_new_candidates=2,candidates=['AgentFormer','PECNet'],
 synthetic_agents=[1,2,5,12],observations=8,predictions=12,dt_seconds=.4,
 permutation='Synchronized agent data/mask permutation; deterministic latent control and identical RNG, then inverse outputs',
 tolerances=dict(atol=1e-5,rtol=1e-5),future_isolation='Same history and explicit fixed latent/noise, change or poison all future labels; inference unchanged',
 gradients='Finite original training objective and backward only, no optimizer update; all tested gradients finite',
 checkpoint='Strict random-state save/load roundtrip, not official pretrained reproduction',
 limits='Bounded source/protocol/synthetic checks only; no real scene ADE or fivefold training; do not repair architecture to force admission',
 closures=['TOHI graph modules','G0/G1 prototypes','SocialVAE social-attention modification','capacity and Placebo social attribution','Stage6p1 history-stratified sampling'],
 automatic_push=False))
print('Protected prior artifacts:',len(paths))
