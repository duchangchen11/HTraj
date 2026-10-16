"""Protect history; freeze two ego-capacity runs and conditional stage-two gate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess
from src.data.preprocessing import sha256
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='04d95937282e8a84c1f497c6c1170f81a5a753ea'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage5p3/protected_artifacts.json'))['files']}
for rec in ['biwi_hotel','crowds_zara02']:
    paths|={r['path'] for r in read_csv(f'results/tables/stage5p3/{rec}/checkpoint_inventory.csv')}
json_new('configs/stage5p4/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
plan=json.load(open('configs/stage5p3/protocol.json'))
for key in ['capacity_control','stopping','primary_attribution','checkpoint_saving']:plan.pop(key,None)
plan.update(stage='5.4-phase1',base_commit=base,branch='feat/stage5p4-capacity-attribution',model='SocialVAE-EgoCapacity',
    initialization='fresh seed17 per run; original base allocated first, then default PyTorch Linear MLP initialization; no checkpoint import; extra allocation consumes RNG, so later stochastic stream not promised identical to old architectures',
    reference=dict(Full='results/tables/stage5p2',NoSocial='results/tables/stage5p3',models_frozen=True),
    ego_capacity=dict(dimensions=[256,1024,256],activation='ReLU6',placement='h_x + MLP(h_x) before original rnn_fy_init; posterior b unchanged',
        weight_masks=False,initialization='unchanged torch.nn.Linear default, no zero-output branch',
        added_parameters=525568,nominal_parameters=2669570,structural_effective_upper_bound=1985474,
        Full_structural_effective_upper_bound=2045698,relative_upper_bound_difference=-60224/2045698,
        caveat='similar scalar counts do not match function class; extra residual ego path changes inductive bias; empirical nonzero gradients audited separately'),
    checkpoint_policy='immutable checkpoint whenever best K1 improves, at epoch1/every10/epoch100; metrics/target errors/parameter states every epoch; no old file overwritten',
    checkpoints='best improvements + periodic10 + last100; strict restore at1/every10 and selected best',
    phase1_gate=dict(primary='best K1 latent-prior ADE, paired FDE, exact same target cohort',
        weightings=['target-window','equal-person-ID'],FDE_substantial_regression_relative=.01,
        capacity_sufficient='Ego ADE <= Full ADE and Ego FDE <= Full FDE*1.01 in both recordings and both weightings; stop without placebo',
        Full_meaningful_ADE_gain_relative=.01,
        continue_to_placebo='Full relative ADE gain (Ego-Full)/Ego >=1% and Full FDE <= Ego FDE*1.01 in both recordings and both weightings, plus capacity and isolation tests passed',
        otherwise='conditional/conflicting/small differences: stop without placebo; no alternative seeds or architecture attempts',
        thresholds='preregistered development decision criteria, not statistical significance or formal noninferiority tests'),
    phase2_authorization=dict(max_runs=2,only_if_phase1_gate_passes=True,separate_protocol_commit_required=True,
        pseudo_context='one deterministic target-history-only construction, frozen before any phase2 training; no real neighbor count/ID/position/velocity; no rule search',
        Full_architecture_loss_preserved=True,additional_seeds=False),
    max_total_formal_runs=4,phase1_formal_runs=2,new_paper_method=False,
    no_extra_runs_or_extension=True)
json_new('configs/stage5p4/ego_protocol.json',plan)
print('Protected files:',len(paths))
