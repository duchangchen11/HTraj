from pathlib import Path
import json
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
root=Path('results/tables/stage5p4')
decision=json.load(open(root/'phase1_decision.json'));assert decision['phase2_allowed']
plan=json.load(open('configs/stage5p4/ego_protocol.json'))
plan.update(stage='5.4-phase2',model='SocialVAE-PlaceboSocial',phase1_gate_decision_sha256=sha256(root/'phase1_decision.json'),
 phase2_formal_runs=2,initialization='fresh seed17; exact Full architecture/state schema/default initialization; no checkpoint import',
 synthetic_context=dict(rule='p_j(t)=x(t)+r_j+0.2*s_j*tanh((x(t)-x(0))/1m), componentwise',
  offsets_m=[[.8,0],[-.8,0],[0,.8],[0,-.8]],signs=[1,-1,1,-1],count=4,
  velocity='successive synthetic position differences, per-step displacement as official API',
  acceleration='successive synthetic displacement differences; boundary zero placeholders ignored by official initialization',
  future='all twelve future neighbor slots are 1e9 padding, including training posterior',
  raw_neighbor_information_used=False,causality='each synthetic frame uses target prefix only',
  distribution_caveat='four always-available near neighbors vs variable real count/entry/missing velocity; fixed cardinal directions and bounded drift change inductive bias; initial relative offsets are constant, not equal function-class capacity',
  rule_search=False),
 phase2_interpretation=dict(primary='K1 at best K1 checkpoint; paired FDE; both recordings and both weightings',
  Full_meaningful_relative_ADE_gain=.01,FDE_substantial_regression_relative=.01,
  sufficient_target_only='Placebo ADE <= Full ADE and Placebo FDE <= Full*1.01 across all four cells',
  provisional_content_support='Full relative ADE gain >=1% and Full FDE <= Placebo*1.01 across all four cells; conditional on input-distribution and single-seed caveats',
  otherwise='conditional/conflicting evidence; no more social modules or runs this stage',K20='separate descriptive outcome, never a replacement for K1 attribution'),
 execution_schedule='two independent processes may overlap on same RTX3080; RNG/model/optimizer separate; wall times not exclusive-device throughput')
plan.pop('ego_capacity',None)
json_new('configs/stage5p4/placebo_protocol.json',plan)
(root/'placebo_frozen_protocol.json').write_bytes(Path('configs/stage5p4/placebo_protocol.json').read_bytes())
s=Path('src/socialvae_capacity/training.py').read_text().replace('from .model import','from .placebo import').replace('EgoCapacity checkpoint','PlaceboSocial checkpoint')
Path('src/socialvae_capacity/placebo_training.py').write_text(s)
s=Path('scripts/stage5p4/train.py').read_text().replace('from src.socialvae_capacity.model import','from src.socialvae_capacity.placebo import').replace('from src.socialvae_capacity.training import','from src.socialvae_capacity.placebo_training import').replace('ego_protocol.json','placebo_protocol.json').replace('stage5p4/ego_capacity','stage5p4/placebo_social').replace('stage5p4/checkpoints/ego_capacity','stage5p4/checkpoints/placebo_social').replace('scripts/stage5p4/train.py','scripts/stage5p4/train_placebo.py').replace('src/socialvae_capacity/model.py','src/socialvae_capacity/placebo.py').replace('src/socialvae_capacity/training.py','src/socialvae_capacity/placebo_training.py')
Path('scripts/stage5p4/train_placebo.py').write_text(s)
s=Path('scripts/stage5p4/audit_capacity.py').read_text().replace('from src.socialvae_capacity.model import SocialVAEEgoCapacity','from src.socialvae_capacity.placebo import SocialVAEPlacebo').replace('ego_protocol.json','placebo_protocol.json').replace("[('EgoCapacity',SocialVAEEgoCapacity)]","[('PlaceboSocial',SocialVAEPlacebo)]").replace('capacity_by_parameter.csv','placebo_capacity_by_parameter.csv').replace('capacity_audit.json','placebo_capacity_audit.json')
Path('scripts/stage5p4/audit_placebo_capacity.py').write_text(s)
