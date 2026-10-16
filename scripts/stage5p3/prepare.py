"""Protect the complete historical inventory and freeze this stage's limits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess
from src.data.preprocessing import sha256
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='bc62cf89570c584f297f71e368107015c5498f0c'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage5p2/protected_artifacts.json'))['files']}
for rec in ['biwi_hotel','crowds_zara02']:
    paths|={r['path'] for r in read_csv(f'results/tables/stage5p2/{rec}/checkpoint_inventory.csv')}
json_new('configs/stage5p3/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
plan=json.load(open('configs/stage5p2/protocol.json'))
plan.update(stage='5.3',base_commit=base,branch='feat/stage5p3-social-attribution',model='SocialVAE-NoSocial',
    initialization='fresh seed17 per run, same original parameter allocation/order as Full; no trained weight import',
    reference=dict(stage='5.2',commit=base,training_code_commit='eb8d494bb1373462fcd07da46abeb5e7bb2533f7',
        results_root='results/tables/stage5p2',models_frozen=True),
    new_network=False,new_graph_module=False,
    primary_attribution='Full minus NoSocial same-target K1 ADE; paired FDE, K20 evaluated separately',
    stopping=dict(additional_seeds=False,extra_training_after_two_runs=False,
        useful_ADE_relative_gain=.01,meaningful_FDE_regression_relative=.01,
        rule='Full ADE improvement >=1% and FDE regression <=1% in both recordings and both window/person weighting: worth followup, capacity confound remains. One recording only: conditional. Neither recording: stop complex SocialVAE social-module development. Report exact differences for smaller effects/conflicting metrics; no posthoc new threshold.'),
    capacity_control=dict(name='Ego-Capacity-Control',status='design_only_not_trained_in_stage5p3',
        decision='do not append training in this stage; complete strict NoSocial first, report unresolved capacity confound; next-stage authorization needed for separate control',
        proposal='target-motion GRU final state -> residual ego MLP H256->1142->256 to initialize decoder; fixed 310 first-layer weights masked to zero; 585792 structurally active extra scalars (2045698 total upper bound), distinct architecture/inductive bias',
        proposed_training_budget=dict(seed=17,epochs=100,batch_size=128,learning_rate=.0005,optimizer='Adam',
            validation_recordings=['crowds_zara02','biwi_hotel'],selection='same K1 ADE rule'),
        limitation='additional ego path/depth/initialization changes inductive bias; nominal matching does not guarantee function-class matching'),
    diagnostics=dict(checkpoint_restore_epochs=[1,10,20,30,40,50,60,70,80,90,100],
        epoch_gradient_tracking='per parameter scalar count with nonzero gradient, maxabs, ever_nonzero; structural dead masks audited',
        capacity_definition='separate nominal, module-forward-used, graph-gradient-connected, observed-nonzero, structural scalar upper bound; no claim of equal effective capacity'),
    checkpoint_saving='immutable every epoch; best/last references; unused weights retained, optimizer has no state for disconnected tensors')
json_new('configs/stage5p3/protocol.json',plan)
print('Protected files:',len(paths))
