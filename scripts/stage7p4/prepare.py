"""Freeze original inputs, one reference-only DLow override, and all stopping rules."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import shutil
import subprocess
from src.agentformer_feasibility.common import ROOT,SOURCE,OUT,CACHE,digest
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import read_csv,csv_new

base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='d87551873338183bbeb178bec3e321f54c732256'
old=json.loads(Path('configs/stage7p3/protected_artifacts.json').read_text())
protected={r['path']:r['sha256'] for r in old['files']}
new=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
# Large Stage7.3 prediction payloads are protected through their existing compressed SHA indices.
new |= {str(p) for p in Path('experiments/stage7p3/cache').rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'predictions' not in p.parts and not p.is_symlink()}
for path in sorted(new-protected.keys()):protected[path]=digest(path)
indices=[f'results/tables/stage7p3/{fold}_prediction_file_SHA256.csv.gz' for fold in ['eth','hotel','univ','zara1','zara2']]
json_new('configs/stage7p4/protected_artifacts.json',dict(base_commit=base,
    files=[dict(path=k,sha256=v) for k,v in sorted(protected.items())],
    prediction_SHA_indices=[dict(path=p,sha256=digest(p)) for p in indices]))
for name in ['hotel_agentformer_pre','hotel_agentformer']:
    source=SOURCE/f'cfg/eth_ucy/hotel/{name}.yml'
    text=source.read_text()
    if name=='hotel_agentformer':
        assert 'pred_epoch: 50' in text;text=text.replace('pred_epoch: 50','pred_epoch: 1')
    with Path(f'configs/stage7p4/training/{name}.yml').open('x') as f:f.write(text)
Path('configs/stage7p4/training/.gitattributes').write_text('# Keep original upstream YAML formatting.\n*.yml -whitespace\n')
Path('configs/stage7p4/training/LICENSE').write_bytes((SOURCE/'LICENSE').read_bytes())
windows=read_csv('results/tables/stage7p2/official_window_inventory.csv.gz')
train=[r for r in windows if r['fold']=='hotel' and r['sequence'].endswith('_train')]
val=[r for r in windows if r['fold'] in ['hotel','zara2'] and r['sequence'].endswith('_val')]
csv_new('configs/stage7p4/train_windows.csv.gz',train)
csv_new('configs/stage7p4/validation_windows.csv.gz',val)
parent=json.loads(Path('configs/stage7p3/evaluation_protocol.json').read_text())
pairs=[r for r in parent['pairs'] if r['fold'] in ['hotel','zara2']]
inputs=parent['official_sources']+parent['data_files']
for p in pairs:
    for kind in ['CVAE','DLow']:
        path=str(Path(p[kind+'_checkpoint']).relative_to(ROOT))
        inputs.append(dict(path=path,sha256=p[kind+'_sha256']))
for path in [*Path('configs/stage7p4/training').glob('*'),Path('configs/stage7p4/train_windows.csv.gz'),Path('configs/stage7p4/validation_windows.csv.gz')]:
    if path.is_file():inputs.append(dict(path=str(path),sha256=digest(path)))
execution=[*Path('scripts/stage7p4').glob('*.py'),*Path('src/agentformer_feasibility').glob('*.py'),
    Path('tests/test_agentformer_feasibility.py'),Path('scripts/stage7p1/runtime.py'),Path('src/agentformer_benchmark/official.py')]
fold_counts={}
for fold in ['eth','hotel','univ','zara1','zara2']:
    t=[r for r in windows if r['fold']==fold and r['sequence'].endswith('_train')]
    v=[r for r in windows if r['fold']==fold and r['sequence'].endswith('_val')]
    fold_counts[fold]=dict(train_windows=len(t),train_targets=sum(int(r['people']) for r in t),
        train_max_context=max(int(r['people']) for r in t),val_windows=len(v),val_targets=sum(int(r['people']) for r in v))
json_new('configs/stage7p4/protocol.json',dict(base_commit=base,official_commit=parent['official_commit'],
    name='Stage7p4-Official-AgentFormer-FullEpoch-and-Validation-v1',test_evaluation_forbidden=True,
    protocol='Official offline complete20; 8 history, 12 future, .4 seconds, metres',
    model_changes=False,loss_changes=False,split_changes=False,validation_folds=['hotel','zara2'],
    validation_rule='Original official val sequences, NOT the held-out test recording; original author pairs; mean=True K20',
    val_vs_test_comparison=False,CV_budget='K1 only; never interpreted as equal-budget evidence against K20',
    validation_precision='Native float32 trajectories with float64 metric arithmetic; no 3-decimal text rounding',
    history_speed='mean of seven adjacent distance/.4, thresholds .05/.4/1.8 m/s',
    speed_groups=['stationary','low','normal','fast'],motion_groups=['stationary','moving'],
    density_groups=['none','1-3','4-9','10+'],
    density_definition='Other full8-history-visible agents within2m at last history frame; future existence not consulted; diagnostic only',
    context_count_separate='Complete20 model context count is descriptive only, never a historical group definition',
    timestep_errors='First DLow head, whole-trajectory ADE-best head and FDE-best head; oracle selection only for post-hoc validation analysis',
    endpoint_dispersion='RMS radius around mean endpoint and mean of all190 head-pair distances; not a selector',
    train_fold='hotel',seed=1,optimizer='Original torch.optim.Adam(model.parameters(),lr=cfg.lr), defaults preserved',
    learning_rate=.0001,CVAE_scheduler=dict(type='StepLR',step_size=10,gamma=.5),
    DLow_scheduler=dict(type='StepLR',step_size=5,gamma=.5),
    training='Original train.py train(0) AST body, original shuffle over ALL candidate slots incl invalid; one update per valid window',
    CVAE_full_epochs=1,DLow_full_epochs=1,original_configured_epochs=dict(CVAE=100,DLow=50),
    DLow_only_override='pred_epoch 50->1; same pred_cfg name in NEW workspace resolves this stage CVAE epoch1 checkpoint',
    author_pretrained_initialization=False,smoke_initialization=False,
    optimized_CVAE_in_DLow=False,base_gradients='Original unregistered CVAE remains requires_grad=True/eval; gradients may accumulate, optimizer contains only DLow params',
    instrumentation='Original losses/backward/Adam untouched; observe scalar losses, foreach gradient and parameter norms, synchronize step-end; overhead separately recorded',
    steady_timing='Exclude first10 optimizer updates only for steady mean; full epoch still includes ALL updates/windows',
    resume='Strict model/Adam/scheduler/generator/RNG restoration; ONE shadow next-epoch first-valid update per stage; excluded from epoch budget, never saved as formal model',
    engineering_probe_updates_max=2,automatic_retry=False,automatic_extension=False,
    stop_rules='Any OOM, nonfinite loss/gradient/parameter, data mismatch, restore failure => stop pipeline; no deleting actors, smaller K, altered LR or architecture',
    budget_estimation=dict(CVAE_epochs=100,DLow_epochs=50,
        fold_scaling='Fit nonnegative [1,N,N^2] timing on measured HOTEL steps; official training-context histograms; ETH original CVAE cap32 and extra architecture uncertainty declared',
        sensitivity_factors=[.7,1.5],validation_overhead='Assume every10 CVAE/every5 DLow epochs, 0.5-2x measured K20 validation time; separate from official train.py which has no validation loop',
        checkpoint_overhead='Official save every10 CVAE/every5 DLow; preserve all, no deletion',
        engineering_time_limits=dict(HOTEL_upper_hours=72,fivefold_upper_hours=336),
        time_limit_provenance='Predeclared planning guard, not a user-authorized formal training budget',
        storage_gate='NO-GO for immediate full five-fold launch when current free disk cannot retain official checkpoints plus128MiB reserve; still report epoch feasibility separately'),
    stage_minimum_free_bytes=400*1024**2,author_pairs=pairs,fold_counts=fold_counts,
    frozen_inputs=inputs,execution_source_SHA256={str(p):digest(p) for p in execution},automatic_push=False))
free=shutil.disk_usage(ROOT).free
json_new(OUT/'storage_preflight.json',dict(free_bytes=free,minimum_free_bytes=400*1024**2,
    rough_stage_budget_bytes=280*1024**2,stage_fits=free>=400*1024**2,
    old_results_will_not_be_deleted=True,full_fivefold_checkpoint_space_must_be_estimated_from_actual_saved_files=True))
assert free>=400*1024**2
print('Protected',len(protected),'historical files plus Stage7.3 file-SHA indices; HOTEL train',len(train),'validation',len(val),'frozen')
