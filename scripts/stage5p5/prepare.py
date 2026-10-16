"""Freeze all eight runs, sampling and clustered descriptive inference upfront."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import read_csv
from src.data.preprocessing import sha256
base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert base=='e8c69194f228a19d4e7410a9825adbeedb2b2b42'
paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
paths|={r['path'] for r in json.load(open('configs/stage5p4/protected_artifacts.json'))['files']}
for kind in ['ego_capacity','placebo_social']:
 for rec in ['biwi_hotel','crowds_zara02']:
  paths|={r['path'] for r in read_csv(f'results/tables/stage5p4/{kind}/{rec}/checkpoint_inventory.csv')}
json_new('configs/stage5p5/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
old=json.load(open('configs/stage5p4/placebo_protocol.json'))
keys=['protocol','data_root','manifests','hidden_dim','ob_radius_m','batch_size','epochs','optimizer','learning_rate','scheduler','gradient_clip','weight_decay','augmentation','loss','model_selection','official_commit','official_source_sha256','history_groups','embargo','evaluation','synthetic_context']
plan={k:old[k] for k in keys}
plan.update(stage='5.5',branch='feat/stage5p5-social-stability',base_commit=base,
 models=['Full','PlaceboSocial'],new_seeds=[42,2026],reference_seed=17,
 formal_runs=[dict(seed=s,model=m,recording=r) for s in [42,2026] for r in ['crowds_zara02','biwi_hotel'] for m in ['Full','PlaceboSocial']],
 max_new_runs=8,pretrained_initialization=False,
 initialization='seed_all(run_seed) before frozen model constructor; verify all shared tensors and post-construction RNG exactly equal for Full/Placebo per seed before training',
 training_order='numpy default_rng(run_seed+epoch).permutation(all train targets), retain final partial batch, exactly one complete pass',
 checkpoint_policy='epoch1/25/50/75/100 immutable periodic snapshots plus selected best serialized in RAM on each improvement and written once at completion; last=epoch100; no old or per-epoch artifact overwritten',
 checkpoint_restore_epochs=[1,25,50,75,100],
 scheduling=dict(max_concurrent=4,max_Full=2,max_PlaceboSocial=2,
  ordered_queue_per_model='seed42 ZARA2, seed42 HOTEL, seed2026 ZARA2, seed2026 HOTEL',
  caveat='independent processes/RNG; shared single GPU, run wall times include contention; configuration never changes after observing seed42'),
 bootstrap=dict(replicates=2000,base_seed=20261016,namespace='stage5p5-cluster-bootstrap-v1',
  stable_seed='SHA256(namespace|base20261016|recording|training_seed), first8 bytes modulo2^63',
  unit='recording-local person ID; sample n_person IDs with replacement; keep all windows for each selected ID multiplicity',
  common_draws='same sampled cluster-index matrix for ADE/FDE, both weightings and all three metric families within a recording/training-seed',
  window_weighted='sum cluster error sums with multiplicity / sum cluster window counts with multiplicity',
  person_equal='mean selected within-person mean paired errors',
  CI='percentile 2.5/97.5, 2000 paired cluster bootstrap draws; descriptive, not independence or exposure correction',
  groups='all frozen history groups reported descriptively; bootstrap main whole recording only, no subgroup significance searches'),
 aggregation=['per recording target-window','per recording equal person-ID','two-recording equal macro separately per weighting','all three individual seeds; mean and sample SD ddof1'],
 small_group=dict(person_IDs_less_than=10,target_windows_less_than=30,rule='flag if either; retain empty groups with null metrics'),
 stability=dict(primary='best K1 latent-prior ADE paired Full-minus-Placebo; strict negative indicates Full better',
  stronger_development_evidence='all 3 seeds x 2 recordings x 2 weightings have negative ADE deltas; paired FDE must also be negative for joint ADE/FDE support',
  unstable='any ADE direction reversal across seeds in any recording/weighting: current benefit not stable, stop complex social-module development',
  ADE_FDE_conflict='ADE consistently negative but FDE not: ADE-only evidence, explicitly conflicting; no joint performance claim',
  magnitude='report mean paired delta, sample SD of deltas and per-model across-seed errors; no significance decision from n=3 or chosen SNR cutoff',
  bootstrap_not_gate=True),
 history_reference_only=['NoSocial-seed17','EgoCapacity-seed17','CV-Last2','CV-OLS4','CV-OLS8'],
 independent_confirmation=False,ETH_evaluated=False,extra_seeds=False,extra_epochs=False,architecture_changes=False,
 failure_policy='first nonfinite stops affected run and saves failure payload; no hyperparameter retry; other frozen runs still execute',automatic_push=False)
plan['evaluation'].update(stable_key='protocol|seed<run_seed>|recording|start_frame|target_id',
 seed_namespace='stage5p2-prior-noise-v1',posterior_seed_namespace='stage5p2-posterior-noise-v1',
 seed_rule='namespace unchanged; identity contains run_seed 17/42/2026. Seed17 must exactly reproduce frozen evaluator; same seed Full/Placebo share epsilons',
 batch_size=128,samples=20,FPC=False)
json_new('configs/stage5p5/protocol.json',plan)
print('Protected historical files',len(paths))
