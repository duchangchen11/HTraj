"""Create immutable inputs BEFORE inference. Never compute test errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from src.agentformer_benchmark.official import ROOT, SOURCE
from src.agentformer_official_evaluation.runtime import digest, FOLDS, SEQUENCES
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new, read_csv

base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert base == '507ef5ca613663ef4c31ea45a7090a4b28ed3cd0'
old = json.load(open('configs/stage7p2/protected_artifacts.json'))
protected = {r['path']: r['sha256'] for r in old['files']}
new = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
new |= {str(p) for p in Path('experiments/stage7p2/cache').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for path in sorted(new-protected.keys()):
    protected[path] = digest(path)
json_new('configs/stage7p3/protected_artifacts.json', dict(base_commit=base,
    inherited_SHA_verified_at_delivery=True, files=[dict(path=p, sha256=h) for p, h in sorted(protected.items())]))
pairs = json.load(open('results/tables/stage7p2/official_checkpoint_strict_loading.json'))['cases']
files = []
for row in pairs:
    for kind in ['CVAE', 'DLow']:
        source = SOURCE/f"cfg/eth_ucy/{row['fold']}/{row[kind+'_config']}.yml"
        destination = Path('configs/stage7p3/official')/row['fold']/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as f:
            f.write(source.read_bytes())
        assert digest(destination) == row[kind+'_config_sha256']
        files.append(dict(path=str(destination), sha256=digest(destination)))
        checkpoint = Path(row[kind+'_checkpoint'])
        assert digest(checkpoint) == row[kind+'_sha256']
        files.append(dict(path=str(checkpoint.relative_to(ROOT)), sha256=row[kind+'_sha256']))
sources = {r['path']: r for r in json.load(open('results/tables/stage7/sources/AgentFormer.json'))['files'] if 'error' not in r}
sources.update({r['path']: r for r in json.load(open('results/tables/stage7/sources/AgentFormer_single_retry.json'))['attempts']})
source_files = [dict(path=r['local_path'], sha256=r['sha256'], git_blob=r['git_blob']) for r in sources.values()]
data = json.load(open('results/tables/stage7p2/official_data_acquisition.json'))['files']
data_files = [dict(path=r['path'], sha256=r['sha256'], git_blob=r['git_blob'], upstream_path=r['upstream_path']) for r in data]
windows = [r for r in read_csv('results/tables/stage7p2/official_window_inventory.csv.gz') if r['sequence'] in SEQUENCES[r['fold']]]
csv_new('configs/stage7p3/test_windows.csv.gz', windows)
counts = {fold: dict(windows=sum(r['fold'] == fold for r in windows),
    targets=sum(int(r['people']) for r in windows if r['fold'] == fold),
    max_people=max(int(r['people']) for r in windows if r['fold'] == fold)) for fold in FOLDS}
execution_files = [*Path('scripts/stage7p3').glob('*.py'), *Path('src/agentformer_official_evaluation').glob('*.py'),
    Path('tests/test_agentformer_official_evaluation.py'), Path('src/agentformer_benchmark/official.py'),
    Path('src/agentformer_benchmark/evaluation.py'), Path('scripts/stage7p1/runtime.py')]
json_new('configs/stage7p3/evaluation_protocol.json', dict(base_commit=base,
    name='HTraj-AgentFormer-OfficialETHUCY8x12-DLow20-v1', official_commit='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',
    folds=FOLDS, test_sequences=SEQUENCES, counts=counts,
    history=8, future=12, dt_seconds=.4, unit='m', K=20, mean=True,
    semantics='DLow learned latent heads z=b, NOT IID CVAE prior samples; original samples directory only',
    context='Official complete20 offline selection; future existence may determine context eligibility',
    previous_causal_protocol_comparison=False, formal_training_epochs=0, optimizer_updates=0,
    selection='Fixed author pairs from Stage7.2; no checkpoint selection; inference seed=original cfg.seed',
    metric='Independent per-target ADE/FDE minima over 20; within-fold target-weighted; five-fold equal macro',
    ETH_traj_scale=2, other_traj_scale=1, FPC=False, GT_reordering=False,
    serialization='Original test.py save_prediction, %.3f world coordinates; recon excluded from metrics',
    evaluator_full_precision_tolerance_m=1e-5, official_CLI_4decimal_tolerance_m=.000051,
    preliminary_reproduction_threshold=dict(absolute_m=.01, relative_fraction=.05,
        rule='Each fold and macro ADE/FDE absolute gap <= max(.01m, author value*5%); engineering gate, not improvement'),
    rounding_compatibility_halfwidth_m=.005,
    author_report=dict(eth=[.45,.75], hotel=[.14,.22], univ=[.25,.45], zara1=[.18,.30], zara2=[.14,.24], macro=[.23,.39]),
    author_reference='https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5#ethucy',
    failures='Stop scene on OOM, NaN, ID mismatch; preserve partial files; never trim actors/change K/reselect weights',
    pairs=pairs, official_sources=source_files, data_files=data_files, checkpoints_and_configs=files,
    frozen_test_window_manifest_SHA256=digest('configs/stage7p3/test_windows.csv.gz'),
    execution_source_SHA256={str(p): digest(p) for p in execution_files}, automatic_push=False))
print('Prepared', len(protected), 'protected files; counts:', counts, flush=True)
