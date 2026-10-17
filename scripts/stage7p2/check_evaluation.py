"""Official formulas, serializer, GT alignment and real-data eval CLI checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import numpy as np
import torch
from src.agentformer_benchmark.official import install_compatibility, configuration, SOURCE, DATA, CACHE
from src.agentformer_benchmark.evaluation import target_metrics, aggregate_targets
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

OUT = Path('results/tables/stage7p2')


def main():
    install_compatibility()
    from data.preprocessor import preprocess
    from data.ethucy_split import get_ethucy_split
    from utils.utils import mkdir_if_missing
    spec = importlib.util.spec_from_file_location('stage7p2_official_eval', SOURCE/'eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Two winners MUST differ: picking one trajectory for both is incorrect.
    pred = np.full((1, 20, 12, 2), 100., dtype=np.float64)
    truth = np.zeros((1, 12, 2), dtype=np.float64)
    pred[0, 0, :, 0], pred[0, 0, :, 1] = 1., 0.
    pred[0, 0, -1, 0] = 10.
    pred[0, 1, :, 0], pred[0, 1, :, 1] = 2., 0.
    pred[0, 1, -1, 0] = 0.
    ref = target_metrics(pred, truth)
    assert ref['ADE_winner'].item() != ref['FDE_winner'].item()
    assert module.compute_ADE(pred, truth) == ref['minADE20'].item() == 1.75
    assert module.compute_FDE(pred, truth) == ref['minFDE20'].item() == 0.
    cfg = configuration('hotel')
    tree = ast.parse((SOURCE/'test.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'save_prediction')
    env = dict(np=np, cfg=cfg, mkdir_if_missing=mkdir_if_missing)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE/'test.py'), 'exec'), env)
    audit = json.load(open(OUT/'official_data_audit.json'))
    row = next(r for r in audit['official_files'] if r['fold'] == 'hotel' and r['sequence'] == 'crowds_zara02_val')
    from src.agentformer_benchmark.official import read_official_file, enumerate_windows
    window = enumerate_windows(read_official_file(row['path']))[0]
    processor = preprocess(str(DATA), row['sequence'], cfg, io.StringIO(), split='val', phase='testing')
    packet = processor(window['last_observed_frame'])
    hist = torch.stack(packet['pre_motion_3D'])
    gt = torch.stack(packet['fut_motion_3D']).numpy()
    last, velocity = hist[:, -1], hist[:, -1]-hist[:, -2]
    cv = last[:, None]+velocity[:, None]*torch.arange(1, 13)[None, :, None]
    candidates = torch.stack([cv+torch.tensor([k*.01, -k*.005]) for k in range(20)], dim=1)
    directory = CACHE/'evaluation_equivalence_v2'
    samples, recon = directory/'samples', directory/'recon'
    for sequence in get_ethucy_split('hotel')[1]:
        (samples/sequence).mkdir(parents=True, exist_ok=True)
    for k in range(20):
        env['save_prediction'](candidates[:, k], packet, f'/sample_{k:03d}', str(samples))
    # Save exact-GT recon separately: eval samples must NOT read this directory.
    env['save_prediction'](torch.from_numpy(gt), packet, '', str(recon))
    prediction_files = sorted((samples/row['sequence']).glob('frame_*/*txt'))
    assert len(prediction_files) == 20
    serialized = np.stack([np.loadtxt(p, ndmin=2) for p in prediction_files])
    ids = sorted(map(int, packet['valid_id']))
    by_id, true_by_id = [], []
    raw = read_official_file(row['path'])
    for actor in ids:
        idx = np.flatnonzero(serialized[0, :, 1] == actor)
        agent_prediction, actor_gt = module.align_gt(serialized[:, idx], raw[raw[:, 1] == actor])
        by_id.append(agent_prediction)
        true_by_id.append(actor_gt)
    paired = target_metrics(np.stack(by_id), np.stack(true_by_id))
    official = dict(minADE20=module.compute_ADE(by_id, true_by_id), minFDE20=module.compute_FDE(by_id, true_by_id))
    expected = aggregate_targets([paired])
    assert max(abs(official[k]-expected[k]) for k in official) < 1e-12
    workspace = CACHE/'eval_workspace_v2'
    (workspace/'datasets').mkdir(parents=True, exist_ok=True)
    (workspace/'datasets/eth_ucy').symlink_to(DATA, target_is_directory=True)
    subprocess_env = os.environ.copy()
    subprocess_env['PYTHONPATH'] = str(SOURCE)+os.pathsep+str(Path('experiments/stage7/cache/deps').resolve())
    command = [sys.executable, str(SOURCE/'eval.py'), '--dataset', 'hotel', '--data', 'val',
        '--results_dir', str(samples.resolve()), '--log_file', str((directory/'official_cli_eval.txt').resolve())]
    result = subprocess.run(command, cwd=workspace, env=subprocess_env, capture_output=True, text=True)
    with (OUT/'official_eval_cli_v2.txt').open('x') as f:
        f.write(result.stdout+result.stderr)
    assert result.returncode == 0, result.stderr
    for metric, key in [('ADE', 'minADE20'), ('FDE', 'minFDE20')]:
        count, value = re.findall(r'(\d+) '+metric+r': ([0-9.]+)', result.stdout)[-1]
        assert int(count) == len(ids)
        assert abs(float(value)-expected[key]) <= .000051
    unrounded = target_metrics(candidates.numpy(), gt)
    rounded_delta = max(abs(expected[k]-float(unrounded[k].mean())) for k in official)
    assert rounded_delta < .001
    # Explicit unequal group sizes catch accidental window-mean weighting.
    keys = ['minADE20', 'minFDE20', 'K1_first_ADE', 'K1_first_FDE']
    groups = [{k: np.array([1.]) for k in keys}, {k: np.array([3., 3., 3.]) for k in keys}]
    weighted = aggregate_targets(groups)
    assert all(value == 2.5 for value in weighted.values())
    json_new(OUT/'evaluation_equivalence.json', dict(original_eval_formulas_equal=True,
        independent_ADE_FDE_winners_verified=True, synthetic_ADE_winner=int(ref['ADE_winner'][0]),
        synthetic_FDE_winner=int(ref['FDE_winner'][0]), official_serializer_and_real_GT_alignment=True,
        official_eval_CLI_passed=True, real_engineering_sequence=row['sequence'], people=len(ids),
        candidates=20, trajectories_are='CV plus fixed offsets, evaluator engineering check only; not model benchmark scores',
        only_samples_evaluated=True, GT_reconstruction_directory_excluded=True,
        serialization_rounding_error_m=rounded_delta, official_original_cli_returncode=result.returncode,
        unit='target-window, target-weighted within fold; UNIV two sequences pooled by target count',
        five_fold_aggregation='Unweighted mean of five scene metrics, not pooled target mean',
        future_model_scoring_started=False, ETH_test_scores_computed=False,
        evaluator_SHA256=sha256(SOURCE/'eval.py'), serializer_source_SHA256=sha256(SOURCE/'test.py'),
        source_SHA256=sha256(__file__)))
    print('Official metric formulas, separate winners, serialization, real GT alignment and eval CLI passed', flush=True)


if __name__ == '__main__':
    main()
