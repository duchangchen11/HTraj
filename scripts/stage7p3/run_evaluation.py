"""Serial frozen inference, then official+independent scoring of identical files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import time
import traceback
import numpy as np
import torch
from src.agentformer_official_evaluation.runtime import *
from src.agentformer_benchmark.evaluation import target_metrics, aggregate_targets
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.data import array_sha
from src.socialvae_causal.io import csv_new, read_csv
from scripts.stage7p1.runtime import ResourceMeter
from scripts.stage7p3.preflight import load_pair


def infer():
    freeze_commit = committed_clean_gate()  # Must pass before creating any results.
    checked = verify_inputs()
    assert json.loads((OUT/'preflight.json').read_text())['no_test_errors']
    json_new(OUT/'execution_start.json', dict(freeze_commit=freeze_commit, verified_inputs=checked,
        git_clean_before_execution=True, started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        optimizer_updates=0, formal_training_epochs=0))
    install_compatibility()
    from data.dataloader import data_generator
    from utils.utils import print_log, mkdir_if_missing
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(0)
    workspace()
    windows = read_csv(ROOT/'configs/stage7p3/test_windows.csv.gz')
    failures = []
    original = Path.cwd()
    for case in protocol()['pairs']:
        fold = case['fold']
        expected = {(r['sequence'], int(r['last_observed_frame'])): set(json.loads(r['target_ids']))
                    for r in windows if r['fold'] == fold}
        raw_log = CACHE/f'{fold}_official_inference.txt'
        raw_log.parent.mkdir(parents=True, exist_ok=True)
        output = CACHE/'predictions'/fold
        assert not output.exists(), 'Refuse to overwrite predictions'
        output.mkdir(parents=True)
        seen, prediction_seconds = set(), []
        meter, model = None, None
        print('START serial inference', fold, 'expected', len(expected), 'windows', flush=True)
        try:
            os.chdir(WORK)
            with raw_log.open('x') as log, contextlib.redirect_stdout(log):
                cfg, model = load_pair(case, gpu=True)
                generator = data_generator(cfg, log, split='test', phase='testing')
                class CheckedGenerator:
                    split = 'test'
                    def is_epoch_end(self):
                        return generator.is_epoch_end()
                    def __call__(self):
                        data = generator()
                        if data is not None:
                            key = data['seq'], int(data['frame'])
                            assert key in expected and key not in seen, key
                            assert set(map(int, data['valid_id'])) == expected[key], key
                            assert all(torch.all(m == 1) for m in data['pre_motion_mask'])
                            assert all(torch.all(m == 1) for m in data['fut_motion_mask'])
                            seen.add(key)
                            if len(seen) % 50 == 0:
                                print(f'{fold}: {len(seen)}/{len(expected)} windows, no errors scored', file=sys.__stdout__, flush=True)
                        return data
                env = dict(np=np, torch=torch, sys=sys, os=os, cfg=cfg, model=model,
                    device=torch.device('cuda:0'), log=log, print_log=print_log, mkdir_if_missing=mkdir_if_missing)
                official_functions(env)
                original_predict = env['get_model_prediction']
                def checked_prediction(data, sample_k):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    recon, samples = original_predict(data, sample_k)
                    torch.cuda.synchronize()
                    prediction_seconds.append(time.perf_counter()-start)
                    assert samples.shape == (20, len(data['valid_id']), 12, 2)
                    if not torch.isfinite(samples).all() or not torch.isfinite(recon).all():
                        raise FloatingPointError('Nonfinite author prediction')
                    return recon, samples
                env['get_model_prediction'] = checked_prediction
                with ResourceMeter() as meter:
                    env['test_model'](CheckedGenerator(), str(output), cfg)
                assert seen == expected.keys()
            json_new(OUT/f'{fold}_inference.json', dict(fold=fold, success=True, windows=len(seen),
                targets=sum(len(v) for v in expected.values()), K=20, traj_scale=cfg.traj_scale,
                output_directory=str(output), resource=meter.result, model_only_seconds=sum(prediction_seconds),
                model_call_seconds_min=min(prediction_seconds), model_call_seconds_max=max(prediction_seconds),
                official_inference_log=str(raw_log), official_inference_log_SHA256=digest(raw_log),
                checkpoint_CVAE_SHA256=case['CVAE_sha256'], checkpoint_DLow_SHA256=case['DLow_sha256'],
                freeze_commit=freeze_commit, no_test_errors_yet=True, optimizer_updates=0))
            print('DONE', fold, 'seconds', round(meter.seconds, 2), flush=True)
        except Exception:
            failure = dict(fold=fold, success=False, completed_windows=len(seen),
                traceback=traceback.format_exc(), output_directory=str(output), no_retry=True)
            failures.append(failure)
            json_new(OUT/f'{fold}_inference_failure.json', failure)
            print('STOPPED scene', fold, failure['traceback'], file=sys.__stdout__, flush=True)
        finally:
            os.chdir(original)
            if model is not None:
                del model
            torch.cuda.empty_cache()
    json_new(OUT/'inference_completion.json', dict(all_five_succeeded=not failures, failures=failures,
        all_inference_before_metric_computation=True))


def score():
    assert (OUT/'inference_completion.json').exists(), 'Finish all inference before evaluation'
    verify_inputs()
    install_compatibility()
    official = evaluator()
    window_rows = read_csv(ROOT/'configs/stage7p3/test_windows.csv.gz')
    fold_results, comparisons, failures = [], [], []
    for fold in FOLDS:
        if not (OUT/f'{fold}_inference.json').exists():
            failures.append(dict(fold=fold, reason='Inference incomplete; no metrics'))
            continue
        begin = time.perf_counter()
        try:
            samples = CACHE/'predictions'/fold/'samples'
            rows = [r for r in window_rows if r['fold'] == fold]
            expected_dirs = {str(Path(r['sequence'])/f"frame_{int(r['last_observed_frame']):06d}") for r in rows}
            actual_dirs = {str(d.relative_to(samples)) for seq in samples.iterdir() for d in seq.iterdir()}
            assert actual_dirs == expected_dirs
            raw = {s: np.genfromtxt(DATA/fold/(s+'.txt'), dtype=str)[:, [0,1,13,15]].astype(np.float32) for s in SEQUENCES[fold]}
            groups, target_rows, hashes, alignments = [], [], [], []
            official_sums = dict(minADE20=0., minFDE20=0.)
            for row in rows:
                seq, frame = row['sequence'], int(row['last_observed_frame'])
                ids = sorted(json.loads(row['target_ids']))
                directory = samples/seq/f'frame_{frame:06d}'
                files = sorted(directory.iterdir())
                p32, truth, serialized = aligned_candidates(files, ids, frame, raw[seq])
                ref = target_metrics(p32.astype(np.float64), truth.astype(np.float64))
                groups.append(ref)
                author_p, author_y = [], []
                for actor in ids:
                    idx = np.flatnonzero(serialized[0, :, 1] == actor)
                    pa, ya = official.align_gt(serialized[:, idx], raw[seq][raw[seq][:, 1] == actor])
                    author_p.append(pa)
                    author_y.append(ya)
                assert np.array_equal(np.stack(author_y), truth)
                assert np.array_equal(np.stack(author_p), p32)
                official_sums['minADE20'] += official.compute_ADE(author_p, author_y)*len(ids)
                official_sums['minFDE20'] += official.compute_FDE(author_p, author_y)*len(ids)
                for i, actor in enumerate(ids):
                    target_rows.append(dict(fold=fold, sequence=seq, last_observed_frame=frame, person_id=actor,
                        GT_SHA256=array_sha(truth[i]), **{k: (int(v[i]) if 'winner' in k else float(v[i])) for k, v in ref.items()}))
                hashes.extend(dict(path=str(f.relative_to(ROOT)), sha256=digest(f), bytes=f.stat().st_size,
                    kind='samples', fold=fold) for f in files)
                alignments.append(dict(fold=fold, sequence=seq, frame=frame, people=len(ids),
                    candidates=len(files), future_frames=12, exact_GT_alignment=True,
                    GT_SHA256=array_sha(truth), prediction_SHA256=array_sha(p32)))
            n = len(target_rows)
            metrics = aggregate_targets(groups)
            full_official = {k: v/n for k, v in official_sums.items()}
            assert all(abs(full_official[k]-metrics[k]) <= protocol()['evaluator_full_precision_tolerance_m'] for k in full_official)
            # Record recon and serialized GT hashes too, but NEVER score them.
            for kind in ['recon', 'gt']:
                other = sorted((CACHE/'predictions'/fold/kind).rglob('*.txt'))
                assert len(other) == len(rows)
                hashes.extend(dict(path=str(f.relative_to(ROOT)), sha256=digest(f), bytes=f.stat().st_size,
                    kind=kind, fold=fold) for f in other)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(SOURCE)+os.pathsep+str(ROOT/'experiments/stage7/cache/deps')
            command = [sys.executable, str(SOURCE/'eval.py'), '--dataset', fold, '--data', 'test',
                '--results_dir', str(samples), '--log_file', str(CACHE/f'{fold}_official_eval_original.txt')]
            result = subprocess.run(command, cwd=WORK, env=env, capture_output=True, text=True)
            (OUT/f'{fold}_official_eval_cli.txt').write_text('\n'.join(line.rstrip() for line in (result.stdout+result.stderr).splitlines())+'\n')
            assert result.returncode == 0, result.stderr
            cli = {}
            for name, key in [('ADE', 'minADE20'), ('FDE', 'minFDE20')]:
                count, value = re.findall(r'(\d+) '+name+r': ([0-9.]+)', result.stdout)[-1]
                assert int(count) == n == protocol()['counts'][fold]['targets']
                cli[key] = float(value)
                assert abs(cli[key]-metrics[key]) <= protocol()['official_CLI_4decimal_tolerance_m']
            csv_new(OUT/f'{fold}_prediction_file_SHA256.csv.gz', hashes)
            csv_new(OUT/f'{fold}_target_metrics.csv.gz', target_rows)
            csv_new(OUT/f'{fold}_alignment.csv.gz', alignments)
            seq_rows = []
            for seq in SEQUENCES[fold]:
                subset = [r for r in target_rows if r['sequence'] == seq]
                seq_rows.append(dict(fold=fold, sequence=seq, targets=len(subset),
                    **{key: float(np.mean([r[key] for r in subset])) for key in metrics}))
            csv_new(OUT/f'{fold}_sequence_metrics.csv', seq_rows)
            summary = dict(fold=fold, windows=len(rows), targets=n, **metrics,
                official_full_precision=full_official, official_CLI_4decimals=cli,
                evaluator_max_difference_m=max(abs(full_official[k]-metrics[k]) for k in full_official),
                samples_scored_only=True, K=20, SHA_index_rows=len(hashes),
                candidate_files=len(rows)*20, full_alignment_passed=True, scoring_seconds=time.perf_counter()-begin)
            json_new(OUT/f'{fold}_metrics.json', summary)
            fold_results.append(summary)
            print('SCORED', fold, n, metrics, flush=True)
        except Exception:
            fail = dict(fold=fold, traceback=traceback.format_exc())
            failures.append(fail)
            json_new(OUT/f'{fold}_scoring_failure.json', fail)
            print('SCORING FAILED', fold, fail['traceback'], flush=True)
    for row in fold_results:
        for key, author in zip(['minADE20', 'minFDE20'], protocol()['author_report'][row['fold']]):
            local = row[key]
            comparisons.append(dict(fold=row['fold'], metric=key, author_m=author, local_m=local,
                signed_difference_m=local-author, absolute_difference_m=abs(local-author),
                relative_difference_percent=(local-author)/author*100,
                absolute_relative_difference_percent=abs(local-author)/author*100,
                within_published_rounding=abs(local-author) <= .005+1e-9, preliminary_close=close_to_author(local, author)))
    macro = None
    if len(fold_results) == 5:
        macro = {k: float(np.mean([r[k] for r in fold_results])) for k in ['minADE20','minFDE20','K1_first_ADE','K1_first_FDE']}
        for key, author in zip(['minADE20','minFDE20'], protocol()['author_report']['macro']):
            local = macro[key]
            comparisons.append(dict(fold='macro', metric=key, author_m=author, local_m=local,
                signed_difference_m=local-author, absolute_difference_m=abs(local-author),
                relative_difference_percent=(local-author)/author*100, absolute_relative_difference_percent=abs(local-author)/author*100,
                within_published_rounding=abs(local-author)<=.005+1e-9, preliminary_close=close_to_author(local, author)))
    csv_new(OUT/'author_comparison.csv', comparisons, fieldnames=list(comparisons[0]) if comparisons else ['fold','metric'])
    csv_new(OUT/'five_scene_metrics.csv', [{k: r[k] for k in ['fold','windows','targets','minADE20','minFDE20','K1_first_ADE','K1_first_FDE']} for r in fold_results],
        fieldnames=['fold','windows','targets','minADE20','minFDE20','K1_first_ADE','K1_first_FDE'])
    json_new(OUT/'evaluation_summary.json', dict(folds=fold_results, macro=macro, failures=failures,
        all_five_completed=len(fold_results)==5 and not failures,
        preliminary_reproduction_passed=len(fold_results)==5 and not failures and all(r['preliminary_close'] for r in comparisons),
        all_predictions_before_scoring=True, no_training=True, author_values_rounded_2decimals=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['infer','score'])
    args = parser.parse_args()
    globals()[args.action]()
