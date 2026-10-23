"""Matched K20 evaluation with a physically history-only prediction phase.

This is a reviewed future entry, not permission to launch training: Stage11C-1
formal execution remains blocked whenever the frozen resource admission fails.
Only the six semantic input maps differ from Stage11B-1C: they are zero,
including padding.  Sampling, KMeans, path decoding and metric arithmetic
remain the frozen implementations.  All predictions are committed to a new
cache and their completeness/reproducibility gates pass before GT is read.
"""
from .common import (
    ROOT, OUT, CACHE, sha, jnew, cnew, shapes, zero_semantic, history_records,
)
from scripts.stage11b1c.data import truths, manifest
from scripts.stage11b1c.validate import calculate as baseline_calculate
from scripts.stage11b1c.validate import METRICS as BASELINE_METRICS
from scripts.stage11b1r.predictor import Predictor
from scripts.stage11b0.runtime import state_digest
import time
import traceback
import numpy as np
import torch


METRICS = list(BASELINE_METRICS) + [
    'Cover10', 'Cover25', 'Cover50', 'A25fail', 'K1_K20_gap',
]


def calculate(paths, truth, goals):
    """Preserve old float/double arithmetic; append descriptive quantities."""
    result = baseline_calculate(paths, truth, goals)
    for threshold in (10, 25, 50):
        result[f'Cover{threshold}'] = (result['minGoalFDE20'] <= threshold).double()
    result['A25fail'] = (result['minGoalFDE20'] > 25).double()
    result['K1_K20_gap'] = result['ADE1'] - result['minADE20']
    return result


def _checkpoint_identity(net, checkpoint_info):
    """Check the live model against its immutable, already saved checkpoint."""
    path = ROOT / checkpoint_info['path']
    assert path.is_file() and sha(path) == checkpoint_info['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    live = state_digest(net.state_dict())
    assert state_digest(saved['model']) == live, 'Live weights differ from checkpoint'
    return live


def _history_identity(records, source, frozen_records=None):
    assert len(records) == len({row['sample_id'] for row in records}) == 998
    assert {row['sample_id'] for row in records} == set(source)
    frozen = None if frozen_records is None else {row['sample_id']: row for row in frozen_records}
    if frozen is not None:
        assert set(frozen) == set(source)
    for row in records:
        assert set(row) == {'sample_id', 'scene', 'history'}, 'No GT field permitted'
        assert row['scene'] == source[row['sample_id']]['scene']
        history = torch.as_tensor(row['history'])
        assert history.shape == (8, 2) and history.dtype == torch.float32
        assert torch.isfinite(history).all()
        if frozen is not None:
            expected = frozen[row['sample_id']]
            assert row['scene'] == expected['scene']
            assert torch.equal(history, torch.as_tensor(expected['history'])), 'History differs from frozen GT-free cache'


def predict_all(net, cfg, templates, records, epoch, checkpoint_info, budget,
                scope='validation'):
    """Generate all998 paths without a GT loader or segmentation forward call."""
    assert scope == 'validation', 'Formal evaluator accepts the frozen validation pool only'
    source = {row['sample_id']: row for row in manifest('validation')}
    _history_identity(records, source, history_records('validation'))
    model_sha = _checkpoint_identity(net, checkpoint_info)
    table = shapes()
    predictor = Predictor(net, cfg, templates)
    device = next(net.parameters()).device
    stem = f'epoch{epoch:03d}'
    path = CACHE / f'predictions/{stem}.pt'
    assert not path.exists()
    start = time.perf_counter()
    cache, traces, paths, goals, gate = {}, [], [], [], []
    current = None
    try:
        for i, row in enumerate(records):
            current = row
            budget.guard(30)
            scene = row['scene']
            if scene not in cache:
                cache[scene] = zero_semantic(scene, device, table)
                assert state_digest({'tensor': cache[scene]}) == table[scene]['semantic_zero_SHA']
            history = torch.as_tensor(row['history']).unsqueeze(0)
            p, g, trace = predictor.predict(row['sample_id'], history, semantic=cache[scene])
            assert trace['segmentation_seconds'] == 0
            assert p.shape == (20, 1, 12, 2) and g.shape == (20, 1, 2)
            assert p.dtype == g.dtype == torch.float32
            assert trace['proposal_shape'] == [1, 1, 10000, 2]
            assert torch.isfinite(p).all() and torch.isfinite(g).all()
            paths.append(p)
            goals.append(g)
            traces.append(dict(
                epoch=epoch, sample_id=row['sample_id'], scene=scene,
                history_SHA=source[row['sample_id']]['history_SHA'],
                history_tensor_SHA=state_digest({'tensor': torch.as_tensor(row['history'])}),
                checkpoint_SHA=checkpoint_info['sha256'], model_state_SHA=model_sha,
                semantic_zero_SHA=table[scene]['semantic_zero_SHA'],
                semantic_shape=list(cache[scene].shape), semantic_all_zero=True, **trace,
            ))
            if (i + 1) % 50 == 0:
                print('VALIDATE epoch', epoch, 'target', i + 1, '/998',
                      'seconds', round(time.perf_counter() - start), flush=True)
        predicted = torch.cat(paths, dim=1)
        sampled_goals = torch.cat(goals, dim=1)
        assert predicted.shape == (20, 998, 12, 2)
        assert sampled_goals.shape == (20, 998, 2)
        # Preserve actual output even if a subsequent repeat gate fails.  Such a
        # validation never receives COMPLETE status or checkpoint-selection use.
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            sample_ids=[row['sample_id'] for row in records], paths=predicted,
            goals=sampled_goals, checkpoint_SHA=checkpoint_info['sha256'],
            model_state_SHA=model_sha, epoch=epoch, K=20, TTST_proposals=10000,
            units='original reference pixels', semantic_mode='all_zero_including_padding',
        ), path)
        for i in dict.fromkeys([997, 499, 0]):
            row = records[i]
            history = torch.as_tensor(row['history']).unsqueeze(0)
            for mode in ['cached_reverse_repeat', 'recreated_zero']:
                budget.guard(30)
                semantic = cache[row['scene']] if mode == 'cached_reverse_repeat' else zero_semantic(row['scene'], device, table)
                p, g, trace = predictor.predict(row['sample_id'], history, semantic=semantic)
                assert trace['segmentation_seconds'] == 0
                assert torch.equal(p, paths[i]) and torch.equal(g, goals[i]), 'K20 repeat/order/zero-cache reproducibility failed'
                assert all(trace[key] == traces[i][key] for key in ['proposal_SHA', 'goal_SHA', 'path_SHA'])
                gate.append(dict(epoch=epoch, sample_id=row['sample_id'], mode=mode, bitwise=True, **trace))
        assert state_digest(net.state_dict()) == model_sha, 'Evaluation changed model state'
        cnew(OUT / f'prediction_traces_{stem}.csv.gz', traces)
        cnew(OUT / f'reproducibility_{stem}.csv', gate)
        info = dict(
            path=str(path.relative_to(ROOT)), sha256=sha(path), targets=998,
            K=20, TTST_proposals=10000, checkpoint_SHA=checkpoint_info['sha256'],
            model_state_SHA=model_sha, all_predictions_completed_before_GT_scoring=True,
            prediction_seconds=time.perf_counter() - start,
            trace_SHA=sha(OUT / f'prediction_traces_{stem}.csv.gz'),
            reproducibility_SHA=sha(OUT / f'reproducibility_{stem}.csv'),
            scope=scope, segmentation_forward_calls=0, semantic_all_zero=True,
            history_only_input=True, all998_history_ID_and_scene_checked=True,
            model_equals_checkpoint=True, repeats=6, repeat_bitwise=True,
        )
        jnew(OUT / f'prediction_manifest_{stem}.json', info)
        return predicted, sampled_goals, info
    except Exception as error:
        jnew(OUT / f'prediction_failure_{stem}.json', dict(
            epoch=epoch, checkpoint_SHA=checkpoint_info['sha256'],
            status='INCOMPLETE', completed_predictions=len(paths),
            sample_id=None if current is None else current['sample_id'],
            error_type=type(error).__name__, error=str(error),
            prediction_cache_saved=path.exists(), GT_scoring_started=False,
        ))
        raise


def score(paths, goals, records, epoch, checkpoint_info, prediction_info):
    """GT is first read after the verified all-target prediction artifact exists."""
    assert prediction_info['all_predictions_completed_before_GT_scoring']
    assert prediction_info['targets'] == len(records) == 998
    assert prediction_info['checkpoint_SHA'] == checkpoint_info['sha256']
    artifact = ROOT / prediction_info['path']
    assert artifact.is_file() and sha(artifact) == prediction_info['sha256']
    truth_index = truths('validation')
    y = torch.from_numpy(np.stack([truth_index[row['sample_id']] for row in records]))
    result = calculate(paths, y, goals)
    source = {row['sample_id']: row for row in manifest('validation')}
    paired, candidate = [], []
    distance = (paths - y.unsqueeze(0)).norm(dim=-1)
    ade, fde = distance.mean(-1), distance[:, :, -1]
    goal_error = (goals - y[None, :, -1]).norm(dim=-1)
    for i, row in enumerate(records):
        identity = source[row['sample_id']]
        paired.append(dict(
            epoch=epoch, sample_id=row['sample_id'], scene=row['scene'],
            person_id=identity['person_id'], frame_id=identity['frame_id'],
            history_SHA=identity['history_SHA'], GT_SHA=identity['GT_SHA'],
            checkpoint_SHA=checkpoint_info['sha256'],
            **{key: float(result[key][i]) for key in METRICS},
            best_ADE_head=int(result['best_ADE_head'][i]),
            best_FDE_head=int(result['best_FDE_head'][i]),
        ))
        for head in range(20):
            candidate.append(dict(
                epoch=epoch, sample_id=row['sample_id'], head=head,
                ADE=float(ade[head, i]), FDE=float(fde[head, i]),
                GoalFDE=float(goal_error[head, i]),
                goal_path_gap=float((goals[head, i] - paths[head, i, -1]).norm()),
            ))
    cnew(OUT / f'paired_epoch{epoch:03d}.csv', paired)
    cnew(OUT / f'candidate_errors_epoch{epoch:03d}.csv.gz', candidate)
    scenes = []
    for scene in sorted({row['scene'] for row in records}):
        subset = [row for row in paired if row['scene'] == scene]
        scenes.append(dict(
            epoch=epoch, scene=scene, targets=len(subset),
            checkpoint_SHA=checkpoint_info['sha256'],
            **{key: float(np.mean([row[key] for row in subset])) for key in METRICS},
        ))
    cnew(OUT / f'scene_metrics_epoch{epoch:03d}.csv', scenes)
    window = {key: float(result[key].mean()) for key in METRICS}
    macro = {key: float(np.mean([row[key] for row in scenes])) for key in METRICS}
    coverage = {f'within_{threshold}px': window[f'Cover{threshold}'] for threshold in (10, 25, 50)}
    return window, macro, scenes, coverage


def validate(net, cfg, templates, epoch, checkpoint_info, budget):
    start = time.perf_counter()
    net.eval()
    result = dict(epoch=epoch, status='INCOMPLETE', checkpoint=checkpoint_info,
                  targets_expected=998, targets_scored=0)
    try:
        records = history_records('validation')
        p, g, prediction_info = predict_all(net, cfg, templates, records, epoch, checkpoint_info, budget)
        window, macro, scenes, coverage = score(p, g, records, epoch, checkpoint_info, prediction_info)
        result.update(
            status='COMPLETE', targets_scored=998, K=20, TTST_proposals=10000,
            window=window, recording_macro=macro, scenes=scenes,
            goal_coverage=coverage, prediction_manifest=prediction_info,
            unit='original reference pixels', semantic_all_zero=True,
            segmentation_forward_calls=0,
        )
        return result
    except Exception as error:
        traceback.print_exc()
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['total_seconds'] = time.perf_counter() - start
        jnew(OUT / f'validation_epoch{epoch:03d}.json', result)
