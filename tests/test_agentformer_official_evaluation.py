"""Failure-sensitive tests of the strict saved-output scoring boundary."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.agentformer_official_evaluation.runtime import aligned_candidates, close_to_author
from src.agentformer_benchmark.evaluation import target_metrics, aggregate_targets


def fixture_files(tmp_path, ids=(7, 2), frame=40):
    raw = np.array([[t, i, i*.3+t*.01, -i*.1+t*.02] for i in ids for t in range(frame-7, frame+13)], dtype=np.float32)
    future = raw[raw[:, 0] > frame]
    files = []
    # Reverse row order in half the candidates: joins must depend on ID/frame, not row position.
    for k in range(20):
        rows = future.copy()
        rows[:, 2:] += (k+1)*.01
        if k % 2:
            rows = rows[::-1]
        path = tmp_path/f'sample_{k:03d}.txt'
        np.savetxt(path, rows, fmt='%.6f')
        files.append(path)
    return files, raw


@pytest.mark.parametrize('ids', [(7,), (7,2), (7,2,50,1,19)])
def test_exact_identity_frame_alignment_independent_of_row_order(tmp_path, ids):
    files, raw = fixture_files(tmp_path, ids)
    p, gt, _ = aligned_candidates(files, list(ids), 40, raw)
    assert p.shape == (len(ids),20,12,2)
    np.testing.assert_allclose(p[:, 0]-gt, .01, atol=2e-6)
    result = target_metrics(p.astype(np.float64), gt.astype(np.float64))
    np.testing.assert_allclose(result['minADE20'], 2**.5*.01, atol=2e-6)


@pytest.mark.parametrize('failure', ['missing_candidate','extra_candidate','duplicate_ID_frame','missing_future','nonfinite','wrong_ID','wrong_frame'])
def test_corrupt_outputs_must_stop_scoring(tmp_path, failure):
    files, raw = fixture_files(tmp_path)
    if failure == 'missing_candidate':
        files = files[:-1]
    elif failure == 'extra_candidate':
        files = files+[files[0]]
    else:
        a = np.loadtxt(files[0], ndmin=2)
        if failure == 'duplicate_ID_frame': a[0] = a[1]
        if failure == 'missing_future': a = a[:-1]
        if failure == 'nonfinite': a[0,2] = np.nan
        if failure == 'wrong_ID': a[0,1] = 999
        if failure == 'wrong_frame': a[0,0] += 100
        np.savetxt(files[0], a)
    with pytest.raises(ValueError):
        aligned_candidates(files, [7,2], 40, raw)


def test_independent_winners_and_target_not_window_weighting():
    p = np.full((1,20,12,2),100.)
    y = np.zeros((1,12,2))
    p[0,0,:,0], p[0,0,:,1], p[0,0,-1,0] = 1,0,10
    p[0,1,:,0], p[0,1,:,1], p[0,1,-1,0] = 2,0,0
    r = target_metrics(p,y)
    assert r['ADE_winner'][0] == 0 and r['FDE_winner'][0] == 1
    assert r['minADE20'][0] == 1.75 and r['minFDE20'][0] == 0
    keys = ['minADE20','minFDE20','K1_first_ADE','K1_first_FDE']
    groups = [{k: np.array([1.]) for k in keys}, {k: np.array([3.,3.,3.]) for k in keys}]
    assert all(v == 2.5 for v in aggregate_targets(groups).values())


def test_predeclared_reproduction_gate_is_symmetric():
    assert close_to_author(.14+.009, .14)
    assert close_to_author(.75-.037, .75)
    assert not close_to_author(.14+.011, .14)
    assert not close_to_author(.75-.039, .75)
