"""Metric counterexamples and official protocol invariants, no training."""
import io
import importlib.util
import numpy as np
import pytest
import torch
from src.agentformer_benchmark.official import (install_compatibility, configuration,
    read_official_file, enumerate_windows, DATA, SOURCE)
from src.agentformer_benchmark.evaluation import target_metrics, aggregate_targets


def test_metric_winners_are_separate_and_first_sample_is_not_best20():
    p = np.full((1, 20, 12, 2), 100.)
    y = np.zeros((1, 12, 2))
    p[0, 0] = [1., 0.]
    p[0, 0, -1] = [10., 0.]
    p[0, 1] = [2., 0.]
    p[0, 1, -1] = [0., 0.]
    r = target_metrics(p, y)
    assert r['minADE20'].item() == 1.75
    assert r['minFDE20'].item() == 0.
    assert r['K1_first_FDE'].item() == 10.
    assert r['ADE_winner'].item() == 0 and r['FDE_winner'].item() == 1


def test_target_weighted_aggregation_is_not_mean_of_windows():
    keys = ['minADE20', 'minFDE20', 'K1_first_ADE', 'K1_first_FDE']
    groups = [{k: np.array([1.]) for k in keys}, {k: np.array([3., 3., 3.]) for k in keys}]
    assert all(v == 2.5 for v in aggregate_targets(groups).values())


@pytest.mark.parametrize('fold', ['eth', 'hotel', 'univ', 'zara1', 'zara2'])
def test_official_split_has_no_held_recording_in_train_or_val(fold):
    install_compatibility()
    from data.ethucy_split import get_ethucy_split
    train, val, test = get_ethucy_split(fold)
    assert set(test).isdisjoint(s.removesuffix('_train') for s in train)
    assert set(test).isdisjoint(s.removesuffix('_val') for s in val)
    assert len(train) == len(val) == (6 if fold == 'univ' else 7)
    assert 'crowds_zara03_train' in train and 'uni_examples_train' in train
    assert len(test) == (2 if fold == 'univ' else 1)


def test_official_ETH_scale_two_restores_world_meters_without_test_scores():
    install_compatibility()
    from data.preprocessor import preprocess
    cfg = configuration('eth')
    assert cfg.traj_scale == 2 and cfg.max_train_agent == 32
    sequence = 'biwi_hotel_train'  # ETH fold's TRAIN source, never ETH test.
    a = read_official_file(DATA/'eth'/(sequence+'.txt'))
    w = enumerate_windows(a)[0]
    packet = preprocess(str(DATA), sequence, cfg, io.StringIO())(w['last_observed_frame'])
    for actor, h, f in zip(packet['valid_id'], packet['pre_motion_3D'], packet['fut_motion_3D']):
        restored = torch.cat([h, f]).numpy()*cfg.traj_scale
        actual = a[(a[:, 1] == actor) & (a[:, 0] >= w['start_frame']) & (a[:, 0] <= w['end_frame']), 2:]
        np.testing.assert_allclose(restored, actual, atol=1e-6, rtol=1e-6)


def test_future_existence_is_declared_offline_qualification():
    # Same history; removing a neighbor's future legitimately changes this
    # official offline benchmark's population. Do not call this protocol causal.
    a = np.array([[t, i, t*.1, i] for t in range(20) for i in range(2)])
    before = enumerate_windows(a)
    after = enumerate_windows(a[~((a[:, 0] == 19) & (a[:, 1] == 1))])
    assert before[0]['target_ids'] == [0, 1] and after[0]['target_ids'] == [0]


@pytest.mark.parametrize('k', [1, 19, 21])
def test_wrong_candidate_budget_rejected(k):
    with pytest.raises(ValueError):
        target_metrics(np.zeros((1, k, 12, 2)), np.zeros((1, 12, 2)))
