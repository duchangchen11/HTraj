"""Admission contract tests; artificial records do not imply accuracy."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from src.data.preprocessing import Recording
from src.agentformer_admission.causal import (build_context, labels_for_context,
    inference_packet, original_training_packet, UnlabeledContextError, stable_epsilon)
from scripts.stage7p1.raw_variants import variants


def fixture_record(n=3):
    by_frame = {10*t: {i: np.array([i+t*.1, i*.2], np.float32) for i in range(n)} for t in range(20)}
    return Recording('unit', 'UNIT', np.arange(20)*10, by_frame)


def test_context_reads_only_declared_history():
    r = fixture_record()
    class Spy(dict):
        def get(self, key, default=None):
            assert 0 <= key <= 70
            return super().get(key, default)
    r.by_frame = Spy(r.by_frame)
    c = build_context(r, 0)
    assert c.ids.tolist() == [0, 1, 2]
    assert c.support == tuple(range(0, 80, 10))


@pytest.mark.parametrize('n', [1, 2, 5, 12])
def test_actual_labels_packet_and_no_update(n):
    r = fixture_record(n)
    c = build_context(r, 0)
    p = original_training_packet(c, r)
    assert len(p['pre_motion_3D']) == n
    assert all(v.eq(1).all() for v in p['fut_motion_mask'])
    expected = torch.from_numpy(r.by_frame[80][0])
    assert torch.equal(p['fut_motion_3D'][0][0], expected)


def test_missing_labels_are_nan_and_original_training_fails_closed():
    r = fixture_record()
    del r.by_frame[90][1]
    del r.by_frame[100][2]
    c = build_context(r, 0)
    y, m = labels_for_context(c, r)
    assert c.ids.tolist() == [0, 1, 2]
    assert np.isnan(y[1, 1]).all() and not m[1, 1]
    with pytest.raises(UnlabeledContextError):
        original_training_packet(c, r)
    p = inference_packet(c)
    assert all(torch.isnan(v).all() for v in p['fut_motion_3D'])
    assert all(v.eq(0).all() for v in p['fut_motion_mask'])


def test_partial_history_excluded_without_imputation():
    r = fixture_record()
    del r.by_frame[0][1]
    c = build_context(r, 0)
    assert c.ids.tolist() == [0, 2] and c.partial_ids.tolist() == [1]
    assert inference_packet(c)['pre_motion_3D'][0].shape == (8, 2)


def test_stable_noise_follows_ids_and_prefix():
    c = build_context(fixture_record(), 0)
    p = np.array([2, 0, 1])
    cp = replace(c, ids=c.ids[p], history=c.history[p])
    assert torch.equal(stable_epsilon(cp, 20), stable_epsilon(c, 20)[:, p])
    assert torch.equal(stable_epsilon(c, 1), stable_epsilon(c, 20)[:1])


def test_raw_variants_parse_again_and_preserve_context(tmp_path):
    r = fixture_record()
    for t in range(11, 20):
        del r.by_frame[t*10][1]
    raw = np.array([[f, i, *xy] for f, row in r.by_frame.items() for i, xy in row.items()])
    c = build_context(r, 0)
    names = []
    for name, changed, details in variants(raw, c, 0):
        path = tmp_path/f'{name}.txt'
        np.savetxt(path, changed)
        reread = Recording.read(path, 'UNIT')
        rebuilt = build_context(reread, 0)
        assert np.array_equal(c.history, rebuilt.history)
        assert np.array_equal(c.ids, rebuilt.ids)
        assert details['changed_future_rows'] > 0
        names.append(name)
    assert len(names) == 5


def test_original_math_mask_does_not_mask_KL_or_denominators():
    # Import the frozen loss file by filename so unrelated old model packages
    # cannot shadow its namespace during the full suite.
    import importlib.util
    from pathlib import Path
    path = Path('experiments/stage7/cache/AgentFormer/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/model/agentformer_loss.py')
    assert path.is_file()
    spec = importlib.util.spec_from_file_location('stage7p1_original_loss', path)
    loss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss)
    class Distribution:
        def __init__(self, values): self.values = values
        def kl(self, _): return self.values
    d = dict(fut_motion_orig=torch.zeros(2, 12, 2), train_dec_motion=torch.ones(2, 12, 2),
        fut_mask=torch.tensor([[1.]*12, [0.]*12]), batch_size=2,
        q_z_dist=Distribution(torch.tensor([[6.], [10.]])), p_z_dist=None,
        infer_dec_motion=torch.ones(2, 20, 12, 2))
    assert float(loss.compute_motion_mse(d, dict(weight=1))[0]) == 12.
    assert float(loss.compute_sample_loss(d, dict(weight=1))[0]) == 12.
    class Config(dict):
        min_clip = 2
    assert float(loss.compute_z_kld(d, Config(weight=1))[0]) == 8.
