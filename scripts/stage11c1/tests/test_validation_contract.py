"""CPU-only contracts, independent oracles and staged GT access tests."""
import inspect
from pathlib import Path
import csv
import json
import numpy as np
import pytest
import torch

from scripts.stage11c1 import validate as module
from scripts.stage11b1c.validate import calculate as old_calculate
from scripts.stage11b0.runtime import state_digest
from scripts.stage11b0.common import sha


def test_old_metric_arithmetic_preserved_and_added_coverage():
    truth = torch.zeros((2, 12, 2), dtype=torch.float64)
    paths = torch.ones((20, 2, 12, 2), dtype=torch.float32) * 4
    goals = torch.zeros((20, 2, 2), dtype=torch.float32)
    paths[0, 0] = 0
    paths[0, 0, -1, 0] = 9
    paths[1, 0, -1] = 0
    goals[:, 1, 0] = 26
    result = module.calculate(paths, truth, goals)
    baseline = old_calculate(paths, truth, goals)
    for key in baseline:
        assert torch.equal(result[key], baseline[key]), key
    assert result['best_ADE_head'][0] == 0
    assert result['best_FDE_head'][0] == 1
    assert result['Cover25'].tolist() == [1., 0.]
    assert result['Cover50'].tolist() == [1., 1.]
    assert result['A25fail'].tolist() == [0., 1.]
    assert torch.equal(result['K1_K20_gap'], result['ADE1'] - result['minADE20'])


def test_no_future_field_even_with_the_same_history():
    source = {str(i): {'scene': 'synthetic'} for i in range(998)}
    records = [dict(sample_id=str(i), scene='synthetic', history=torch.zeros(8, 2)) for i in range(998)]
    module._history_identity(records, source)
    records[0]['future'] = torch.full((12, 2), 1e6)
    with pytest.raises(AssertionError, match='No GT field'):
        module._history_identity(records, source)
    assert 'future' not in inspect.signature(module.Predictor.predict).parameters


def test_history_values_must_match_frozen_history_cache():
    source = {str(i): {'scene': 'synthetic'} for i in range(998)}
    frozen = [dict(sample_id=str(i), scene='synthetic', history=torch.zeros(8, 2)) for i in range(998)]
    records = [dict(sample_id=row['sample_id'], scene=row['scene'], history=row['history'].clone()) for row in frozen]
    records[0]['history'][0, 0] = 1
    with pytest.raises(AssertionError, match='History differs'):
        module._history_identity(records, source, frozen)


class _Budget:
    def guard(self, *args, **kwargs):
        return None


class _Predictor:
    calls = []

    def __init__(self, net, cfg, templates):
        pass

    def predict(self, sample_id, history, image=None, semantic=None):
        assert image is None
        assert semantic.shape == (1, 6, 32, 32)
        assert torch.count_nonzero(semantic) == 0
        assert history.shape == (1, 8, 2)
        self.calls.append(sample_id)
        value = float(int(sample_id) % 3)
        p = torch.full((20, 1, 12, 2), value)
        g = torch.full((20, 1, 2), value)
        return p, g, dict(
            segmentation_seconds=0., proposal_shape=[1, 1, 10000, 2],
            proposal_SHA='proposal:' + sample_id,
            goal_SHA=state_digest({'tensor': g}), path_SHA=state_digest({'tensor': p}),
        )


@pytest.fixture
def synthetic_setup(monkeypatch, tmp_path):
    out = tmp_path / 'tables'
    out.mkdir()
    cache = tmp_path / 'cache'
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'OUT', out)
    monkeypatch.setattr(module, 'CACHE', cache)
    monkeypatch.setattr(module, 'Predictor', _Predictor)
    _Predictor.calls = []
    net = torch.nn.Linear(1, 1)
    cp = tmp_path / 'checkpoint.pt'
    torch.save({'model': net.state_dict()}, cp)
    checkpoint = dict(path='checkpoint.pt', sha256=sha(cp))
    semantic = torch.zeros((1, 6, 32, 32), dtype=torch.float32)
    table = {'synthetic': dict(semantic_shape=list(semantic.shape), semantic_zero_SHA=state_digest({'tensor': semantic}))}
    monkeypatch.setattr(module, 'shapes', lambda: table)
    rows = [dict(sample_id=str(i), scene='synthetic', person_id=str(i), frame_id='0',
                 history_SHA='history:' + str(i), GT_SHA='GT:' + str(i)) for i in range(998)]
    records = [dict(sample_id=row['sample_id'], scene='synthetic', history=torch.zeros((8, 2))) for row in rows]
    monkeypatch.setattr(module, 'manifest', lambda split: rows)
    monkeypatch.setattr(module, 'history_records', lambda split: records)
    return net, checkpoint, rows, records, out, cache


def test_prediction_saved_before_GT_then_full_contract(synthetic_setup, monkeypatch):
    net, checkpoint, rows, records, out, cache = synthetic_setup
    called = []

    def truth_loader(split):
        assert split == 'validation'
        prediction = cache / 'predictions/epoch001.pt'
        artifact = torch.load(prediction, weights_only=True)
        info = json.loads((out / 'prediction_manifest_epoch001.json').read_text())
        assert artifact['paths'].shape == (20, 998, 12, 2)
        assert artifact['sample_ids'] == [row['sample_id'] for row in rows]
        assert 'future' not in artifact and 'GT' not in artifact
        assert info['all_predictions_completed_before_GT_scoring']
        assert info['repeat_bitwise'] and info['repeats'] == 6
        assert len(_Predictor.calls) == 998 + 6
        called.append(True)
        return {row['sample_id']: np.zeros((12, 2), dtype=np.float64) for row in rows}

    monkeypatch.setattr(module, 'truths', truth_loader)
    result = module.validate(net, {}, (), 1, checkpoint, _Budget())
    assert called == [True]
    assert result['status'] == 'COMPLETE' and result['targets_scored'] == 998
    assert result['window']['Cover25'] == 1.
    # Original aggregation uses NumPy(scene rows) and Torch(window tensors),
    # whose reductions can differ at machine precision even for one scene.
    assert result['recording_macro'] == pytest.approx(result['window'], rel=1e-14, abs=1e-14)
    assert result['segmentation_forward_calls'] == 0
    assert (out / 'paired_epoch001.csv').is_file()
    assert (out / 'candidate_errors_epoch001.csv.gz').is_file()
    assert (out / 'scene_metrics_epoch001.csv').is_file()
    with (out / 'paired_epoch001.csv').open() as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 998
    assert paired[0]['history_SHA'] == rows[0]['history_SHA']
    assert paired[0]['GT_SHA'] == rows[0]['GT_SHA']


def test_future_cannot_be_loaded_on_failed_checkpoint(synthetic_setup, monkeypatch):
    net, checkpoint, rows, records, out, cache = synthetic_setup
    monkeypatch.setattr(module, 'truths', lambda split: pytest.fail('GT read before prediction admission'))
    with torch.no_grad():
        net.weight.add_(1)
    with pytest.raises(AssertionError, match='Live weights differ'):
        module.validate(net, {}, (), 5, checkpoint, _Budget())
    result = json.loads((out / 'validation_epoch005.json').read_text())
    assert result['status'] == 'INCOMPLETE' and result['targets_scored'] == 0
    assert not (cache / 'predictions/epoch005.pt').exists()


def test_score_rejects_missing_prediction_artifact(synthetic_setup, monkeypatch):
    net, checkpoint, rows, records, out, cache = synthetic_setup
    monkeypatch.setattr(module, 'truths', lambda split: pytest.fail('GT read with missing prediction cache'))
    info = dict(all_predictions_completed_before_GT_scoring=True, targets=998,
                checkpoint_SHA=checkpoint['sha256'], path='missing.pt', sha256='missing')
    with pytest.raises(AssertionError):
        module.score(torch.zeros(20, 998, 12, 2), torch.zeros(20, 998, 2), records, 1, checkpoint, info)
