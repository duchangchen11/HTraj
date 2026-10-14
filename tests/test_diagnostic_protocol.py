import json
from pathlib import Path
import torch
from src.evaluation.social_diagnostics import history_groups, GROUPS
from src.training.social_diagnostics import choose_social


def test_grouping_uses_only_history_and_fixed_bin_boundaries():
    position = torch.zeros(1, 8, 3, 2)
    position[:, :, 1, 0] = 1.
    position[:, :, 2, 0] = 100.
    batch = dict(observed=position, observed_valid=torch.ones(1, 8, 3, dtype=torch.bool),
                 timestamps=torch.arange(8)[None].double()*.4, future=torch.randn(1, 12, 3, 2))
    first = history_groups(batch)
    assert set(first) == set(GROUPS)
    assert first['no_neighbors'][0].tolist() == [False, False, True]
    assert first['low_neighbors'][0].tolist() == [True, True, False]
    assert not first['medium_neighbors'].any() and not first['high_neighbors'].any()
    assert first['order2_exists'][0].tolist() == [True, True, False]
    assert first['order3_exists'][0].tolist() == [True, True, False]
    batch['future'] *= 10000
    second = history_groups(batch)
    assert all(torch.equal(first[key], second[key]) for key in GROUPS)


def test_fixed_selection_failure_rule_and_maximum_two_social_models():
    protocol = json.loads(Path('configs/stage2p5/protocol.json').read_text())
    rows = [dict(model='PedBase', best_ADE_m=1., best_FDE_m=2.),
            dict(model='Dynamic-GNN-RN', best_ADE_m=1.1, best_FDE_m=2.2),
            dict(model='TOHI-A-RN', best_ADE_m=1.2, best_FDE_m=2.3)]
    result = choose_social(rows, protocol)
    assert result['selected_social_models'] == []
    rows[1]['best_ADE_m'] = .9
    rows[2]['best_ADE_m'] = .93
    result = choose_social(rows, protocol)
    assert result['selected_social_models'] == ['Dynamic-GNN-RN', 'TOHI-A-RN']


def test_large_subset_is_nested_and_validation_unchanged():
    protocol = json.loads(Path('configs/stage2p5/protocol.json').read_text())
    small = json.loads(Path(protocol['small_subset']).read_text())
    large = json.loads(Path(protocol['large_subset']).read_text())
    assert set(small['train']['indices']) <= set(large['train']['indices'])
    assert len(large['train']['indices']) == 1024
    assert small['validation'] == large['validation']
    assert large['test_used'] is False
