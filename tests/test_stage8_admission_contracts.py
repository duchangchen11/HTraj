import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage8_contracts',ROOT/'scripts/stage8/contracts.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_validation_gate_rejects_heldout_files():
    assert module.require_validation_file('hotel/biwi_eth_val.txt').name=='biwi_eth_val.txt'
    for path in ('hotel/biwi_hotel.txt','univ/students001.txt','eth/biwi_eth_test.txt'):
        with pytest.raises(ValueError):module.require_validation_file(path)


def test_separate_means_erase_position_motion_pairing():
    witness=module.source_motion_geometry_witness()
    assert np.array_equal(witness['unbound'],witness['swapped_unbound'])
    assert witness['joint_moment']!=witness['swapped_joint_moment']


def test_no_go_is_not_overridden_by_engineering_success():
    gates=dict(problem=True,mechanism_difference=False,legal_data=True,fair_controls=True,hardware_cost=True,stop_rule=True)
    assert not module.validate_research_gates(gates)
    gates['mechanism_difference']=True
    assert module.validate_research_gates(gates)


def test_unknown_is_not_a_passing_gate():
    gates=dict(problem=True,mechanism_difference=None,legal_data=True,fair_controls=True,hardware_cost=True,stop_rule=True)
    assert not module.validate_research_gates(gates)
    gates.pop('stop_rule')
    with pytest.raises(ValueError):module.validate_research_gates(gates)


def test_resonance_real_probe_labels_and_outputs_are_explicit():
    result=json.loads((ROOT/'results/tables/stage8/resonance_probe.json').read_text())
    assert result['formal_epochs']==0 and result['optimizer_updates']==0
    assert result['passed']
    assert {r['case'] for r in result['cases']}=={'single','two','multiple','maximum_validation_context'}
    for row in result['cases']:
        assert row['output_shape'][1:]==[20,12,2]
        assert row['neighbor_permutation_max_difference']<1e-5
        assert row['translation_equivariance_max_difference']<1e-5


def test_author_checkpoint_is_strict_and_never_a_benchmark_score():
    result=json.loads((ROOT/'results/tables/stage8/resonance_author_checkpoint.json').read_text())
    assert result['strict_load'] and not result['missing_keys'] and not result['unexpected_keys']
    assert result['checkpoint_unchanged'] and result['ADE_FDE_not_computed']
    assert result['author_args']['output_pred_steps']=='2_5_8_11'


def test_moflow_cpu_probe_does_not_claim_gpu_or_trained_sampling():
    result=json.loads((ROOT/'results/tables/stage8/moflow_cpu_probe.json').read_text())
    assert result['device']=='cpu' and result['GPU_seconds']==0
    assert result['passed'] and not result['formal_training']
    assert result['models'][0]['agents']==1
    assert result['models'][1]['synchronized_agent_permutation_max_difference']>1e-5
