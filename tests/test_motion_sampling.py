import numpy as np
import pytest
from src.motion_sampling.policy import classify_history, probabilities, epoch_order, exposure, decision


def history(speed):
    return np.column_stack((np.arange(8)*speed*.4, np.zeros(8)))


@pytest.mark.parametrize('speed,group', [(0.,0),(.049,0),(.051,1),(.399,1),(.401,2),(1.799,2),(1.801,3)])
def test_history_only_classification_and_units(speed, group):
    label, measured = classify_history(history(speed))
    assert label == group
    assert measured == pytest.approx(speed)
    assert classify_history(history(speed)+[100.,-50.])[0] == group


def test_probabilities_exact_formula_and_empty_groups():
    labels = np.array([0,0,0,2])
    p = probabilities(labels)
    np.testing.assert_allclose(p, [.125+.5/(2*3)]*3+[.125+.5/2])
    assert p.sum() == pytest.approx(1.)
    np.testing.assert_allclose(probabilities([1]*5), np.full(5,.2))


def test_equal_budget_original_uniform_order_and_rng_isolation():
    labels = np.array([0]*100+[1]*10+[2]*8+[3]*2)
    np.random.seed(41); before = np.random.get_state()
    u = epoch_order(labels,'Uniform',1)
    s = epoch_order(labels,'Stratified',1)
    np.testing.assert_array_equal(u,np.random.default_rng(18).permutation(len(labels)))
    np.testing.assert_array_equal(s,epoch_order(labels,'Stratified',1))
    assert len(u) == len(s) == len(labels) and len(np.unique(u)) == len(labels)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1],after[1]) and before[2:] == after[2:]


def test_exposure_repeats_coverage_and_empty_group():
    counts,rows = exposure([0,0,2,2],[0,0,0,2])
    assert counts.tolist() == [3,0,1,0]
    assert rows[0]['repeated_visits'] == 2 and rows[0]['repeated_target_ID_windows'] == 1
    assert rows[0]['coverage'] == .5 and rows[1]['coverage'] is None
    assert sum(r['visits'] for r in rows) == 4


@pytest.mark.parametrize('h',[np.zeros((7,2)),np.full((8,2),np.nan),np.full((8,2),np.inf)])
def test_invalid_history_rejected(h):
    with pytest.raises(ValueError): classify_history(h)


def test_labels_cannot_include_future_or_neighbors():
    h = history(.8)
    changed_future = np.full((12,2),1e6)
    # The sampling API accepts only [8,2], not a 20-frame packet or neighbor input.
    with pytest.raises(ValueError): classify_history(np.vstack([h,changed_future]))
    assert classify_history(h) == classify_history(h.copy())


def test_decision_requires_all_recordings_weights_and_stops_on_any_failure():
    rows = [dict(recording=r,weighting=w,Uniform_ADE_m=.5,Stratified_ADE_m=.49,
        Uniform_FDE_m=1.,Stratified_FDE_m=1.) for r in ['biwi_hotel','crowds_zara02']
        for w in ['target-window','equal-person-ID']]
    speed = dict(Uniform_signed_bias_mps=-.2,Stratified_signed_bias_mps=-.1)
    assert decision(rows,speed,rows,True)['all_passed']
    rows[-1]['Stratified_ADE_m'] = .498
    assert decision(rows,speed,rows,True)['decision'] == 'stop_current_sampling_hypothesis'
    with pytest.raises(AssertionError): decision(rows[:2],speed,rows,True)
