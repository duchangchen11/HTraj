import numpy as np
import pytest
from scripts.stage11b1.admission import training_permitted
from scripts.stage11b1.baselines import predict

@pytest.mark.parametrize('method',['CV-Last2','OLS4','OLS8'])
def test_linear_motion_real_dt_and_stationary(method):
    h=np.array([5.,13.])+np.arange(8)[:,None]*np.array([1.2,-.8])*.4
    expected=h[-1]+np.arange(1,13)[:,None]*np.array([1.2,-.8])*.4
    np.testing.assert_allclose(predict(h,method),expected,rtol=0,atol=1e-13)
    np.testing.assert_allclose(predict(np.tile([2.,3.],(8,1)),method),np.tile([2.,3.],(12,1)))

def test_baseline_input_validation():
    with pytest.raises(ValueError):predict(np.zeros((20,2)),'OLS8')
    with pytest.raises(ValueError):predict(np.full((8,2),np.nan),'OLS8')
    with pytest.raises(ValueError):predict(np.zeros((8,2)),'unknown')

def test_timing_or_data_pass_cannot_override_reproducibility_failure():
    budget={'thirty_epochs_plus_seven_validations_fit':True}
    assert not training_permitted(True,True,{'formal_training_gate_passed':False},budget)
    assert not training_permitted(False,True,{'formal_training_gate_passed':True},budget)
    assert not training_permitted(True,False,{'formal_training_gate_passed':True},budget)
    assert not training_permitted(True,True,{'formal_training_gate_passed':True},{})
    assert training_permitted(True,True,{'formal_training_gate_passed':True},budget)
