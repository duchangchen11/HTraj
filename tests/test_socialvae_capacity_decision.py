import copy
import pytest
from src.socialvae_capacity.decision import classify
RULE=dict(weightings=['target-window','equal-person-ID'],FDE_substantial_regression_relative=.01,Full_meaningful_ADE_gain_relative=.01)
def cells():
 return [dict(recording=r,weighting=w,Full_ADE_m=.3,Full_FDE_m=.6,EgoCapacity_ADE_m=.32,EgoCapacity_FDE_m=.62) for r in ['biwi_hotel','crowds_zara02'] for w in RULE['weightings']]
def test_two_recordings_two_weightings_required():
 c=cells();assert classify(c,RULE)['phase2_allowed']
 # Window-weighted wins cannot overrule one person-weighted reversal.
 c[1]['EgoCapacity_ADE_m']=.29
 assert classify(c,RULE)['verdict']=='conditional_or_weak_evidence_stop'
 with pytest.raises(ValueError):classify(c[:3],RULE)
def test_capacity_sufficient_without_requiring_identical_fde():
 c=cells()
 for r in c:r.update(EgoCapacity_ADE_m=.29,EgoCapacity_FDE_m=.604)
 assert classify(c,RULE)['verdict']=='ego_capacity_sufficient_stop'
 assert not classify(c,RULE)['phase2_allowed']
def test_full_ade_gain_cannot_overrule_fde_failure():
 c=cells();c[0]['Full_FDE_m']=.8
 assert not classify(c,RULE)['phase2_allowed']
def test_missing_or_nonfinite_cells_cannot_authorize_training():
 c=cells();c[0]['Full_ADE_m']=float('nan')
 with pytest.raises(ValueError):classify(c,RULE)
 c=cells();c[-1]=copy.deepcopy(c[0])
 with pytest.raises(ValueError):classify(c,RULE)
