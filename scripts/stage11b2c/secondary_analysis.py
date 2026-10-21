"""Descriptive refinements only; no new event thresholds or method selection."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd,numpy as np

def main():
    assert json.loads((OUT/'cache_integrity.json').read_text())['passed']
    p=pd.read_csv(OUT/'probability_mass.csv');f=pd.read_csv(OUT/'sample_level_mechanism.csv');e=pd.read_csv(OUT/'coverage_transition.csv')
    z=p[['sample_id','scene','sigmoid_sum','P_mass25','sigmoid_weight_sum25','threshold_weight_sum25']].copy()
    z['unthresholded_sigmoid_normalized_mass25']=z.sigmoid_weight_sum25/z.sigmoid_sum
    z['actual_sampling_mass_zero']=z.P_mass25==0
    z['raw_positive_neighborhood_removed_by_threshold']=(z.sigmoid_weight_sum25>0)&(z.threshold_weight_sum25==0)
    z['raw_sigmoid_neighborhood_zero']=z.sigmoid_weight_sum25==0
    z.to_csv(OUT/'threshold_support_details.csv',index=False)
    n=e.copy()
    for k in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','SelectionGap_ADE']:n[k+'_diag']=n[k]/n.reference_diagonal
    n.to_csv(OUT/'normalized_mechanism_errors.csv.gz',index=False)
    cols=[k+'_diag' for k in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','SelectionGap_ADE']]
    n.groupby(['radius','scene','category'])[cols].mean().reset_index().to_csv(OUT/'scene_category_normalized_errors.csv',index=False)
    s=z[z.actual_sampling_mass_zero];summary=dict(actual_25px_zero_mass_targets=len(s),raw_positive_neighborhood_removed_by_threshold=int(s.raw_positive_neighborhood_removed_by_threshold.sum()),raw_sigmoid_neighborhood_zero=int(s.raw_sigmoid_neighborhood_zero.sum()),
        scope='Ancillary decomposition of already frozen zero-support event; no threshold changed, no counterfactual resampling or causal claim',source_SHA=sha(Path(__file__)))
    jnew(OUT/'threshold_support_details.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
