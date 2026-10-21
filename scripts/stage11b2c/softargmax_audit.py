"""Ancillary offline audit of existing SoftArgmax2D formula, no mode classifier.

Original epsilon=1e-6 is observed from frozen source, never changed. This
supplement does not alter any preregistered event, radius, case, or decision.
"""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from .math_audit import probability
import numpy as np,pandas as pd,torch

def main():
    assert not torch.cuda.is_initialized() and json.loads((OUT/'cache_integrity.json').read_text())['passed']
    rows=[]
    for r in pd.read_csv(OUT/'tensor_manifest.csv').to_dict('records'):
        a=torch.load(ROOT/r['path'],map_location='cpu',weights_only=True);x=a['tensors'];z=x['logits'][0,0].numpy().astype(float);H,W=z.shape;yy,xx=np.indices((H,W));ex=np.exp(z-z.max());S=ex.sum();q=ex/S;qeps=ex/(S+1e-6);p=probability(x['normalized_weights'].reshape(H,W).numpy());soft=x['softargmax_raw'].numpy().astype(float)
        expected=np.array([(xx*qeps).sum(),(yy*qeps).sum()])/.25
        pmean=np.array([(xx*p).sum(),(yy*p).sum()])/.25
        mask=(xx/.25-soft[0])**2+(yy/.25-soft[1])**2<=25**2
        positive=q[q>0]
        rows.append(dict(sample_id=r['sample_id'],scene=r['scene'],native_softargmax_x=soft[0],native_softargmax_y=soft[1],float64_epsilon_formula_x=expected[0],float64_epsilon_formula_y=expected[1],formula_coordinate_difference_px=float(np.linalg.norm(expected-soft)),
            epsilon=1e-6,qeps_total_mass=float(qeps.sum()),sampling_distribution_mean_x=pmean[0],sampling_distribution_mean_y=pmean[1],sampling_mean_to_native_softargmax_px=float(np.linalg.norm(pmean-soft)),
            total_variation_sampling_vs_normalized_logit_softmax=float(np.abs(p-q).sum()/2),sampling_mass_within25_of_softargmax=float(p[mask].sum()),softmax_mass_within25_of_softargmax=float(q[mask].sum()),
            softmax_entropy_nats=float(-np.sum(positive*np.log(positive))),mode_averaging_cause='UNVERIFIED; no preregistered mode-count/intervention test'))
    pd.DataFrame(rows).to_csv(OUT/'softargmax_formula_audit.csv',index=False)
    jnew(OUT/'softargmax_formula_audit.json',dict(passed=True,scope='Ancillary static-formula/float64 descriptive check; not bitwise recomputation of GPU reduction',targets=len(rows),
        source_SHA=sha(Path(__file__)),native_formula_source='experiments/stage11b0/cache/official/ynet/utils/softargmax.py',
        native_source_SHA=sha(ROOT/'experiments/stage11b0/cache/official/ynet/utils/softargmax.py'),epsilon_unchanged=1e-6,classification_thresholds_unchanged=True,
        max_float64_native_coordinate_difference_px=max(r['formula_coordinate_difference_px'] for r in rows),mean_float64_native_coordinate_difference_px=float(np.mean([r['formula_coordinate_difference_px'] for r in rows])),
        sampling_and_softargmax_weights_not_identical_definition=True,multimodal_averaging_failure='UNVERIFIED'))
    print('Native softargmax formula audit complete; no mechanism threshold or model changed',flush=True)
if __name__=='__main__':main()
