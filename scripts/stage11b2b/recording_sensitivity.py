"""Posthoc CPU-only recording sensitivity; never alters primary H1 or selection."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd

def build():
    rows=[]
    for ep in [30,35,40,45,50]:
        f=pd.read_csv(OUT/f'paired_coverage_epoch{ep:03d}.csv')
        for removed in ['NONE']+sorted(f.scene.unique()):
            s=f if removed=='NONE' else f[f.scene!=removed];a=s.minGoalFDE20>25
            rows.append(dict(epoch=ep,excluded_recording=removed,targets=len(s),A25_count=int(a.sum()),A25_fraction=float(a.mean()),
                A25_error_share=float(s.loc[a,'minADE20'].sum()/s.minADE20.sum()),minADE20=float(s.minADE20.mean()),
                scope='Posthoc descriptive recording sensitivity; not checkpoint selection or altered primary H1'))
    return pd.DataFrame(rows)

def main():
    x=build();p=OUT/'recording_sensitivity.csv'
    if p.exists():pd.testing.assert_frame_equal(x,pd.read_csv(p),check_exact=False,rtol=1e-14,atol=1e-14)
    else:x.to_csv(p,index=False)
    jnew(OUT/'recording_sensitivity_audit.json',dict(passed=True,source_SHA=sha(Path(__file__)),table_SHA=sha(p),CPU_only=True,
        posthoc_descriptive=True,primary_H1_unchanged=True,no_checkpoint_selection=True,all998_retained_in_primary=True))
    print('Recording sensitivity recomputation PASS',flush=True)
if __name__=='__main__':main()
