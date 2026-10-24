"""All fifteen paired state changes; no favorable-case filtering."""
from .common import *
import pandas as pd,numpy as np

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    df=pd.read_csv(OUT/'paired_runtime_by_target.csv')
    a=df[df.state=='NoSemantic-initial'].set_index('sample_id')
    b=df[df.state=='NoSemantic-epoch001'].set_index('sample_id')
    assert set(a.index)==set(b.index) and len(a)==15
    rows=[]
    for sid in sorted(a.index):
        row=dict(sample_id=sid,scene=a.loc[sid,'scene'])
        for key in ['iterations','wall_seconds','outer_prediction_seconds']:
            x=float(a.loc[sid,key]);y=float(b.loc[sid,key])
            row.update({key+'_initial':x,key+'_epoch1':y,key+'_delta_epoch1_minus_initial':y-x})
        rows.append(row)
    cnew(OUT/'nosemantic_state_paired_differences.csv',rows)
    changes={}
    for key in ['iterations','wall_seconds','outer_prediction_seconds']:
        d=np.array([r[key+'_delta_epoch1_minus_initial'] for r in rows])
        changes[key]=dict(lower=int((d<0).sum()),equal=int((d==0).sum()),higher=int((d>0).sum()),
            mean_delta=float(d.mean()),median_delta=float(np.median(d)))
    jnew(OUT/'paired_runtime_change.json',dict(targets=15,same_IDs=True,differences=changes,GT_used=False,
        source_SHA=sha(OUT/'paired_runtime_by_target.csv'),analysis_code_SHA=sha(Path(__file__))))
if __name__=='__main__':main()
