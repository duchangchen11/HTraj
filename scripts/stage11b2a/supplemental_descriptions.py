"""Recording paired-error contributions explicitly requested by the protocol."""
from .common import *
from .report import table
import pandas as pd
def main():
    cpu_guard();f=pd.read_csv(OUT/'motion_features.csv');rows=[]
    for scene,s in f.groupby('scene',sort=True):
        a=s.ADE1-s['OLS8_ADE1'];b=s.FDE1-s['OLS8_FDE1']
        rows.append(dict(scene=scene,targets=len(s),cohort_weight=len(s)/998,
            mean_ADE1_delta=float(a.mean()),weighted_ADE1_contribution=float(a.sum()/998),
            mean_FDE1_delta=float(b.mean()),weighted_FDE1_contribution=float(b.sum()/998),
            ADE_targets_improved=int((a<0).sum()),ADE_targets_worsened=int((a>0).sum())))
    r=pd.DataFrame(rows);r.to_csv(OUT/'recording_paired_contributions.csv',index=False)
    p=REPORT/'05_SCENE_HETEROGENEITY.md'
    with p.open('a') as stream:stream.write('\n## 录像对整体配对差值的贡献\n\n'+
        table(r,['scene','targets','mean_ADE1_delta','weighted_ADE1_contribution','mean_FDE1_delta','weighted_FDE1_contribution'])+
        '\n\nweighted contribution=该录像全部配对差值之和/998；各录像贡献之和恢复完整总体差值，不是删除录像后的新成绩。负值贡献改善，正值贡献退化。\n')
    assert abs(r.weighted_ADE1_contribution.sum()-(f.ADE1-f['OLS8_ADE1']).mean())<1e-10
    cpu_guard();print('Paired recording contributions saved; no new threshold or model analysis')
if __name__=='__main__':main()
