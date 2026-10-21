"""Evidence-bound reporting refinements; primary frozen classifications unchanged."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd,numpy as np

def main():
    d=json.loads((OUT/'final_decision.json').read_text());s=json.loads((OUT/'threshold_support_details.json').read_text());z=pd.read_csv(OUT/'threshold_support_details.csv');q=z[z.actual_sampling_mass_zero];g=pd.read_csv(OUT/'motion_mechanism_summary.csv');g=g[g.radius==25];soft=pd.read_csv(OUT/'softargmax_kmeans_comparison.csv')
    rows=[]
    for factor,levels in [('hist_speed_group',['low','high']),('hist_turn_group',['straight','turning_ge30deg']),('hist_speed_change_group',['stable_CVle0.15','variable_CVge0.4'])]:
        for label in levels:
            a=g[(g.factor==factor)&(g.group==label)].iloc[0]
            rows.append(dict(factor=factor,group=label,targets=int(a.targets),compression_loss_fraction=float(a.compression_loss_fraction),support_zero_fraction=float(a.support_zero_count/a.targets),final_miss_fraction=float(1-a.final_cover_fraction)))
    pd.DataFrame(rows).to_csv(OUT/'history_motion_contrasts.csv',index=False)
    claims=[dict(claim='Original generation and all998 reverse outputs preserved',status='SUPPORTED',evidence='All tensor bitwise and five original SHA checks'),
        dict(claim='Original proposals hit25px but final goals lose coverage',status='SUPPORTED',evidence=f"{d['compression_loss_count']} targets"),
        dict(claim='One dominant bottleneck generalizes consistently across recordings',status='PARTIALLY_SUPPORTED',evidence='Pooled86 compression/77 zero support; deathCircle67 zero/65compression vs other4 10zero/21compression; mechanism inconclusive'),
        dict(claim='Finite random draws with positive support are the main observed failure source',status='NOT_SUPPORTED',evidence='Only1/164 final misses has positive mass but no proposal hit'),
        dict(claim='Semantic unreliability caused goal distribution errors',status='UNVERIFIED',evidence='No semantic ground truth or controlled semantic intervention'),
        dict(claim='Softargmax failures are proven to be caused by multimodal averaging',status='UNVERIFIED',evidence='No preregistered mode identification/intervention; softargmax uses distinct logit exponential weights'),
        dict(claim='Current evidence is sufficient to launch a novel algorithm',status='NOT_SUPPORTED',evidence='Mixed recording mechanisms; prior-art equivalence and new intervention benefits not established')]
    pd.DataFrame(claims).to_csv(OUT/'scientific_claim_status.csv',index=False)
    note=f"\n固定阈值支持细分：77个实际25px零质量目标中，原始Sigmoid邻域权重和>0且被原相对阈值切除的为{s['raw_positive_neighborhood_removed_by_threshold']}个，原Sigmoid邻域权重和为0的为{s['raw_sigmoid_neighborhood_zero']}个。前者未阈值化Sigmoid的数学归一化邻域质量中位数{q.unthresholded_sigmoid_normalized_mass25.median():.8g}，范围{q.unthresholded_sigmoid_normalized_mass25.min():.8g}—{q.unthresholded_sigmoid_normalized_mass25.max():.8g}。这些是已记录张量的静态分解，没有更改阈值、重采样或计算新候选成绩；不能据此推断降低阈值/修改Goal网络的收益。详见threshold_support_details.csv/json。\n"
    R=ROOT/'reports/stage11b2c'
    for file in ['02_PROBABILITY_MASS_ANALYSIS.md','STAGE11B2C_REPORT.md']:
        with (R/file).open('a') as f:f.write(note)
    text='\n历史组实际对照（25px，全部为既有历史分组，非因果）：\n'
    for r in rows:text+=f"\n- {r['factor']} / {r['group']}（n={r['targets']}）：抽样命中后最终失覆盖率{r['compression_loss_fraction']:.4%}；实际支持为0率{r['support_zero_fraction']:.4%}；最终失败率{r['final_miss_fraction']:.4%}。"
    text+='\n\n速度high/low是来源训练历史对角线归一化速度的冻结三分位组，不是绝对米/秒。上述群体有录像构成/重复窗口混杂，不构成独立因果或统计显著性判断。具体数值见history_motion_contrasts.csv。\n'
    with (R/'05_SCENE_AND_MOTION_ANALYSIS.md').open('a') as f:f.write(text)
    with (R/'06_RESEARCH_DECISION.md').open('a') as f:f.write(text+'\n明确的NOT_SUPPORTED判断：只有1/164最终失败属于“正质量但原始抽样未命中”，当前证据不支持随机抽样次数不足是主要失效来源；新算法直接启动条件也未得到支持。语义因果及多峰平均致错保持UNVERIFIED，而非已被否定。全部主张状态见scientific_claim_status.csv。\n')
    for file in ['04_KMEANS_COMPRESSION_ANALYSIS.md','STAGE11B2C_REPORT.md']:
        with (R/file).open('a') as f:f.write(f'\nSoftargmax的作用依赖半径：10px下独立补救KMeans19遗漏{int(soft.SoftRescuesKMeans10.sum())}目标（合计509 vs495），25px/50px下独立补救均为0。不能用25px无新增覆盖推断它在所有精度要求下无作用；1/19/20候选来源比较不是公平重训练实验。\n')
    with (R/'01_GENERATION_PIPELINE_AUDIT.md').open('a') as f:f.write('\n原trace的probability_SHA命名沿用历史，它对应Sigmoid权重图而非全局概率。原阶段没有单独保存逐场景语义张量SHA，本阶段记录新语义缓存SHA，并以同一RGB/分割权重/预处理源码、完整998预测中间SHA及语义重算逐位一致共同核验输入一致性，不虚构不存在的旧语义缓存SHA。\n')
    jnew(OUT/'reporting_addendum.json',dict(source_SHA=sha(Path(__file__)),classification_thresholds_unchanged=True,primary_recommendation=d['recommendation'],no_sampling_or_model_change=True,raw_support_summary=s,history_contrasts=rows))
    print('Reporting addendum complete; primary frozen decision unchanged',flush=True)
if __name__=='__main__':main()
