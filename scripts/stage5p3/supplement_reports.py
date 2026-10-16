"""Actual capacity, trained-isolation and the two specified motion questions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new
from scripts.stage5p3.write_reports import n,table,ROOT,DOC,REC,LABEL


def append(name,text):
    path=DOC/name;marker='## 最终实测补充'
    assert marker not in path.read_text()
    with path.open('a',encoding='utf-8') as f:f.write('\n'+marker+'\n\n'+text.strip()+'\n')


metrics=read_csv(ROOT/'all_group_metrics.csv');normal=[]
for rec in REC:
    for weight in ['target-window','equal-person-ID']:
        get=lambda model,sel,family:next(r for r in metrics if (r['recording'],r['model'],r['selection'],r['metric_family'],r['group'],r['weighting'])==(rec,model,sel,family,'speed_normal',weight))
        cv=get('CV-Last2','analytic','deterministic');full=get('Full','best','K1-prior');ns=get('NoSocial','best','K1-prior')
        c,f,e=[float(r['ADE_m']) for r in [cv,full,ns]]
        normal.append(dict(recording=rec,weighting=weight,target_windows=int(full['target_windows']),people=int(full['people']),
            CV_Last2_ADE_m=c,Full_ADE_m=f,NoSocial_ADE_m=e,Full_minus_NoSocial_ADE_m=f-e,
            Full_improvement_vs_CV_m=c-f,NoSocial_improvement_vs_CV_m=c-e,
            descriptive_fraction_retained_without_neighbors=(c-e)/(c-f) if c-f>0 else None,
            interpretation='descriptive numeric ratio, NOT causal attribution fraction'))
csv_new(ROOT/'normal_motion_decomposition.csv',normal)
normaltext=table(['录像','权重','窗口/ID','CV','Full','NoSocial','Full−NoSocial','无社会保留改善比例'],[
    [LABEL[r['recording']],r['weighting'],str(r['target_windows'])+'/'+str(r['people']),n(r['CV_Last2_ADE_m']),n(r['Full_ADE_m']),n(r['NoSocial_ADE_m']),n(r['Full_minus_NoSocial_ADE_m']),n(r['descriptive_fraction_retained_without_neighbors'])] for r in normal])
normaltext+='\n主参考固定CV-Last2，三个CV的全部组误差也已保存。比例只在Full相对CV确有改善时定义；它只是数值描述，不是“社会贡献百分比”。无邻居组的跨模型差异可以来自不同训练后权重，不能等同固定checkpoint仍接收消息。仅某组改善不支持跨场景新机制。'
append('05_INTERACTION_GROUP_ANALYSIS.md',normaltext)
append('06_RESEARCH_DECISION.md',normaltext)
life=[]
for rec in REC:
    rr=read_csv(ROOT/rec/'lifetime_parameter_gradients.csv')
    life.append(dict(recording=rec,total_parameters=sum(int(r['nominal']) for r in rr),
        ever_nonzero_gradient_scalars=sum(int(r['ever_nonzero_gradient_scalars']) for r in rr),
        never_nonzero_gradient_scalars=sum(int(r['never_nonzero_gradient_scalars']) for r in rr),
        structural_upper_bound=sum(int(r['structural_effective_upper_bound']) for r in rr),
        all_dead_weights_unchanged=all(r['dead_parameter_values_unchanged']=='True' for r in rr)))
json_new(ROOT/'lifetime_capacity_summary.json',life)
append('03_CAPACITY_AUDIT.md',table(['录像','名义参数','全程非零梯度并集','全程从未非零','结构有效上界','死参数保持初值'],[[LABEL[r['recording']],r['total_parameters'],r['ever_nonzero_gradient_scalars'],r['never_nonzero_gradient_scalars'],r['structural_upper_bound'],r['all_dead_weights_unchanged']] for r in life])+'\n这是100轮全部训练步骤的实测。不能将NoSocial的精确并集与Full的Adam二阶矩代理视为同一种统计。Ego-Capacity-Control仍未实现/未训练，性能容量混杂仍未排除。')
trained=json.load(open(ROOT/'trained_isolation_audit.json'))
nsrows=[r for r in trained if r['model']=='NoSocial']
assert all(r['strictly_equal'] for r in nsrows)
probe=[]
for rec in REC:
    for mode in ['K1','K20']:
        rr=[r for r in trained if r['recording']==rec and r['model']=='Full' and r['mode']==mode and r['intervention']!='permutation']
        probe.append([LABEL[rec],mode,max(r['max_abs_output_difference'] for r in rr)])
text='训练后best权重的50项NoSocial检查（含两录像、K1/K20/后验/loss及Jacobian）全部严格不变。原始输出见`trained_isolation_audit.json`和日志。Full相同固定目标与噪声的功能干预保留输入敏感性，示例最大坐标变化如下（不同干预取最大，仅用于功能诊断，不是预测误差）：\n\n'+table(['录像','模式','Full最大坐标变化/m'],[[r[0],r[1],n(r[2],6)] for r in probe])+'\n位置平移、速度feature修改、复制候选等是人工接口干预，不应称为同分布预测负对照或公平重训练结果。Full原始Jacobian为空仍源于官方特征detachment；NoSocial还同时有源码无读取和精确输入干预证据。'
append('02_NOSOCIAL_IMPLEMENTATION.md',text)
append('STAGE5P3_REPORT.md','严格NoSocial的训练后输入隔离检查全部通过；历史SHA及正式训练验收见`delivery_verification.json`。容量控制未训练，社会收益不能独立归因。正常行走的指定问题已在05/06列出同目标CV/Full/NoSocial及两种权重，不能将相对CV收益直接当社会贡献。')
append('01_SOCIAL_INFORMATION_PATH_AUDIT.md','忠实记录官方几何feature公式：`bearing=<dp,v>/(||dp|| ||v||)`（非有限值归零）；`tau=clip(-<dp,dv>/||dv||,0,7)`（非有限值归零）；`mpd=||dp+tau*dv||`。代码的tau分母是范数而非平方范数，不能擅自写成常规最近相遇时间公式；本阶段不更改它，其物理含义和单位解释若需改动须另行核查。NoSocial不计算上述邻居feature。')
