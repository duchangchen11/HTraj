"""Clarify numerical size, endpoint conflicts, group exceptions and no causality."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from scripts.stage5p3.write_reports import ROOT,DOC,REC,LABEL,table,n

effects=read_csv(ROOT/'social_effects_all_groups.csv')
overall=json.load(open(ROOT/'social_effects_overall.json'))
macro=read_csv(ROOT/'social_effects_recording_macro.csv')

def append(name,text):
    path=DOC/name;marker='## 主结果的证据边界核对'
    assert marker not in path.read_text()
    with path.open('a',encoding='utf-8') as f:f.write('\n'+marker+'\n\n'+text.strip()+'\n')

text='''主K1 best规则下Full在两个录像及两种权重均达到预设继续核查门槛。窗口ADE优势仅ZARA2约6.3mm（2.26%）、HOTEL约8.0mm（2.03%）；这是开发结果，不是显著性结论。Full相对NoSocial的优势不能排除585792个结构有效参数上界差异，不能直接命名为社会信息收益。

K20窗口口径：HOTEL Full优于NoSocial（0.2025/0.3253 vs 0.2461/0.4166m），ZARA2 Full反而较差（0.1575/0.3048 vs 0.1551/0.2942m）。ZARA2行人等权K20 ADE略有Full优势、FDE略退化；这与整体K1同方向收益不是同一结论。单首采样也有权重/录像冲突，均已保留。

正常行走问题：ZARA2无需邻居时仍保留窗口口径约74.8%、行人等权约58.9%的相对Last2改善；Full只比NoSocial进一步降低ADE约0.0210/0.0401m。该比例是描述性差值，不是社会因果贡献份额。HOTEL正常组Full 0.7992m、NoSocial 0.8925m都弱于Last2 0.5333m；移除社会输入未消除退化，故不能将原退化简单归因为社会分支，Full在该组反而较小。这也不证明个体主干/后验目标的具体失效原因，仍需独立核查。

同一目标下仍有退化组：Full历史静止组相对NoSocial在HOTEL/ZARA2的ADE分别增加0.0689/0.0037m；HOTEL无合法邻居组增加0.0115m；ZARA2新进入邻居及速度不可观测组增加0.0073/0.0107m。低速组两录像ADE小改善、FDE却小退化。不能只展示正常行走组的成功案例。HOTEL静止组106窗口只有4个ID，正常组91窗口覆盖24个ID，因而窗口与行人等权强调的对象不同；模型间目标集合本身完全相同。

按两个录像各1/2的宏平均，best窗口K1为Full 0.3299/0.6795 vs NoSocial 0.3370/0.6956m；Full仍不如Last2宏0.3129/0.6393m。last100时ZARA2 NoSocial K1明显好于Full，而HOTEL Full较好，说明主best规则结论不能推广到所有训练终点。last宏FDE也出现Full小退化。全部曲线/终点已保留，没有反向修改选择规则。

下一步仅值得先核查容量/训练泛化，不能据此增加GNN、高阶、关系GRU、注意力或门控。Ego-Capacity-Control的提案不是已实现/已复现结果；其额外ego路径改变归纳偏置，若未来实施仍需独立结构与梯度测试。在性能容量混杂未排除、无额外seed/独立确认时，不确认第四章新方法路线或社会因果收益。本阶段停止，不自动追加训练。
'''
append('06_RESEARCH_DECISION.md',text)
append('STAGE5P3_REPORT.md',text)
append('05_INTERACTION_GROUP_ANALYSIS.md',text)
j=json.load(open(ROOT/'numerical_input_jacobian.json'))
assert all(r['max_abs_central_difference_jacobian']==0 for r in j if r['model']=='NoSocial')
rows=[]
for rec in REC:
    for model in ['Full','NoSocial']:
        rr=[r for r in j if r['recording']==rec and r['model']==model]
        rows.append([LABEL[rec],model,len(rr),max(r['max_abs_central_difference_jacobian'] for r in rr),max(r['other_target_max_abs_jacobian'] for r in rr)])
append('02_NOSOCIAL_IMPLEMENTATION.md','固定步长0.01m、历史距离<1.9m的3个源位置坐标，中心差分Jacobian结果如下。选择坐标只依赖历史，不用未来GT。Full原始autograd虽然detach，数值差分仍有非零响应；NoSocial严格为0，其他batch目标也无依赖。此小规模功能诊断不能当作全部目标敏感性上界或预测收益证明。\n\n'+table(['录像','模型','坐标数','最大数值Jacobian','其他batch目标Jacobian'],rows))
