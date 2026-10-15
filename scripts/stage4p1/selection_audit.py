"""Posthoc description from saved curves; never changes checkpoint selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv
from src.training.social_diagnostics import csv_new


def main():
    root=Path('results/tables/stage4p1');rows=[]
    for kind in ('Full','NoSocial-ConsistentNorm'):
        curve=list(csv.DictReader((root/kind/'curve_30.csv').open()))
        for label,key in [('preregistered_NLL','validation_NLL'),('posthoc_min_K1_ADE_descriptive_only','validation_K1_distribution_mean_ADE')]:
            r=min(curve,key=lambda r:float(r[key]))
            rows.append(dict(model=kind,rule=label,epoch=int(r['epoch']),NLL=float(r['validation_NLL']),
                K1_mean_ADE=float(r['validation_K1_distribution_mean_ADE']),K1_mean_FDE=float(r['validation_K1_distribution_mean_FDE']),
                checkpoint_changed=False,new_evaluation=False,used_for_main_pair_comparison=label=='preregistered_NLL'))
    csv_new(root/'selection_rule_diagnostic.csv',rows)
    table='\n'.join(['| 模型 | 规则 | epoch | NLL | K1均值ADE/FDE |','| --- | --- | --- | --- | --- |']+
        [f"| {r['model']} | {r['rule']} | {r['epoch']} | {r['NLL']:.6f} | {r['K1_mean_ADE']:.6f} / {r['K1_mean_FDE']:.6f} |" for r in rows])
    text='''

## 选择规则敏感性：事后描述，不改变正式checkpoint

从已记录的30轮曲线可见，Full最低K1均值ADE在第29轮：0.690975/1.317263米；预设NLL则选择第30轮：0.788508/1.540245米。NoSocial两种规则均指向第30轮：0.759140/1.430127米。因此**若事后按ADE挑轮次，K1相对排序会反转**。没有重新训练、没有重新评价第29轮、没有替换正式best或配对分析；下面只抄录原曲线，不能当作另一项预注册独立验证。

'''+table+'''

主结果“Full未超过NoSocial”仅适用于本阶段**预设NLL选择的checkpoint**，不能推广为所有轮次或所有选择策略均无社会收益。结果敏感性、单录像单seed以及归一化/核几何混杂，共同使当前结论应为“独立社会图价值尚未建立”，而不是“社会信息无价值”。NLL与累积轨迹误差的目标差异、Full训练顺序对running统计的影响都是待验证解释，未运行反事实，不能写成确定机制。

下一阶段应先固定与研究主指标一致的checkpoint规则，并审查规范化/优化波动，再做真正单因素图消融；不在本轮事后改规则制造优势。上述描述性低值没有替代正式NLL选择和所有失败记录。
'''
    for name in ('02_PUBLIC_BASELINE_TRAINING.md','04_SOCIAL_ABLATION.md','05_RESEARCH_DECISION.md','STAGE4P1_REPORT.md'):
        path=Path('reports/stage4p1')/name
        assert '## 选择规则敏感性：事后描述' not in path.read_text()
        with path.open('a') as stream:stream.write(text)
    print(table)


if __name__=='__main__':main()
