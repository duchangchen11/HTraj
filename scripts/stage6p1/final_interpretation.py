"""Reporting-only supplement; frozen sampler/model/selection/gates are unchanged."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
ROOT=Path('results/tables/stage6p1');REPORT=Path('reports/stage6p1')


def append(name,marker,text):
    path=REPORT/name;assert path.exists()
    assert marker not in path.read_text()
    with path.open('a') as stream:stream.write('\n'+marker+'\n\n'+text.strip()+'\n')


def main():
    overall=read_csv(ROOT/'analysis/paired_overall_metrics.csv')
    motion=read_csv(ROOT/'analysis/all_motion_group_diagnostics.csv')
    def find(rec,cond,group):
        return next(r for r in motion if r['recording']==rec and r['condition']==cond and r['selection']=='best'
                    and r['group']==group and r['weighting']=='target-window')
    u=find('biwi_hotel','Uniform','speed_normal');s=find('biwi_hotel','Stratified','speed_normal')
    a,b=abs(float(u['speed_bias_mps'])),abs(float(s['speed_bias_mps']))
    ma,mb=float(u['absolute_target_speed_bias_mps']),float(s['absolute_target_speed_bias_mps'])
    static_u=find('biwi_hotel','Uniform','speed_stationary');static_s=find('biwi_hotel','Stratified','speed_stationary')
    text=f'''HOTEL正常行走{u['targets']}窗口/{u['people']}ID，真实未来均速{float(u['truth_mean_speed_mps']):.6f}m/s；U/S预测{float(u['predicted_mean_speed_mps']):.6f}/{float(s['predicted_mean_speed_mps']):.6f}m/s。有符号偏差{float(u['speed_bias_mps']):.6f}→{float(s['speed_bias_mps']):.6f}m/s，绝对组均值偏差{a:.6f}→{b:.6f}，下降{100*(1-b/a):.2f}%。mean(|目标偏差|)也从{ma:.6f}降至{mb:.6f}m/s（下降{100*(1-mb/ma):.2f}%），故该组改善不只是组均值正负抵消；这仍不等于每个目标都改善。

同组ADE/FDE从{float(u['ADE_m']):.6f}/{float(u['FDE_m']):.6f}降至{float(s['ADE_m']):.6f}/{float(s['FDE_m']):.6f}m。但HOTEL历史静止{static_u['targets']}窗口仅{static_u['people']}ID，FDE从{float(static_u['FDE_m']):.6f}升至{float(static_s['FDE_m']):.6f}m，增加{float(static_s['FDE_m'])-float(static_u['FDE_m']):.6f}m，超过max(0.01m,5%)。必须保留这个退化，不能只展示正常行走。

ZARA2窗口与行人权重ADE/FDE门槛均失败；HOTEL静止FDE两权重失败。即使两录像宏平均有某些改善，也不能覆盖单录像失败。结论是固定采样产生条件性改变，未获得预设的跨录像稳健收益；停止当前采样假设，不再改比例/阈值/seed/模型/epoch。ZARA2-U的best=30，结论受固定预算限制；没有证明更长训练必然无效，也没有授权延长。'''
    append('04_MOTION_ERROR_ANALYSIS.md','## 实际运动改善与失败代价',text)
    append('06_RESEARCH_DECISION.md','## 对应实际结果的解释',text)
    append('STAGE6P1_REPORT.md','## 核心速度结果与停止原因',text)
    actual=read_csv(ROOT/'actual_class_sampling_summary.csv')
    rows=['| 留出录像 | 条件 | 类别 | 目标窗口/来源ID | 实际30轮访问 | 每目标平均访问 | 每轮平均覆盖率 | 实际类别占比 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in actual:
        rows.append('| '+' | '.join([r['recording'],r['condition'],r['category'],r['pool_windows']+'/'+r['pool_person_IDs'],
            r['total_visits'],f"{float(r['actual_mean_visits_per_window']):.3f}",f"{float(r['mean_epoch_coverage']):.6f}",f"{float(r['mean_epoch_sample_fraction']):.6f}"])+' |')
    exposure='\n'.join(rows)+'''

两个S累计30轮均覆盖全部原目标窗口，但每轮约57%的独特覆盖仍与U的100%不同；正常类每目标约21次而非U的30次。fast只有30/18窗口，涉及13/7来源ID，在S中每窗口累计约3405/4932次，U只有30次。固定混合并非“增加正常行走样本”：正常类比例下降、稀有高速比例大增，多个训练构成变化共同发生。重复曝光、优化噪声、状态比例和模型选择时间都可能影响结果，未独立证明某一项是原因；不事后另试比例。recording-person ID可跨历史类别出现，不能把各类别ID数相加当独立行人数。

实际类别计数/覆盖/重复见actual_class_sampling_summary.csv，每ID曝光见actual_source_person_exposure.csv.gz，全部逐epoch访问计数含零访问已保存。Uniform前5轮与旧同轮ADE/FDE一致，仅作实现复现检查；本次比较完整使用新30轮U，没有以旧100轮替代。
'''
    append('03_TRAINING_RESULTS.md','## 实际类别曝光与稀有目标',exposure)
    append('05_PAIRED_COMPARISON.md','## 条件性收益的边界',text)
    json_new(ROOT/'interpretation_values.json',dict(HOTEL_normal_absolute_mean_bias_reduction=1-b/a,
        HOTEL_normal_mean_absolute_target_bias_reduction=1-mb/ma,
        HOTEL_stationary_FDE_regression_m=float(static_s['FDE_m'])-float(static_u['FDE_m']),
        sampling_rule_changed=False,selection_changed=False,decision_changed=False,reporting_only=True))
    print('Actual conditional speed benefit and failed safety/generalization gates documented')


if __name__=='__main__':main()
