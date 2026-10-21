"""Evidence-linked reports, written only from real completed results."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd,numpy as np
R=ROOT/'reports/stage11b2b'
def table(f,cols):
    f=f[cols];rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in f.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)
def write(name,text):
    with (R/name).open('x') as f:f.write(text.strip()+'\n')

def main():
    d=json.loads((OUT/'final_decision.json').read_text());m=pd.read_csv(OUT/'validation_metrics.csv');sc=pd.read_csv(OUT/'scene_metrics.csv');gr=pd.read_csv(OUT/'motion_group_metrics.csv')
    old=d['epoch30'];fin=d['final'];last=d['final_complete_validation_epoch'];execution=json.loads((OUT/'training_execution.json').read_text());epochs=pd.read_csv(OUT/'epochs.csv')
    gb=d['global_best_metrics'];cm=json.loads((OUT/'checkpoint_manifest.json').read_text());ck=cm['global_best_checkpoint']
    t=d['transitions'];fixed=pd.read_csv(OUT/'a25_fixed_cohort.csv');cstate=pd.read_csv(OUT/'goal_path_mismatch_persistence.csv').iloc[-1];macro=pd.read_csv(OUT/'recording_macro_metrics.csv');common='适用范围：PROTOCOL_DEVELOPMENT_ONLY / CONVERGENCE_DIAGNOSTIC_ONLY。单seed17、5个开发验证录像、998个重叠目标窗口，并非998个独立场景。单位为SDD参考图像像素，不能与ETH/UCY米制排名。分割语义ID映射与预训练成员身份未完全核实，验证可能存在语义预训练曝光。未读取封存测试集，未新增模型、loss、调参或数据增强。'
    audit=json.loads((OUT/'resume_admission.json').read_text());log=json.loads((OUT/'gpu_budget.json').read_text())
    with (R/'01_RESTORE_AUDIT.md').open('a') as f:f.write(f'\n最终恢复结果：`{audit["status"]}`。模型状态SHA `{audit["model_state_SHA"]}`；Adam摘要 `{audit["optimizer_SHA"]}`；RNG摘要 `{audit["RNG_SHA"]}`。下一场景顺序SHA `{audit["next_scene_order_SHA"]}`。实际验证参数量总计53,155,339、训练核心1,641,381；核心无BN/Dropout/RNN模式依赖层，train/eval输出及恢复后下一Adam更新逐位相同，K20重复/反序/缓存重算与百万像素未来GT反事实通过。完整记录见 resume_admission.json。\n')
    write('02_TRAINING_EXECUTION.md',f'''# 训练延长执行

{common}

执行状态 `{execution['status']}`，原Epoch30完整状态恢复后完成Epoch31—{d['final_trained_epoch']}，新增{d['new_optimizer_updates']}次Adam更新，累计global_step={d['cumulative_global_step']}。每完整轮7496目标各一次、1882次更新。新旧checkpoint分目录且不可覆盖；全部新checkpoint包含原Epoch30 SHA、配置、模型/Adam/RNG/进度。

GPU累计进程wall={d['new_GPU_seconds']:.3f}秒（{d['new_GPU_seconds']/3600:.3f}小时），包含失败准入、成功准入、正式延长进程的启动/加载/检查/训练/验证/保存/退出，低于16200秒硬限；15000秒安全阈值前完成。CPU-only回归、统计与绘图不初始化CUDA。整卡资源逐15秒记录GPU_card_*.csv；各步allocated/reserved见training_steps.csv.gz。

训练学习率固定0.0001，无scheduler/AMP/clip/augmentation；预测核心及原损失未改。正式训练期间冻结文件无改动。原模式状态审计见01报告。首次合成GT越界失败是准入fixture问题，正式训练未开始，修订前后日志和Git提交保留，不属于恢复失败或新超参数尝试。

平均完整epoch耗时{epochs.train_seconds.mean():.3f}秒，范围{epochs.train_seconds.min():.3f}—{epochs.train_seconds.max():.3f}秒。实际覆盖检查见training_access_audit.json；全量steps见training_steps.csv.gz；epochs.csv保存所有新轮曲线。验证仅在实际完整节点进行。

{table(m,['epoch','ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20'])}

选择规则：全部旧7次及新增完整验证中最低窗口加权minADE20，完全相同则较早epoch。新最佳Epoch{d['new_best_epoch']}，全体最佳Epoch{d['global_best_epoch']}。最佳checkpoint `{ck['path']}`，SHA `{ck['sha256']}`。最后训练Epoch{d['final_trained_epoch']}、最后完整验证Epoch{last}独立报告，未用Goal覆盖率选模。完整checkpoint_manifest.json保留每个不可变文件及SHA。
''')
    write('03_GOAL_COVERAGE_PERSISTENCE.md',f'''# Goal覆盖持续性

{common}

严格沿用 `minGoalFDE20 > 10/25/50px` 作为失败，不使用>=。所有候选生成完成、身份检查与缓存保存之后才读GT评分。NumPy独立重算六项误差与训练评价器的绝对差不超过1e-8px。

{table(m[m.epoch>=30],['epoch','A10_miss_count','A25_miss_count','A50_miss_count','A25_miss_fraction','A25_error_share','SelectionGap_ADE_mean','SelectionGap_FDE_mean','GoalPathMismatch_count'])}

固定原164目标与动态失败群体分开：固定群体在a25_fixed_cohort.csv；动态阈值群体在goal_coverage_epochs.csv。Epoch30→{last}，恢复{t['recovered']}、新失败{t['new_failures']}、持续失败{t['persistent_failure']}、持续成功{t['persistent_success']}，合计998。a25_transition.csv同时报告相对于Epoch30与上一完整验证节点的转移；target_cohort_transitions.csv保留998目标身份和原7个GOAL_PATH_MISMATCH身份。

A25失败率从{old['A25_miss_fraction']:.6%}变到{fin['A25_miss_fraction']:.6%}；其minADE20误差份额从{old['A25_error_share']:.6%}变到{fin['A25_error_share']:.6%}。目标比例、群体平均误差、总误差份额是三种不同量。阈值群体均值与GoalFDE分布P50/P90/P95及分录像表均保存，未删困难目标。

冻结判据：最终完整新增验证的A25比例>=.10且误差份额>=.50则支持；比例<.05且份额<.25则削弱；其余INCONCLUSIVE。结果 `{d['H1']}`。它仅说明有限预算末端的持续性，不证明充分收敛、不可消除瓶颈或新算法可改进。

SelectionGap_ADE均值{old['SelectionGap_ADE_mean']:.6f}→{fin['SelectionGap_ADE_mean']:.6f}px；FDE均值{old['SelectionGap_FDE_mean']:.6f}→{fin['SelectionGap_FDE_mean']:.6f}px。中位数/P90、第一候选胜出率和ADE/FDE同head率见validation_metrics.csv及scene_metrics.csv。此差距是GT oracle与默认候选的预算差，不直接等于部署排序失败。

同head Goal<=10且Path ADE>=25的目标数量{int(old['GoalPathMismatch_count'])}→{int(fin['GoalPathMismatch_count'])}；全部19960候选每轮同head GoalFDE、Path ADE/FDE及goal-path gap均留档，原7个与新增者逐ID可核查。不能将路径偏离归因为语义。原7个中恢复{int(cstate.original_C7_recovered)}、持续{int(cstate.original_C7_persistent)}、最终新增{int(cstate.new_C_mismatch)}个；附加严格中段定义（同head Goal<=10、Path ADE>=25、Path FDE<=10且前11步均值>=25）从7降至{int(cstate.middle_mismatch)}个，详见goal_path_mismatch_persistence.csv。

固定164群体的均值：

{table(fixed.groupby("epoch")[["ADE1","minADE20","minFDE20","minGoalFDE20"]].mean().reset_index(),["epoch","ADE1","minADE20","minFDE20","minGoalFDE20"])}
''')
    write('04_SCENE_AND_MOTION_ANALYSIS.md',f'''# 分录像与冻结历史运动组

{common}

所有5个录像完整保留。以下最终Epoch{last}为各自窗口均值；录像宏平均须等权，不能与998目标窗口加权混为一谈。

{table(sc[sc.epoch==last],['scene','targets','ADE1','FDE1','minADE20','minFDE20','A25_miss_count','A25_miss_fraction','A25_error_share','SelectionGap_ADE_mean'])}

两个聚合口径同时保留；录像宏平均结果（各录像等权）：

{table(macro[macro.epoch>=30],["epoch","ADE1","FDE1","minADE20","minFDE20"])}

每录像A25_error_share分母是该录像全部目标minADE20之和；A25_share_of_all_errors使用全部998目标误差分母，两者均保存于scene_metrics.csv。deathCircle_3的困难只是一条录像的描述，不能直接证明场景语义失效。

历史高速、转向、变速、静止组全部沿用Stage11B-2A冻结标签，未重新调整阈值。原始表来源及SHA见protocol_frozen.json，不能把未来运动标签作为历史分组。

{table(gr[gr.epoch==last],['factor','group','targets','ADE1','minADE20','minFDE20','A25_miss_fraction','A25_error_share'])}

Epoch30至每个新完整验证的全部组均保存在motion_group_metrics.csv，空缺/未运行状态不填零。组间差异是描述性分层，不能作为独立因果证据。固定案例使用原failure_case_manifest.csv的21个身份（未按新结果挑选），PNG/SVG及真实全K20坐标保存在figure_manifest.json和fixed_case_prediction_coordinates.csv.gz。图中oracle仅用于离线解释，保留全部失败候选。
''')
    dc=sc[(sc.epoch==last)&(sc.scene=='deathCircle_3')].iloc[0]
    q=[
    ('是否完成Epoch50',f"{d['status']}；最终训练{d['final_trained_epoch']}，最终完整验证{last}。"),
    ('minADE20变化',f"{old['minADE20']:.6f}→{fin['minADE20']:.6f}px；变化{fin['minADE20']-old['minADE20']:+.6f}px（{(fin['minADE20']-old['minADE20'])/old['minADE20']:+.2%}，正值为退化）。"),
    ('A25失败率',f"{old['A25_miss_fraction']:.6%}→{fin['A25_miss_fraction']:.6%}。"),
    ('A25误差份额',f"{old['A25_error_share']:.6%}→{fin['A25_error_share']:.6%}。"),
    ('原164个恢复数量',str(t['recovered'])),('新失败数量',str(t['new_failures'])),
    ('deathCircle_3',f"最终A25={int(dc.A25_miss_count)}/{int(dc.targets)}，minADE20={dc.minADE20:.6f}px。与全部录像比较见scene_metrics.csv；不作语义归因。"),
    ('K1/K20差距',f"ADE均值{old['SelectionGap_ADE_mean']:.6f}→{fin['SelectionGap_ADE_mean']:.6f}px；FDE{old['SelectionGap_FDE_mean']:.6f}→{fin['SelectionGap_FDE_mean']:.6f}px。"),
    ('Goal正确但Path错误',f"{int(old['GoalPathMismatch_count'])}→{int(fin['GoalPathMismatch_count'])}；身份转移见target_cohort_transitions.csv。"),
    ('是否平台期',"训练loss缓慢下降，但4次新增验证均未超过Epoch30，未见验证随训练持续改善。该结果更接近泛化波动/退化，不能证明充分收敛或区分过拟合与目标函数不一致；未追加Epoch60。"),
    ('有限预算H1',d['H1']),
    ('是否值得直接开发目标候选方法',"仅可决定是否值得下一步机制/文献审查，不具备直接创新开发GO。"),
    ('语义不确定性直接证据',"没有；无可靠语义GT/映射及受控语义干预，不能因录像困难或路径偏离作因果归因。"),
    ('下一步一个待验证假设',"持续覆盖失败是否主要表现为GT终点附近热图概率质量不足，还是固定TTST proposal→KMeans压缩丢失覆盖？需要先作固定权重的概率/抽样分解和已有方法查重；本阶段未执行，待科研决策批准。")]
    write('05_CONVERGENCE_DECISION.md','# 收敛与研究决策\n\n'+common+'\n\n'+'\n\n'.join(f'**Q{i+1} {title}**：{answer}' for i,(title,answer) in enumerate(q))+f'''\n\n证据索引：Q1/2/8/10见epochs.csv、validation_metrics.csv和training_execution.json；Q3/4/11见goal_coverage_epochs.csv；Q5/6见a25_transition.csv；Q7见scene_metrics.csv；Q9见candidate_errors_epoch*.csv.gz及target_cohort_transitions.csv；Q12—14为限定性研究判断/待验证假设。所有判断仅适用于本开发协议和本次预算。\n\n工程恢复GO={d['restore_GO']}，执行状态={d['status']}；H1={d['H1']}。进一步Goal覆盖问题审查={d['further_goal_coverage_audit_worthwhile']}；直接算法开发GO=False；语义因果证据=False；自动追加训练GO=False；4次新增验证均不及Epoch30，不建议立即追加相同训练到Epoch60。完成后停止，不进入Stage11C。''')
    write('STAGE11B2B_REPORT.md',f'''# Stage11B-2B：有限训练延长与Goal覆盖持续性审计

{common}

**执行：{d['status']}；H1：{d['H1']}。**

完整恢复原Epoch30模型/Adam/RNG，从Epoch31继续到{d['final_trained_epoch']}，累计{d['cumulative_global_step']}更新，新增GPU进程wall {d['new_GPU_seconds']:.3f}秒。不存在从随机权重重新训练、空Adam续训、修改学习率/结构/loss或测试选模。首次GPU准入的合成标签越界失败、修订及重新冻结全部保留；正式训练未发生对应错误。

{table(m[m.epoch>=30],['epoch','ADE1','FDE1','minADE20','minFDE20','A25_miss_count','A25_miss_fraction','A25_error_share','GoalPathMismatch_count'])}

最终Epoch{last}与固定Epoch30：minADE20变化{fin['minADE20']-old['minADE20']:+.6f}px（正值为退化）；minFDE20 {old['minFDE20']:.6f}→{fin['minFDE20']:.6f}px；ADE1 {old['ADE1']:.6f}→{fin['ADE1']:.6f}px，FDE1 {old['FDE1']:.6f}→{fin['FDE1']:.6f}px。上述维度分别解释，不能把oracle改善等同所有预测能力改善。

原164目标中恢复{t['recovered']}，新失败{t['new_failures']}，持续失败{t['persistent_failure']}。A25误差份额{old['A25_error_share']:.4%}→{fin['A25_error_share']:.4%}。H1按最终完整验证而非最佳checkpoint判断。

全体最佳Epoch{d['global_best_epoch']}，新最佳Epoch{d['new_best_epoch']}，均按窗口minADE20选。最佳文件 `{ck['path']}`，SHA `{ck['sha256']}`。最后训练epoch与最后完整验证均单独记录。完整旧7+新4节点、NOT_RUN状态以及checkpoint/cache SHA见validation_metrics.csv、validation_status.csv和checkpoint_manifest.json。

报告导航：01恢复与模式；02执行、预算及选模；03固定/动态覆盖及K1/K20；04全部录像及历史组；05逐项科研问答和停止判断。10类图均有PNG/SVG与对应CSV/JSON，案例身份沿用上一阶段。

录像宏平均指标另见recording_macro_metrics.csv；窗口加权是主口径，两者不混用。

完整测试、历史SHA及交付检查见full_tests_final.txt/xml、history_after.json和delivery_audit.json（最终检查后生成）。未将skip写为pass。源码与数据来源见protocol_frozen.json及verified_validation_sources.json，GPU失败/成功进程均计费，旧成果保持只读。

本阶段不构成新算法，不证明充分收敛或语义因果，不自动启动任何后续训练。下一阶段至多建议在批准后进行覆盖机制与新颖性审查。''')
    print('Reports written from actual results',last,flush=True)
if __name__=='__main__':main()
