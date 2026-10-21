"""Evidence-grounded report rendering from saved descriptive tables."""
from .common import *
import numpy as np,pandas as pd

def table(df,cols):
    data=df[cols];lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in data.iterrows():
        values=[]
        for v in r:
            if pd.isna(v):values.append('未定义')
            elif isinstance(v,(float,np.floating)):values.append(f'{v:.6f}')
            else:values.append(str(v).replace('|','/'))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)
def put(name,s):
    with (REPORT/name).open('x') as f:f.write(s.strip()+'\n')

def main():
    cpu_guard();f=pd.read_csv(OUT/'motion_features.csv');s=pd.read_csv(OUT/'scene_normalized_analysis.csv')
    g=pd.read_csv(OUT/'motion_error_groups.csv');dist=pd.read_csv(OUT/'metric_distributions.csv');run=readj(OUT/'analysis_execution.json')
    c=pd.read_csv(OUT/'goal_path_decomposition.csv');tax=pd.read_csv(OUT/'failure_type_summary.csv');loo=pd.read_csv(OUT/'leave_one_recording_out.csv')
    curves=pd.read_csv(OUT/'frozen_validation_trends.csv');std=pd.read_csv(OUT/'scene_motion_standardization.csv')
    corr=pd.read_csv(OUT/'descriptive_correlations.csv');cases=pd.read_csv(OUT/'failure_case_manifest.csv')
    a25=tax[(tax.scene=='ALL_WINDOWS')&(tax.type=='GOAL_COVERAGE_MISS')].iloc[0]
    missdc=int(((f.scene=='deathCircle_3')&f.goal_miss25).sum());clos=f[f.minGoalFDE20<=10]
    metadata=f'''数据：998个固定开发验证目标、5录像、seed17、Epoch30，K20/10000 proposals；误差原参考图像像素。来源缓存SHA `{PSHA}`，checkpoint SHA `{CK}`。所有结果均为**开发集选模之后的描述性分析**，不是新算法、正式测试或独立确认。本阶段CPU离线计算，模型前向0次、训练0步、测试池未加载。阈值与规则在 `{readj(OUT/'protocol_freeze.json')['commit']}` 提交后才执行分组。
'''
    put('02_GOAL_PATH_ERROR_ANALYSIS.md',f'''# Goal/Path逐候选对应关系

{metadata}

记第k条路径为p_ik、goal为g_ik，真实未来为y_i。ADE_ik=mean_t||p_ik,t-y_i,t||；FDE_ik=||p_ik,12-y_i,12||；GoalFDE_ik=||g_ik-y_i,12||；Gap_ik=||g_ik-p_ik,12||。分别保留argmin_Goal、argmin_ADE、argmin_FDE，精确同值保留原顺序较早head。**不将不同head的minGoalFDE与minADE相减当成误差分解。**

窗口均值：GoalFDE1={f.GoalFDE1.mean():.6f}，minGoalFDE20={f.minGoalFDE20.mean():.6f}；ADE1={f.ADE1.mean():.6f}，minADE20={f.minADE20.mean():.6f}；路径FDE1={f.FDE1.mean():.6f}，minFDE20={f.minFDE20.mean():.6f}。

最近goal对应head与ADE最优head相同为{int(f.closest_goal_is_ADE_best.sum())}/998，与FDE最优相同为{int(f.closest_goal_is_FDE_best.sum())}/998。最近goal对应完整路径ADE均值{f.closest_goal_ADE.mean():.6f}，FDE均值{f.closest_goal_FDE.mean():.6f}。这两个值可与oracle说明对应路径差距，但最近goal也依赖GT，不能在线使用。所有19960个候选配对保存在goal_path_decomposition.csv；每个候选还保存前/中/后4步误差和goal-path间距。

## 覆盖条件与错误

{table(g[g.family.str.startswith('goal_coverage')],['family','group','targets','minADE20','SelectionGap_ADE'])}

25px邻域下，{int(a25.targets)}/998个目标全部goal均未覆盖，平均minADE20={a25.minADE20:.6f}，贡献全体minADE20误差总和的{a25.cohort_minADE20_error_share:.2%}。834个覆盖目标平均minADE20={f[~f.goal_miss25].minADE20.mean():.6f}。这支持“较大的剩余误差与goal覆盖不足共现”，不是goal模块因果贡献百分比分解。未覆盖目标只占16.43%，不能称为多数目标普遍失败。

## 目标正确而中段错误

严格使用同一候选：GoalFDE<=10px且ADE>=25px，共{int(f.GOAL_PATH_MISMATCH.sum())}个目标，最近goal候选也均满足此条件。其中{int(f.mismatch_middle_target.sum())}个具有FDE<=10px且前11步平均误差>=25px，终点FDE>=25px的子类型为{int(f.mismatch_terminal_target.sum())}。说明这7例主要是路径中段偏离，而非goal与解码末点的普遍不一致。该类型总样本少，不能代替总体结论，不能推断由语义错误造成。

{table(f[f.GOAL_PATH_MISMATCH],['sample_id','closest_goal_head','closest_goal_ADE','closest_goal_FDE','minGoalFDE20','minADE20'])}

图2、3与病例对应路径只作离线解释；图中ADE oracle、FDE oracle和最近goal head有明确不同含义。
''')
    put('03_K1_K20_GAP_ANALYSIS.md',f'''# 第一候选与GT oracle差距

{metadata}

SelectionGap_ADE=ADE1-minADE20，SelectionGap_FDE=FDE1-minFDE20；全部998目标均非负。相对gap=gap/max(第一候选误差,1px)，不是独立概率校准指标。下面的分位数使用冻结linear方法，不删极端值。

{table(dist[(dist.scene=='ALL_WINDOWS')&dist.metric.isin(['SelectionGap_ADE','SelectionGap_FDE','SelectionGap_ADE_relative','SelectionGap_FDE_relative'])],['metric','valid_n','mean','median','p25','p75','p90','p95'])}

第一候选已是ADE oracle的目标为{int(f.default_is_ADE_best.sum())}/998（{f.default_is_ADE_best.mean():.2%}）；ADE和FDE最优head相同{int(f.ADE_FDE_same_head.sum())}/998。绝对ADE gap>=25px为{int(f.high_gap_ADE25.sum())}/998。严格B类同时满足默认ADE>=25、oracle ADE<=10、gap>=15、相对gap>=0.5，为{int(f.DEFAULT_ORACLE_GAP.sum())}个目标。全部目标分布和每录像分布均保留，不能只分析B类。

goal在10px内有覆盖的{len(clos)}个目标中，仍有{int(clos.high_gap_ADE25.sum())}个ADE gap>=25px；其平均gap={clos.SelectionGap_ADE.mean():.6f}px。覆盖良好时仍可能存在默认候选差距，但较大gap并非全部由这种情况构成。

{table(s,['scene','targets','SelectionGap_ADE','SelectionGap_FDE'])}

## 历史特征的关联限度

{table(corr[(corr.scene=='ALL_WINDOWS')&(corr.y=='SelectionGap_ADE')&(corr.feature_scope=='HISTORY_INPUT')],['x','finite_pairs','Spearman'])}

历史幅度和归一化速度与gap存在描述性关联；速度CV和转角的关联较弱。逐录像相关也在descriptive_correlations.csv保存，避免把合并相关当稳定跨录像规律。本轮没有训练任何可部署选择器，不能声称历史输入已能找出oracle head。

源码scripts/stage11b1r/predictor.py把softargmax goal放在第0位，其后19个KMeans中心不构成训练过的置信度排序。没有本研究候选排序头。因此不能把gap称为“排序头失效”，更不能用GT选出的候选冒充在线预测能力。Y-Net已有TTST和goal条件路径，gap本身不构成新算法创新。
''')
    put('04_MOTION_DIFFICULTY_ANALYSIS.md',f'''# 运动难度：历史输入与未来事后信息分开

{metadata}

历史8帧有7个0.4秒间隔，速度=相邻位移/0.4；未来累计路程包括最后历史点到未来第1点，共12间隔4.8秒。速度单位px/s；归一化速度为图像对角线/秒，仍不是m/s。加速度由相邻二维速度差/0.4获得，转角只在相邻两个位移均>=1px时定义。静止/低速时方向无效保留NaN，不填成0度直行。曲率相关量为累计转角/累计路程；直线性为净位移/路程，静止时未定义。

历史速度/幅度阈值来自7496个来源训练目标的历史特征三分位；未来幅度阈值仅用于本998目标事后诊断，规则事前冻结，数值见derived_thresholds.json。历史平均速度与历史累计路程仅相差固定2.8秒，二者的分组相同，**不是两个独立证据**。

## 输入可观测分组

{table(g[g.feature_scope=='HISTORY_INPUT'],['family','group','targets','ADE1','minADE20','minADE20_diag','minADE20_future_path_ratio','SelectionGap_ADE'])}

历史转向组270目标的minADE20=29.206982px，直行组322目标为13.128514px；对角线归一化后仍有差异。历史速度变化CV>=0.4组164目标为31.981155px，稳定组394目标为14.217757px。它们是关联性难度线索，未通过随机干预隔离原因，也未排除录像组成和标签质量。每组录像构成与完整误差均在motion_error_groups.csv。

所有目标都有8帧完整历史；166个目标仅0—1个>=1px可靠位移，其minADE20均值5.172818px。147个历史全步长<=1px目标均值4.573171px。因此本数据不支持把总体大误差归为“历史缺帧/历史运动信息不足”；只可描述方向证据不足的小群体。

## 未来GT事后难度，禁止作为在线输入

{table(g[(g.feature_scope=='FUTURE_GT_POSTHOC')&(~g.family.str.startswith('goal_coverage'))],['family','group','targets','minADE20','minADE20_diag','minADE20_future_path_ratio','SelectionGap_ADE'])}

未来大路程组原像素误差显著较大，但minADE20/未来路程均值高组0.091455、中组0.083544，差距明显收窄。不能仅凭大位移ADE高宣布高速预测存在系统性缺陷。未来转向组的路程归一化误差约0.180137，直行组约0.096863，仍有描述性差异，提示中段路径形状值得核查。

路程归一化使用max(未来路程,1px)防止除零，未来近静止目标会有很大的比值；例如低未来幅度组均值1.978182，不应将该比值当统一物理难度或跨组损失。全部原始/对角线/路程归一化结果同时提供，不选择最有利口径。

本轮没有通过历史特征预测未来难度标签；不将未来转角、位移、CV或CV预测误差用于模型输入、建图或在线路由。
''')
    put('05_SCENE_HETEROGENEITY.md',f'''# 五录像异质性与留一敏感性

{metadata}

{table(s,['scene','targets','diagonal','minADE20','minADE20_diag','hist_speed_mean','future_path_mean','goal_miss25_fraction','YNet_ADE1_minus_OLS8_ADE1'])}

deathCircle_3的443目标minADE20=27.077129px，图像对角线归一化为0.011457；其他录像约0.002843—0.004438。该录像历史平均速度37.922796px/s、未来平均路程171.391028px，高于其他四录像约22—25px/s和103—110px，但跨录像像素速度不是统一物理速度。

25px目标覆盖失败中，{missdc}/{int(a25.targets)}（{missdc/int(a25.targets):.2%}）来自deathCircle_3，其录像内失败率30.02%，其他录像4.17%—6.67%。图像大小、运动幅度、交通对象组成、未来路线、标注质量及预训练曝光均可能混杂，不能据此认定哪个原因已经证明。

## 按运动幅度的描述性标准化

{table(std,['scene','factor','all_three_bins_present','standardized_minADE20_diag'])}

采用同一全队列三分位比例权重、每录像每bin实际误差，不补缺bin。按历史速度标准化后deathCircle_3约0.011447；按未来路程事后标准化后约0.010755，均仍高于其余录像。这支持“尺度和粗粒度运动幅度不能完全消除观察到的差异”，不是严格匹配、因果控制或语义残差证明。完整scene_motion_strata.csv保留每个bin样本数与运动均值，分层内部仍可能有不平衡。

## 同目标OLS8与留一录像

{table(loo[loo.baseline=='OLS8'],['excluded_recording','targets','paired_ADE1_delta','paired_FDE1_delta','recording_macro_ADE1_delta','recording_macro_FDE1_delta'])}

差值均Y-Net减OLS8，负值改善。整体窗口加权ADE差-0.633333px；去掉deathCircle_3后变成+1.235789px，FDE差变成+1.586607px。去掉nexus_4后，录像宏平均ADE差亦轻微反转为+0.027404px。完整录像中Y-Net相对OLS8的ADE方向为3/5改善、2/5退化，窗口改善只有408/998目标，不能称跨录像稳定提升。

5次留一分析只作敏感性，完整998总体不变，不删除该录像后挑选新指标。没有窗口级bootstrap、p值或有限5簇的可靠总体显著性结论。参考图像对角线归一化不是米制标定，也没有识别出真正物理难度的倍数关系。
''')
    outside=pd.read_csv(OUT/'geometric_out_of_image_points.csv');badheads=c[c.outside_points>0]
    put('06_FAILURE_TAXONOMY.md',f'''# 非互斥失效类型与几何核查

{metadata}

类型定义完整保存在冻结analysis_protocol.json。A25为所有goal距GT终点>25px；A10/A50同时报告。B为默认ADE>=25、oracle ADE<=10、绝对gap>=15且相对gap>=0.5。C为至少一个**同候选**goal误差<=10且路径ADE>=25。D为oracle ADE>=25且历史转向/速度变化/可靠方向证据弱之一，属于关联标签。E为候选越出参考范围或最大步长>max(50px,历史最大步长4倍)。绝对阈值受像素尺度限制，不能当通用物理标准。

{table(tax[tax.scene=='ALL_WINDOWS'],['type','targets','fraction','ADE1','minADE20','cohort_minADE20_error_share'])}

各类型不互斥。A25与D重叠59例，A25与B重叠2例，C与D重叠3例，C与E重叠1例。详细5×5交集、每录像数量与精确标签组合分别在failure_overlap.csv、failure_type_summary.csv、failure_exact_combinations.csv；比例不能加总为100%。

## 几何越界

核实原有51个越界预测点，涉及11目标、{len(badheads)}条候选，而非新增预测；占239520点的{51/239520:.4%}。这些目标平均minADE20=10.645096px，贡献总体minADE20误差总和约0.7153%，因此不是当前总体大误差的主要组成。该描述不等于“删除越界候选不会影响oracle”，本轮没有删除后重新选择候选或重排名。

逐点sample_id/head/未来步/x/y/图像范围与对应候选误差均在geometric_out_of_image_points.csv。全部候选的outside_points和goal是否越界也有记录。11目标没有触发冻结的相对历史大步长阈值；这不代表所有预测速度都合理，只说明该定义下没有额外标记。图像范围外可能是模型外推、图像边界或坐标支持问题，单凭当前缓存无法判定原因。

{table(outside.groupby('sample_id',sort=True).agg(outside_points=('head','size'),outside_heads=('head','nunique')).reset_index(),['sample_id','outside_points','outside_heads'])}

## 真实案例

7类各3例，共21个固定选择、21个不同目标，无跨类重复。本次选择不是为了美观；每类按预先冻结score排序取25/50/75%秩。全景和局部共42幅案例图，保留20路径/20goal、第一候选、ADE oracle、对应最近goal路径、CV和OLS8；各图坐标都来自原缓存或核验过的解析基线。

{table(cases,['case','sample_id','ADE1','minADE20','minGoalFDE20','failure_types'])}

三个优先核查类型：A25目标覆盖失败；B默认与oracle差距；C对应goal准确但中段偏离。C类7个目标中的bookstore_2 track208可见未来回转，准确终点不保证正确中途路线；RGB只能显示背景，不能给出道路、草地、树冠的可通行真值。未来直行/转向、oracle高亮与最近goal路径全部是离线GT诊断，不是部署推理。
''')
    allc=curves[curves.scene=='ALL_WINDOWS'];c25=allc[allc.epoch==25].iloc[0];c30=allc[allc.epoch==30].iloc[0]
    change=curves[(curves.scene!='ALL_WINDOWS')&curves.epoch.isin([25,30])].pivot(index='scene',columns='epoch',values=['ADE1','minADE20'])
    ch=pd.DataFrame(dict(scene=change.index,ADE1_25_to30=change['ADE1'][30]-change['ADE1'][25],minADE20_25_to30=change['minADE20'][30]-change['minADE20'][25])).reset_index(drop=True)
    ch.to_csv(OUT/'recording_epoch25_to30_changes.csv',index=False)
    put('07_CONVERGENCE_DIAGNOSTIC.md',f'''# 只使用已有七次验证的收敛诊断

{metadata}

{table(allc,['epoch','ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','within_10px','within_25px','within_50px','endpoint_diversity'])}

epoch25→30，窗口minADE20变化{c30.minADE20-c25.minADE20:+.6f}px、minFDE20变化{c30.minFDE20-c25.minFDE20:+.6f}px；ADE1变化{c30.ADE1-c25.ADE1:+.6f}px、FDE1变化{c30.FDE1-c25.FDE1:+.6f}px。25px覆盖率变化{(c30.within_25px-c25.within_25px)*100:+.3f}个百分点，10px覆盖率变化{(c30.within_10px-c25.within_10px)*100:+.3f}个百分点，不能笼统说所有终点覆盖持续改善。

{table(ch,['scene','ADE1_25_to30','minADE20_25_to30'])}

bookstore_2的minADE20在25→30退化，其他四录像改善；K1在bookstore/gates/hyang退化。末期训练loss从0.092134降到0.091851，K20仍有改善迹象，但不是K1持续改善的证明。最佳在末轮仅说明预算截断，不等于严重欠收敛；只有七节点不能可靠拟合或外推50epoch实际成绩。

**建议：可以单独申请一次冻结的有限延长诊断，不在本阶段执行。** 若下一阶段只允许一个实验，优先保留Epoch30状态与全部协议，另行批准后继续最多20完整epoch、固定35/40/45/50完整验证；主选模规则仍为minADE20，K1和goal/path类型变化作为配对辅助。必须明确同一开发集已多轮暴露，不形成独立确认，不追求最小误差而不断延长。

其可检验信息是：A25误差集中是否随训练明显消退、goal正确而路径偏离是否持续、默认/oracle差距是否扩大、各录像退化方向是否重复。不能承诺成绩提高或预填50epoch数字。若没有预算批准或恢复状态核验失败则不执行；若两次固定验证K20改善均<1%且K1持续退化，可按下一阶段预先冻结的停止规则结束，不能反过来搜索loss/学习率。此停止阈值仅为**待批准设计建议**，不是本阶段执行结果。
''')
    put('08_INNOVATION_HYPOTHESIS_AUDIT.md',f'''# 语义创新假设审查：尚未通过算法开发门槛

{metadata}

## 已支持事实与尚未支持解释

SUPPORTED：A25覆盖失败164目标贡献61.08% oracle ADE总和；C类7个同候选goal正确但中段偏离；历史转向/变速与误差有关联；deathCircle_3经过对角线与粗幅度分层仍相对高误差。

UNVERIFIED：这些现象由分割错误、语义不确定性或语义预训练曝光导致。没有可靠的SDD数字类别→名称映射，没有本批目标对应的已核验语义GT，也没有分割像素错误/校准证据。本阶段没有运行分割器、语义干预或新增模型。剩余录像差异也不能自动称为语义因果效应。

NOT_SUPPORTED：把简单goal/path误差差值称严格根因分解；把默认候选gap称排序头失效；把完整8帧下方向不可靠称缺帧；把像素越界称碰撞；把现有结果称创新算法提升。

## 候选假设H1：覆盖瓶颈能否在收敛后持续

- 研究问题：有限预算末轮观测到的goal覆盖集中误差，在原网络进一步收敛后是否仍存在，而不是未充分训练的暂时现象？
- 支持：A25占16.43%但贡献61.08% oracle误差，25→30的minADE20仍下降；最近goal路径与oracle ADE差距总体较小，但有7例独立中段形状错误。
- 缺口：没有50轮同seed恢复实验，更无跨seed/独立确认；高误差可能有标签异常或运动混杂。
- 文献重合：Y-Net本来已分解goal/waypoint/path并使用TTST；PECNet/Goal-GAN已有终点条件路径。问题诊断可做，简单换采样/增加goal条件不构成新颖性。
- 最低成本验证：仅在新阶段批准后做报告07的固定训练延长，重用合法998目标和checkpoint恢复；不新增模块。需要完整模型/优化器/RNG与manifest SHA，已保存但仍须核查。
- 支持条件建议（下阶段另行冻结）：后期误差稳定后A25仍>=10%目标且贡献>=50% oracle误差，提示持续覆盖瓶颈；若下降至<5%且其误差份额显著转移，当前持久瓶颈假设被削弱。中间结果继续标记未定，不再追分。
- 贡献边界：仅可支撑选择研究问题，不能称新算法；后续具体数学机制和公平消融尚不存在。

## 候选假设H2：真实语义错误与覆盖失败是否有关

- 当前状态：**UNVERIFIED，不批准语义门控或新网络。**
- 研究问题：在相近历史运动和图像尺度下，独立标注的语义错误/歧义是否关联goal覆盖失败，且可由不依赖未来GT的指标识别？
- 当前支持限度：仅有录像残余差异及少量中段路径错误，两者均不是分割不可靠的直接证据。
- 缺少：可核实类别字典、语义GT、分割训练成员、冻结模型输出或合法缓存、校准数据；高softmax熵不等于真实语义错误。
- 最低成本验证：先做类别与标签来源审计；若通过，另行冻结从历史位置与录像分层抽取的少量ROI人工盲标方案，标注者不看模型误差；用已有或另行批准的冻结分割输出检验误差/可靠性关系。未来GT只用于最终轨迹评分，不构造可靠性输入。本阶段不执行。
- 支持条件：有明确语义GT错误证据，历史可观测可靠性指标与像素错误的校准关系成立，且逐录像控制描述性运动幅度后与goal失败方向一致，才进入独立确认；映射/标注无效、关联消失或只在单录像成立即不支持普遍机制。
- 文献重合风险：Y-Net、Goal-GAN、Semantic scene upgrades已研究场景/语义条件预测；[UATP](https://arxiv.org/abs/2603.29362)为2026预印本，已提出位置与语义不确定性联合建模。领域/栅格差异不能自动构成机制创新。CINet/McTrans/GDS全文缺口继承旧审计，不能声称前人未做。
- 贡献边界：需要先建立真实错误与可观测可靠性的证据链；无数据不能制造“语义不可靠造成退化”的论文结论。

文献范围：重用reports/stage11a/06_NOVELTY_AUDIT.md和stage11ap/07_NOVELTY_GAP_UPDATE.md已声明的阅读范围。本轮CVF两页重访403，未伪称重新阅读全文；UATP作者arXiv元数据可访问，仅用于核对潜在重合，不据摘要证明完整数学等价。访问记录见literature_scope.json。本阶段没有完成新的系统文献查重。

**创新研究NO-GO。** 当前应把问题暂定为“终点覆盖和路径形状错误的可重复瓶颈验证”，而非已经确立的“语义不确定性算法”。
''')
    decisions=[
      dict(question='Q1 当前主导误差',status='SUPPORTED',answer='A25少数目标集中贡献61.08% oracle ADE误差；是共现的主要误差群体，不是因果分解'),
      dict(question='Q2 覆盖失败是否普遍',status='SUPPORTED',answer='25px为16.43%，其中81.10%来自deathCircle_3；10/50px结论随阈值变化，非多数目标普遍失败'),
      dict(question='Q3 K1/K20差距是否值得研究',status='PARTIALLY_SUPPORTED',answer='gap真实且较大，但没有证明可在线识别oracle；不是排序头失效或新颖性证据'),
      dict(question='Q4 deathCircle_3能否由尺度/运动解释',status='PARTIALLY_SUPPORTED',answer='尺度和运动分布能解释部分表面差异；粗分层后仍高，不能认定剩余为语义作用'),
      dict(question='Q5 goal与path是否不同失效',status='SUPPORTED',answer='覆盖失败164例与对应goal正确中段偏离7例不同；不能将二者加总分解原因'),
      dict(question='Q6 语义可靠性方向有直接证据吗',status='UNVERIFIED',answer='类别、语义GT和分割错误/校准均未核实；没有证明语义导致预测误差'),
      dict(question='Q7 是否先延长训练',status='PARTIALLY_SUPPORTED',answer='值得另行申请一次固定有限延长以判别持续瓶颈；未执行，不能保证提高'),
      dict(question='Q8 只做一个下一阶段实验',status='PARTIALLY_SUPPORTED',answer='优先固定协议的Epoch30到最多50收敛诊断，先排除预算截断；需要另行批准'),
      dict(question='Q9 是否满足创新模块开发门槛',status='NOT_SUPPORTED',answer='没有语义因果证据、机制创新或公平新算法消融，不具备开发条件'),
      dict(question='Q10 是否调整小论文问题',status='PARTIALLY_SUPPORTED',answer='暂以goal覆盖/路径形状瓶颈及收敛验证替代已确定的语义因果叙事；不自动换模型或立项')]
    pd.DataFrame(decisions).to_csv(OUT/'research_decision_matrix.csv',index=False)
    writej(OUT/'final_decision.json',dict(analysis_GO=True,innovation_GO=False,semantic_causation='UNVERIFIED',new_training_executed=False,new_test_scores=False,
        next_stage_automatic=False,recommendation='Submit one separately approved frozen convergence diagnostic; no automatic training or semantic module',questions=decisions,
        targets=998,recordings=5,GPU_seconds=0,source_cache_SHA=PSHA,source_checkpoint_SHA=CK))
    put('STAGE11B2A_REPORT.md',f'''# Stage11B-2A Y-Net失效机制分析总报告

{metadata}

## 一、数据和缓存完整性

冻结权重与原始预测缓存SHA匹配，全部history/GT/reference和六项逐目标误差核验通过，NaN/Inf为0。25训练录像历史只用于预定阈值估计，不加载测试池、不执行模型推理或训练。原始来源列split=train与HTraj的development_validation成员已区分。详见01报告和data_cache_audit.json。

## 二、Goal/Path逐候选分析

goal覆盖失败10/25/50px分别489/164/99个目标（49.00%/16.43%/9.92%）。A25目标平均minADE20=60.971240px，共现群体贡献61.08%总oracle误差，不是严格根因分解。对应goal<=10px但路径ADE>=25px的7例均是中段明显偏离、末端仍接近；详见02及19960候选逐一配对。

## 三、K1/K20差距

ADE gap均值15.689964px、中位9.553162px；FDE gap均值34.904992px、中位22.522168px。默认候选ADE最优69/998，ADE/FDE最优head相同708/998。大gap存在，但第一候选只是softargmax goal路径，没有本研究排序头，也没有可部署oracle选择能力。

## 四、运动难度

历史转向/变速与误差较高共现；未来大位移的原像素差异在路程归一化后大幅缩小。静止比值受1px分母floor影响，全部口径同时保留。完整历史8帧均存在，没有证据把总体错误归因于历史缺帧。输入特征与未来GT事后特征分表/分组，未用于在线输入。

## 五、录像差异

deathCircle_3占133/164个A25失败；归一化和运动幅度分层后仍相对较难，原因未确定。去掉该录像，Y-Net相对OLS8窗口ADE差从-0.633333反转为+1.235789px。仅5录像，不能视998个窗口为998个独立场景。

## 六、非互斥分类

{table(tax[tax.scene=='ALL_WINDOWS'],['type','targets','fraction','minADE20'])}

A与D重叠59例，不加总为100%。51个图像越界点涉及11目标、非总体主要误差，不能称碰撞。所有候选与高误差目标保留。

## 七、真实案例与图件

7类各3例，共21个真实目标，每例全景/局部，完整20轨迹与goal、默认/对应goal/ADE oracle、CV/OLS8均保留。选样依据冻结分位秩，无美观重选。10组量化图及案例组合PNG/SVG，坐标/身份/权重SHA可复核。图10为每类中间选例的组合，其余案例独立提供。GT高亮仅离线解释，不是新预测。

## 八、收敛趋势

只用1/5/10/15/20/25/30七次真实验证。25→30 oracle ADE/FDE继续改善，默认ADE/FDE退化，bookstore oracle也退化。最佳在末轮说明预算截断，不能确认收敛，也不能断定严重欠收敛。值得另行申请一次有限延长诊断，但当前没有执行，更没有50epoch成绩外推。

## 九、语义创新审核

没有可靠类别映射、对应语义GT或分割错误直接证据，**没有证明场景语义导致预测误差**。提出H1持续覆盖瓶颈、H2语义错误关联两项待验证假设，不开发模块，不将旧思想改名为创新。Y-Net/Goal-GAN/SSU/UATP的主题重合和全文缺口保持。

## 十、统计限制

checkpoint由同998目标选出，属于反复开发数据；本轮规则在分析前提交，但不等于独立确认。只有一个seed、5录像。无窗口bootstrap/p值，无可靠总体因果推断。raw pixel、对角线归一化和未来路程归一化不是同一评价口径，更非米制指标。分层内部仍有混杂，类别阈值是描述性规则，未来难度标签不能在线使用。

## 十一、十项明确判断

{table(pd.DataFrame(decisions),['question','status','answer'])}

## 十二、GO/NO-GO与交付

**离线失效分析完成GO；创新算法开发NO-GO；语义因果UNVERIFIED。** 最值得优先尝试的是另行批准的固定收敛诊断，验证覆盖瓶颈是否持续；不直接添加语义门控或评分头。保留现有Y-Net基线，不自动进入Stage11C，不延长、不换seed、不修改旧预测。

全部脚本、配置、逐目标/逐候选CSV、图件、案例坐标和统计记录在本阶段目录。适用测试及全仓CPU回归、原始失败/版式修订记录与历史SHA审计均保留。最终测试数量和保护文件数以full_unit_tests.xml、full_test_execution.json及history_after.json为准。
''')
    cpu_guard();print('Eight detailed analyses and consolidated decision report saved',flush=True)
if __name__=='__main__':main()
