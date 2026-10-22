"""Generate evidence-linked reports and an unexecuted single-control proposal."""
from .common import *
import pandas as pd
import numpy as np
import torch

def table(f,columns=None):
    f=f[columns] if columns else f
    rows=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for _,r in f.iterrows():
        rows.append('| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in r)+' |')
    return '\n'.join(rows)

def main():
    cpu_guard();rep=ROOT/'reports/stage11b2d';rep.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'target_diagnostics.csv');cl=pd.read_csv(OUT/'cluster_diagnostics.csv')
    r=pd.read_csv(OUT/'recording_comparison.csv');z=pd.read_csv(OUT/'threshold_support_review.csv')
    lit=pd.read_csv(OUT/'literature_review.csv');matrix=pd.read_csv(OUT/'literature_overlap_matrix.csv')
    summary=json.loads((OUT/'analysis_summary.json').read_text());cfg=json.loads((CONF/'analysis_protocol.json').read_text())
    failed=f[f.category25=='proposal_hit_final_miss'];hit=f[f.category25=='both_hit']
    corr=pd.read_csv(OUT/'descriptive_correlations.csv')
    cp=dict(phase='PROPOSAL_ONLY_NOT_EXECUTED',name='Fixed-Voronoi observed-sample representative control',
      hypothesis='At fixed draw cloud, partition, K and network, replacing continuous cluster means with a within-cell observed representative may change coverage and downstream paths; effect unverified',
      candidate_count=20,head0='original frozen Softargmax unchanged',other_heads=19,
      partitions='freeze final-center Voronoi assignment of original10000 draws; no reclustering',
      representative='argmin_{p in assigned existing draws} sum_{q in same cell} ||q-p||^2; ties lowest original draw index',
      empty_cell='retain original center; current final-cell audit has zero empty cells',duplicates='retain repeated draws and duplicate representatives; report effective distinct goals; no ad hoc filling',
      deploy_inputs=['existing10000 proposals','frozen19 centers','frozen Softargmax','original history and semantics only for frozen decoder'],
      GT_for_selection=False,new_training=False,changes_loss=False,novelty='ordinary squared-medoid control; NOT a proposed original algorithm',
      fairness='same998 IDs/history/GT/weights/K20/head0; baseline original cache immutable; alternative output in separate directory',
      decode='Any changed Goal MUST be sent through frozen original Path Decoder; old20 paths cannot stand in for changed Goals',
      metrics=['GoalCover10/25/50','GoalFDE20','minADE20','minFDE20','goal pairwise dispersion','empirical squared distortion','radius mass coverage under fixed p','CPU wall','GPU wall','memory'],
      analysis=['all998 windows','five-recording macro','each recording','deathCircle versus other4','all frozen history groups'],
      provisional_success_rule={'macro_cover25_gain_pp_min':1.,'macro_minADE20_relative_improvement_min':.01,
          'macro_minFDE20_relative_degradation_max':.01,'OTHER4_cover25_gain':'strictly positive',
          'interpretation':'development engineering gate only, not original-method or independent-generalization proof'},
      failure_rule='If these fail, reject this single representative-control hypothesis; no medoid/FPS/radius/seed sweep or additional training',
      confirmation='Before any SCI claim reserve genuinely unaccessed source recording(s), disjoint from baseline training/development and sealed test. Metadata-only feasibility check first; if unavailable report limitation. Do not repurpose sealed test autonomously.',
      approval='Requires a new user instruction and separate frozen protocol before execution',
      costs='No training; CPU representative control not measured. Priorstage decoder mean0.124399s/target (~124.15s/998), excludes startup/I/O; not a new timing result.')
    jnew(OUT/'hypothesis_decision.json',dict(hypotheses=[cp],maximum_two=True,innovation_admission=False,
        reason='Only a conventional control is justified; no distinct new mathematics or independent confirmation evidence'))
    decision=dict(status='COMPLETE',scope=cfg['scope'],compression_targets_analyzed=86,all_targets=998,
      historical_prediction_unchanged=True,zero_support_targets=77,new_GPU_seconds=0,new_training_steps=0,new_model_inference=0,
      assignment='FINAL_CENTER_VORONOI_ASSIGNMENT',primary_observation='low local draw mass and broader local assigned-cell radii coexist with loss',
      universal_large_dense_cluster_absorption='NOT_SUPPORTED',mechanism_interpretation='INFERRED_NOT_CAUSAL',
      novelty_risk='HIGH',new_original_algorithm_GO=False,Stage11C_GO=False,
      ordinary_control='PROPOSAL_ONLY_REQUIRES_NEW_APPROVAL',recommendation='NO_GO_INNOVATION; optional single conventional representative control may falsify support-projection explanation',
      original_Stage2C_decision='MECHANISM_INCONCLUSIVE unchanged',source_commit=commit())
    jnew(OUT/'final_decision.json',decision)
    statcols=['LocalProposalCount25','LocalUniqueLocations25','P_mass25','LocalAssignedClusterCount25',
      'LocalWeightedClusterPopulation','AssignedClusterRadiusP90','NearestCenterToGT25','NearestCenterToGT25_diag',
      'LocalToAssignedCenter_mean','LocalDrawsInLargeClustersFraction']
    med=pd.DataFrame([dict(metric=k,failure86_median=failed[k].median(),success834_median=hit[k].median()) for k in statcols])
    med.to_csv(OUT/'key_group_medians.csv',index=False)
    byscene=failed.groupby('scene')[['LocalProposalCount25','LocalWeightedClusterPopulation','AssignedClusterRadiusP90','NearestCenterToGT25_diag']].median().reset_index()
    byscene.to_csv(OUT/'failure_recording_medians.csv',index=False)
    case=json.loads((OUT/'frozen_case_data.json').read_text())
    def put(name,text):
        with (rep/name).open('x') as out:out.write(text.strip()+'\n')
    boundary='范围：EXPLORATORY_DEVELOPMENT_ANALYSIS；既有5录像998重叠目标窗口，非独立测试。审计全阶段CPU_ONLY，未重新执行冻结YNet模型、采样、路径解码、实验训练或测试集评分；完整pytest包含既有模型的CPU人工张量结构测试，不是新增预测实验。GT只用于既有数据的离线测量。'
    put('01_CACHE_AND_CLUSTER_AUDIT.md',f'''# 缓存与最终簇分配审计

{boundary}

基础提交`{BASE}`，协议冻结提交`ef66d85c47862f3cf5049224e152c5435011afd8`。运行前完成协议/数值源码提交；analysis_summary.json记录实际分析代码commit。完整来源见protocol_frozen.json、cache_audit.json。Checkpoint SHA `{CK}`；原预测SHA `{PSHA}`。核对全部998个唯一sample_id、原始10000个int64索引顺序、float32采样坐标、19中心、Softargmax、20Goal和20路径，均通过；原网络和参数未调用或更新。

## 坐标与身份

网络坐标对应原像素网格(x/0.25,y/0.25)，不加0.5像素中心偏移。原顺序保留；重复抽样没有去重。中心原坐标=centers_network/0.25。所有几何用float64离线计算，原始float32张量不改。每项距离另除参考图像对角线，保留两种单位；SDD像素不与ETH/UCY米制比较。

## 原KMeans源码及不能复原的部分

只读`experiments/stage11a/cache/official/ynet/utils/kmeans.py`，文件SHA见冻结协议。每轮先以旧中心计算choice_cluster，再按这些标签更新中心，最后检验center_shift平方或iter_limit。因此返回的标签不是严格意义上“最终更新中心再次分配”的标签。原Stage2C没有保留原迭代标签/优化过程，不能从最终19中心复原迭代路径、初始化后中心移动或停止原因。

本阶段定义`FINAL_CENTER_VORONOI_ASSIGNMENT`：a_j=argmin_k ||P_j-C_k||²，使用最终冻结中心，分块512点，完全等距时选择原head编号较小者。不调用原KMeans或重新迭代，不称ORIGINAL_LAST_ITERATION_CLUSTER_LABELS。

## 计数和尺度

实际测量{len(cl):,}个最终Voronoi单元，每目标19个，人口总和每目标10000。空单元{int(cl.Empty.sum())}。保存每簇人口、唯一位置、局部25px人口、半径均值/中位/P90/最大值、非局部点GT距离以及最终中心到重新分配样本均值的距离。后者中位{cl.CenterToAssignedEmpiricalMean.median():.8g}px，P99 {cl.CenterToAssignedEmpiricalMean.quantile(.99):.8g}px，最大{cl.CenterToAssignedEmpiricalMean.max():.6f}px。数值不完全为0，不能把冻结中心精确等同于最终重分配的均值，也不能据此宣布KMeans程序错误。

局部抽样统计重复点次数，唯一位置另列；86失覆盖目标共有{summary['local_assignment_rows']}条25px内抽样分配行，见local_proposal_assignments.csv.gz，保留原proposal_index和SHA。834成功、77零支持、1正质量遗漏全部保留。

结果：OBSERVED。历史保护清单继承到Stage2C新增缓存；前后检查见history_before/after.json。大型原缓存、论文PDF和原始数据不上传Git。
''')
    put('02_COMPRESSION_FAILURE_MECHANISM.md',f'''# 压缩失覆盖机制：局部稀少，不是普遍大簇吞并

{boundary}

## 全量结果（OBSERVED）

86压缩失覆盖目标身份与旧阶段完全一致。25px内抽样数范围1—323，中位50，均值70.34884；834覆盖成功目标中位2299、均值2807.39448。67/86（77.91%）局部抽样<=100次即全部抽样的1%，7/86<=10次；这是事先冻结的描述阈值，不是密度因果判定。

{table(med)}

34/86的局部点只触及一个最终单元；其余52触及多个单元。50/86局部点至少80%归入同一单元，中位最大份额0.911687。局部点到其所归最终中心平均距离的目标中位27.62013px，对照9.18269px；局部抽样加权的簇P90半径目标中位29.04614px，对照14.31341px。

“大簇”预先定义为人口>=同目标非空19簇人口中位数，仅指相对人口而非绝对空间密度。失覆盖组局部抽样落入此类簇的比例，先每目标算再等权，均值19.50%、中位0；对照组中位93.52%。失覆盖组局部抽样加权簇人口中位343，对照741.24，方向与“失败普遍因并入更大的簇”相反。可以观察个别大簇吸收案例，不能推广为通用机制（NOT_SUPPORTED）。

## 数学解释（INFERRED）

原始重复抽样已经使经验KMeans近似最小化E_p[min_k ||P-C_k||²]。因此不能说“KMeans完全没有利用概率”；重复点频率已编码质量。再按p对每个重复点加权可能近似重复加权到p²，这不是自然的概率修复。

平方失真目标不保证任意半径r内的局部覆盖。对最终分配单元的经验均值μ_k，精确分解μ_k=α μ_local+(1-α) μ_other，α为该单元内局部点占比；它可解释少数局部点与远处均值共存。冻结中心C_k与μ_k有已记录的非零残差，不能把这条分解冒充原中心移动路径。没有原始迭代轨迹，Q3只能给静态分布解释，不能证明某中心“原来在GT旁后来被拉走”。

失覆盖组内，局部抽样数与最近中心距GT的描述Spearman为-0.78465；局部加权簇人口与局部至中心距离为-0.05229，整体云RMS与中心距GT为0.17322。前者支持局部质量关联，后两者不支持“单靠大簇/全局散布即可解释一切”。相关无p值，不作为因果，也不把86与834两个由结果定义的事后群体当随机分配。

## 每录像结果

{table(r,['scope','targets','compression_fail','compression_loss_rate_all','compression_loss_rate_proposal_hit'])}

全部998窗口加权压缩失覆盖率8.6172%；五录像等权为6.31836%（见recording_macro_rates.csv）。条件于原抽样命中的分母另列，不混用。每个录像都有压缩损失，但gates/hyang/nexus仅1/2/3例。deathCircle为65/443，其他4合计21/555；这是发生率差异，不是单录像代表SDD。

{table(byscene)}

各录像失败组局部抽样中位26—83，稀少方向相似；样本过少不支持跨场景稳定性确认。历史速度、转向、变速和静止所有组都保留在motion_comparison.csv（含0计数）；分组仅沿用历史特征，不用未来定义。

## 冻结真实案例

三个案例全部沿用Stage2C固定ID分位秩，不按美观替换；它们均来自deathCircle，不用它们代表五录像。图03主图为GT邻域放大，内嵌显示完整10000采样云及全部19中心。GT仅离线标记。

track260/frame1536：局部13次、最近中心52.02105px；track497/frame5004：局部189次、中心31.31582px，是局部点主要落入相对大簇的反例情形；track742/frame2316：局部18次、中心39.20136px。全部实际坐标/索引/概率SHA、采样SHA、中心SHA及ADE/FDE见frozen_case_data.json。

结论：局部较低抽样质量、少数最终单元、较宽局部单元半径与失覆盖共现（OBSERVED）；它们与平方失真目标不保证半径覆盖相容（INFERRED）。任何新代表点算法能否改善路径误差仍UNVERIFIED。
''')
    quant=pd.DataFrame([dict(quantile=q,raw_mass=float(z.unthresholded_normalized_mass.quantile(q)),analytic_hit10000=float(z.unthresholded_theoretical_hit10000.quantile(q))) for q in cfg['quantiles']])
    quant.to_csv(OUT/'threshold_mass_quantiles.csv',index=False)
    put('03_THRESHOLD_SUPPORT_REVIEW.md',f'''# 77个阈值后零支持目标的静态审查

{boundary}

77/77原始Sigmoid在GT25px邻域具有正权重；全部正权重邻域像素被原rel_threshold=0.01切除。邻域最大权重/整图最大权重的最大值为{z.neighborhood_max_over_global_max.max():.10f}，低于0.01。每个目标的原始权重和、数学归一化邻域质量、正像素及移除像素数量见threshold_support_review.csv。没有修改原阈值、取消mask、抽样或产生新候选。

{table(quant)}

未阈值化质量中位7.73295e-5（0.007733%），最小1.73935e-23、最大0.00286586；量级差异很大。只看相对质量很小，不能断言10000次抽样都不可能命中。对原Sigmoid数学归一化后，固定分布独立有放回10000抽样的理论命中概率中位0.538524、P25为0.054603、P75为0.994099。

以上理论均为`ANALYTIC_COUNTERFACTUAL_ONLY`。归一化仅CPU静态算式p_raw=s/sum(s)，实际旧预测仍使用阈值后的权重；没有观察此反事实的真实抽样、更没有KMeans或Path Decoder结果。无法宣称取消阈值会恢复77例、降低ADE/FDE或修复Goal网络。原始热图未被证明是校准概率。77例并非原Sigmoid支持严格为0，不能写成“Goal网络完全无效”。

OBSERVED：实际阈值支持消失。INFERRED：相对阈值具有异质的局部质量代价。UNVERIFIED：改变阈值的端到端收益。NOT_SUPPORTED：取消阈值必然改善预测、语义错误已经证实。
''')
    literature_blocks=[]
    for row in lit.to_dict('records'):
        literature_blocks.append(f"### {row['key']}（{row['year']}）\n\n[{row['title']}]({row['official_link']})；{row['authors']}；{row['venue']}。输入：{row['inputs']}；输出：{row['outputs']}。{row['clustering_postprocessing']}。与本题：{row['overlap']}；区别：{row['difference']}。风险{row['novelty_risk']}。证据FULLTEXT_METHOD_REVIEWED，锚点“{row['method_anchor']}”、文本行{row['method_text_lines']}及PDF/text SHA见literature_review.csv；版本{row['fulltext_version']}。{row['notes']}。")
    put('04_LITERATURE_AND_NOVELTY_REVIEW.md',f'''# 候选生成、压缩与新颖性审查

检索日期2026-10-22。使用nature-academic-search多源工作流；专业academic MCP未挂载。使用作者/CVF/arXiv原文及作者代码链接，OpenAlex降级检索因系统Python兼容性失败，改用既有Python3.10后HTTP429，未反复重试。记录见discovery_*failure.txt和literature_search_audit.json。12篇主论文全文下载到忽略缓存并阅读相关方法/公式；PDF、文本SHA与方法定位提交，全文不上传。不是穷尽性系统综述，不提供“已证明原创”结论。

{chr(10).join(literature_blocks)}

## 必须分清的先例

DenseTNT的目标是sum_x p(x) min_c ||x-c||，直接涉及概率加权有限Goal集合；HYPER在轨迹预测中实际使用FPS，故FPS不只是无关数学背景；DPP明确讨论高概率主要模式与少数模式的质量取舍；NPSN已经以历史路径条件化有限采样；TUTR已经讨论聚类后处理丢失输出概率。TUTR在训练数据上聚类运动模板，不能误写为任何阶段都无聚类。

2024 LRVQ §4.4使用200路径到20个KMeans质心。发表来源由作者官方README核为ICPR2024，不沿用初步ECCV猜测。2026 GMM工作§5.1按模式概率加权合并，§5.2一步EM需要训练集GT做M-step；不能把两者都称为无训练无GT。其实验为nuScenes和WOMD，不把参考文献里的Argoverse2误作实验集。GoalFlow为自车规划、PDMS任务，相近思路不等于行人同任务。检索中的SIPTraj同名资料有歧义，未用不同论文互相佐证。

## 六方向创新冲突矩阵

{table(matrix,['direction','proposal','closest_work','current_evidence','innovation_risk'])}

A—E均HIGH：直接替换KMeans、增多样性、加概率权重、保护尾部、引入历史条件都有明显先例；这不等于本次证明每一种特定实现完全相同。F为UNKNOWN新颖性且没有当前语义因果证据，因此不准入。LOW在本次矩阵没有使用。K-center/FPS、k-medoids、DPP、量化的不同目标见mathematical_prior_art.csv；书/经典论文只作为数学背景，未冒充已在同一行人任务实现。

当前链路重复样本已经包含经验概率质量，不能用“原聚类不含概率”建立虚假的创新缺口。单纯改名、套FPS、加权KMeans或常规medoid不构成原创。尚未找到区别明确、证据充分的新数学机制（UNVERIFIED）；不强凑第二个创新假设。
''')
    put('05_NEXT_EXPERIMENT_DESIGN.md',f'''# 唯一后续受控对照建议（未执行）

状态：PROPOSAL_ONLY。需要新的用户指令及独立冻结协议。本阶段无算法实现、新Goal、新采样、模型推理、路径解码或训练。

## 待验证假设H1

问题：固定采样云和最终Voronoi分区，均值代表换为单元内真实已有样本代表，是否改变少数局部覆盖及完整路径质量？证据：86例已有局部采样但最终20目标未覆盖；其局部抽样中位50，局部单元P90半径较大。反证风险：局部质量本身很小，多数簇内样本代表仍可能远离GT；质心能降低噪声（LRVQ已有论点），投影可能更差。

设Z_k是最终冻结Voronoi单元，定义m_k=argmin_(p∈Z_k) sum_(q∈Z_k)||q-p||²。保留重复抽样计数、以原抽样下标打破完全相同损失，空单元保留原中心（现有审计无空单元），不为去重额外补样。它是固定分区平方medoid控制，数学上等价于找离该分区经验均值最近的已有采样点；不是新算法，也不是全局k-medoids重新聚类。与仅找离旧C_k最近的点有潜在微小差异，不能混写。

部署输入只有已有10000点、19个冻结中心、原Softargmax和原解码条件；GT从不参与代表点选择。保留第0个Softargmax，替换其余19个，仍K20。没有新增训练、损失或参数。当前只给公式，不写选择实现。

## 最低成本公平实验

固定Epoch30/网络SHA、10000实际原抽样、998个ID、历史、GT、温度、阈值和全部预算。原KMeans基线来自旧冻结缓存，只读；替代目标和路径保存全新目录。必须从新Goal调用同一冻结Path Decoder获得20条新路径，不能拿旧20条路径冒充新Goal对应ADE/FDE。GT评分独立CPU后处理。

主要记录Goal覆盖10/25/50px、GoalFDE20、完整minADE20/minFDE20、候选分散度、经验平方失真、固定分布的半径覆盖质量、CPU/GPU耗时和显存。保留五录像、deathCircle/其他4、全部历史运动组，以及998窗口加权与五录像等权。不给选择器GT、失败标签或oracle head。

建议未来冻结门槛：五录像等权25px覆盖提升>=1个百分点，等权minADE20改善>=1%，等权minFDE20退化<=1%，且其他4录像合计覆盖改善为正；任一不满足，否定本次普通代表控制，不搜索不同medoid/FPS/radius/seed版本。即便通过，也仅证明开发条件下的普通对照有价值，不能证明原创或SCI泛化。

既有GPU完整解码均值0.124399秒/目标，998目标约124.15秒，仅是旧测量外推，排除加载、条件特征、I/O、校验；新方法CPU成本和完整GPUwall未测。无需训练，但没有擅自批准预算或运行。

## 独立确认与创新门槛

998目标已反复用于选模和探索，不能确认泛化。后续应由用户单独划定从未用于算法选择的新来源录像，并先以manifest元数据确认与既有训练/开发录像和完整20帧时域支持隔离；不得擅用封存测试。如果没有可用独立数据，明确无法完成独立确认，不把旧开发或模型已训练样本称为独立。

创新实验准入NO-GO：当前只有常规控制假设，无区别明确的新数学机制；近年相关工作重合大。不是要求换基线/换数据集，没有证据证明YNet不能用于此机制对照。一次严格冻结的普通对照比无止境继续解释均值更有信息价值，但是否执行交由科研决策AI；本阶段停止。
''')
    questions=[
      ('Q1','OBSERVED','局部抽样中位50、局部涉及2簇、局部到中心距离中位27.62px、簇P90半径中位29.05px；具体见target/cluster_diagnostics。'),
      ('Q2','OBSERVED','67/86局部抽样<=100/10000；大多数局部质量较少，但1—323异质，不能称所有都是极少点。'),
      ('Q3','NOT_SUPPORTED','不能普遍称并入更大高密度簇；大簇局部份额目标均值19.5%、中位0，簇人口中位反而低于成功组。'),
      ('Q4','INFERRED','经验均值局部/非局部分解与宽单元相容，最终重分配中心残差已量化；无原迭代日志，不能恢复中心移动原因。'),
      ('Q5','OBSERVED','五录像都有失败，但65/86来自deathCircle；其他录像1—15例；支持相似静态现象，不支持跨场景因果一致性。'),
      ('Q6','OBSERVED','77均有正原始质量，中位7.73e-5，理论10000未阈值化命中概率中位0.5385；并非全部可忽略，未做抽样反事实。'),
      ('Q7','OBSERVED','YNet/DenseTNT/HYPER/DPP/NPSN/TUTR及2024LRVQ、2026概率合并覆盖许多类似问题；不是标题推断。'),
      ('Q8','OBSERVED','直接KMeans替换HIGH风险；常规medoid/FPS/加权聚类不是新颖贡献；未证明任何具体替代方法原创。'),
      ('Q9','UNVERIFIED','历史运动关联已保留，但没有证明历史条件压缩提供独立价值；NPSN等已有直接条件采样先例。'),
      ('Q10','NOT_SUPPORTED','没有语义致错直接证据，不建议据此增加语义可靠性模块。'),
      ('Q11','UNVERIFIED','唯一建议是固定分区内已有采样点平方medoid代表控制；普通诊断对照而非新算法；性能未知。'),
      ('Q12','UNVERIFIED','固定10000/K20/网络/head0/目标集合；只变单元代表；新Goal重解码，完整路径和分录像评价；详见05。'),
      ('Q13','NOT_SUPPORTED','创新实验准入NO-GO：没有明确区别机制且确认数据未落实；不能自动进入Stage11C。'),
      ('Q14','UNVERIFIED','本轮证据不要求换基线或数据集；需先解决论文问题/独立确认设计，不能为推进阶段不断更换。'),
      ('Q15','INFERRED','获批后一次常规代表控制可直接证伪“均值代表形式”解释，比反复仅做相关统计更有信息；本轮没有执行。')]
    csvnew(OUT/'scientific_decisions.csv',[dict(question=a,status=b,answer=c) for a,b,c in questions])
    qa_text='\n'.join('**'+a+' '+b+'**：'+c+'\n' for a,b,c in questions)
    put('STAGE11B2D_REPORT.md',f'''# Stage11B-2D：目标候选压缩机制与创新可行性审查

**结论：NO_GO_INNOVATION；未达到Stage11C新算法准入条件。** 86目标机制分析完成，具备做一次普通代表点控制的证据，但没有足够新颖性依据。不改变Stage2C的MECHANISM_INCONCLUSIVE。

{boundary}

## 关键事实

全998缓存准入通过，Checkpoint/原预测SHA严格相同，20Goal/20路径逐位相同。新GPU秒、新模型推理、训练步均为0。历史SHA保护92,248项，最终核验见history_after.json。代码/协议已先提交冻结，全部统计只来自既有缓存。

86失覆盖目标共6050次局部抽样，25px局部中位50，对照834目标中位2299；67/86局部<=100次。34目标仅归入一个最终单元，50目标至少80%局部点归于同一单元。局部抽样加权簇人口中位343，对照741.24；簇P90半径中位29.05px，对照14.31px。**普遍被更大簇吞并不获支持**，少数局部样本与平方失真代表形式不保证半径覆盖的机制相容，但不证明因果或KMeans错误。

deathCircle65/443失覆盖，其他4合计21/555；窗口加权8.6172%，五录像等权6.3184%，不可混用。分录像数量bookstore15、deathCircle65、gates1、hyang2、nexus3，全部保留。只有5录像，不做独立样本显著性检验。

77零支持目标的原始邻域权重均>0，邻域最大/全图最大比均<0.01。原始数学归一化质量中位7.73295e-5，10000次未阈值化理论命中中位53.85%；这是ANALYTIC_COUNTERFACTUAL_ONLY，未改阈值、未抽样，不能推出端到端收益。

12篇主论文全文方法核查，含2024/2025/2026；六方向A—E高创新风险，F新颖性未知且缺当前因果证据。特别是DenseTNT概率加权有限Goal、HYPER预测样本FPS、DPP/NPSN有限覆盖、LRVQ过采样压缩与2026概率合并，均不能重新命名为本项目创新。

## 科研问答

{qa_text}

## 交付与复核

报告01缓存/分配定义、02失覆盖机制、03阈值支持、04文献冲突、05唯一未执行实验方案。结果目录包含998目标表、18,962单元表、6050局部抽样分配、五录像/宏平均/历史组、12论文矩阵及六方向创新冲突。图01—05均PNG/SVG，案例完全复用前阶段固定ID，原坐标不改，SHA和坐标见frozen_case_data.json。最终单元分配不是原迭代标签。

测试结果与失败/跳过原因见targeted_tests、full_tests日志/XML；首次目标测试遇到系统ROS自动加载插件缺lark，保留失败并使用既有禁用外部插件设置运行，没有改历史测试。论文检索OpenAlex失败也保留，没有虚称所有源成功。大型张量、PDF、原始数据、权重不提交。

完整执行命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b2d.analyze`（只能在既有空新结果目录初次执行，使用冻结协议和源码）；单元测试及交付SHA详见execution_manifest.json。未运行任何新候选压缩算法，不自动进入Stage11C。
''')
    print('REPORTS 6; final NO_GO_INNOVATION; single ordinary control proposed, not executed',flush=True)

if __name__=='__main__':main()
