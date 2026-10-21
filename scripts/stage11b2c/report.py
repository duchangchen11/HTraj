"""Reports from actual paired exports and frozen event definitions only."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import numpy as np,pandas as pd
R=ROOT/'reports/stage11b2c'
def table(df,cols):
    h=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df[cols].itertuples(index=False,name=None):h.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(h)
def write(name,s):
    with (R/name).open('x') as f:f.write(s.strip()+'\n')
def main():
    d=json.loads((OUT/'final_decision.json').read_text());mass=pd.read_csv(OUT/'probability_mass.csv');prop=pd.read_csv(OUT/'proposal_coverage.csv');final=pd.read_csv(OUT/'final_goal_coverage.csv');soft=pd.read_csv(OUT/'softargmax_kmeans_comparison.csv');s=pd.read_csv(OUT/'scene_mechanism_summary.csv');g=pd.read_csv(OUT/'motion_mechanism_summary.csv');ev=pd.read_csv(OUT/'coverage_transition.csv');ty=pd.read_csv(OUT/'mechanism_type_summary.csv');f=pd.read_csv(OUT/'sample_level_mechanism.csv');macro=pd.read_csv(OUT/'recording_macro_summary.csv');sg=json.loads((OUT/'softargmax_formula_audit.json').read_text());gen=json.loads((OUT/'generation_execution.json').read_text())
    common='适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。'
    allr=s[s.scope=='ALL_998'].sort_values('radius');s25=s[s.radius==25];typ=ty[(ty.scope=='ALL_998')&(ty.radius==25)&(ty.kind=='exclusive_category')]
    write('02_PROBABILITY_MASS_ANALYSIS.md',f'''# Goal权重与GT邻域离散概率

{common}

保留原logits、Sigmoid权重、阈值后未归一化权重、实际torch.multinomial使用的float32权重。Sigmoid不等于全局概率。统计中p=w/sum64(w)仅为数学离散分布定义，不反馈给原采样器；原float32权重及其总和单独留存。GT-neighborhood坐标为network/.25，包含原画布全部支持，同时区分真实图像与padding。

25px质量均值{mass.P_mass25.mean():.8g}、中位数{mass.P_mass25.median():.8g}；分位数P10/P90={mass.P_mass25.quantile(.1):.8g}/{mass.P_mass25.quantile(.9):.8g}。零质量目标{int((mass.P_mass25==0).sum())}/998。以固定10000次有放回理论P_hit<.5为解析低质量参考时，共{d['low_mass_analytic_reference_count']}个；这不是事后调出的分类阈值，也不等同实际未命中。

{table(allr,['radius','targets','P_mass_mean','P_mass_median','support_zero_count','low_mass_reference_count','theoretical_P_hit_mean'])}

各目标支持像素数、熵（自然对数）、max概率、top1%画布概率集中度、原Sigmoid最大值/和、阈值权重和、实际float32权重double求和、padding质量及GT画布内外标记均见probability_mass.csv和sample_level_mechanism.csv。高熵并不证明校准正确；质量为0表示实际阈值后离散支持缺失，不表示原始Sigmoid必为0，更不能证明语义错误。GT是否在画布内、是否在真实图像内分开标注。

原始张量保持float32/int64，无量化，目录在忽略的experiments/stage11b2c/cache/tensors；每文件及各中间张量SHA见tensor_manifest.csv。CPU评分只在generation_execution.json和process_extract.json完整通过后启动。
''')
    write('03_PROPOSAL_COVERAGE_ANALYSIS.md',f'''# 原始10000次抽样覆盖

{common}

每sample_id独立CPU Generator和冻结SHA seed，有放回10000次，重复样点未去重再聚类。唯一位置数均值{prop.unique_proposal_locations.mean():.2f}，重复抽样次数均值{prop.repeated_proposal_draws.mean():.2f}；10000不是10000个不同位置。

{table(allr,['radius','proposal_cover_count','proposal_cover_fraction','positive_mass_sampling_miss_count','theoretical_P_hit_mean'])}

25px实际覆盖{d['proposal_covered']}/998={d['proposal_coverage']:.6%}；正质量但实际抽样未命中{d['positive_mass_actual_sampling_miss']}个。每目标实际最近距离、各邻域命中次数和索引/坐标SHA见proposal_coverage.csv。P_hit=1-(1-P_mass)^10000使用稳定expm1/log1p计算，仅是固定离散分布独立有放回假设下的理论量；不能用它替代真实命中事件。

所有sample_id原probability/weights/proposal/goal/path SHA与旧Epoch30一致，全部998反序二次生成的原始精度中间张量逐位相同。采样与GT评分相隔独立进程，GT未选择随机数或样点。
''')
    write('04_KMEANS_COMPRESSION_ANALYSIS.md',f'''# 原始抽样→19中心+Softargmax覆盖变化

{common}

最终25px覆盖{d['final_covered']}/998={d['final_coverage']:.6%}，与原164个A25失败身份逐样本一致；10/25/50px失败数为489/164/99。原10000点命中但最终K20未命中{d['compression_loss_count']}个。最终失败中零质量共现{d['final_fail_zero_mass']}个、正质量抽样未命中{d['final_fail_positive_mass_sampling_miss']}个、抽样命中后最终失去覆盖{d['final_fail_proposal_previously_covered']}个。它们是观察到的链路事件，不能直接当作模块造成的ADE因果份额。

{table(typ,['type','targets','fraction','fixed_A25_count','fraction_of_fixed164','minADE20','minFDE20'])}

coverage_transition.csv保留3半径×998目标的A/B/C；all_event_combinations.csv保留所有8种组合（不存在的组合计数0、误差均值不伪填0）。T1—T6命名指标有重叠（例如正质量但抽样未命中与最终反而覆盖），不可直接相加；上述exclusive_category是互斥统计。半径10/50的fixed164比例仍以原A25固定164身份为分母，不能当作各阈值自身失败群体。

原始采样未命中而最终中心/Softargmax反而命中共{d['final_recovery_without_proposal_hit']}个（25px）。本次未观测反向恢复，因此这批数据中的最终覆盖是原始抽样覆盖的子集；但算法定义不保证该性质。KMeans中心不是必然从输入样点中选取；均值落入原抽样未命中邻域并非算法错误。原输入10000点的重复性/密度未改变，19个中心顺序不变。

Softargmax补充KMeans19在25px漏掉的目标{d['softargmax_rescues_KMeans25']}个。独立Softargmax/KMeans19/combined20的各阈值命中及距离见softargmax_kmeans_comparison.csv；这是同一冻结集合内来源分解，不是改变K后重训的公平模型实验。

静态源码发现：SoftArgmax2D对exp(logit-max)/(sum(exp)+1e-6)进行坐标加权，原epsilon未改；它不是阈值化Sigmoid采样分布的均值。附加float64公式与native位置差最大{sg['max_float64_native_coordinate_difference_px']:.8g}px（仅反映独立CPU/GPU数值归约差异，不要求该重算逐位一致；真正逐位门槛由原预测/新导出回归满足）。softargmax_formula_audit.csv记录两种分布的差异、Softargmax附近质量和坐标均值，但没有预注册模式数量/干预测试，故“多峰平均导致失败”仍UNVERIFIED，不能据GT距离或单张图证明。
''')
    write('05_SCENE_AND_MOTION_ANALYSIS.md',f'''# 录像与冻结历史运动分层

{common}

{table(s25,['scope','targets','P_mass_mean','proposal_cover_fraction','final_cover_fraction','final_fail_support_zero','final_fail_positive_sampling_miss','final_fail_after_proposal_hit','compression_loss_fraction'])}

五录像等权宏平均单独保存，不能与998窗口加权混用：

{table(macro,['radius','P_mass_mean','proposal_cover_fraction','final_cover_fraction','compression_loss_fraction'])}

固定deathCircle443与其他4合计555同时报告。最终A25失败来源原样保持deathCircle133/164；不能用单录像高误差推出语义错误。原像素距离和参考图对角线归一化距离都保存在样本表及scene_mechanism_summary.csv。画像性质受录像构成和重复窗口影响。

{table(g[g.radius==25],['factor','group','targets','proposal_cover_fraction','final_cover_fraction','compression_loss_fraction','minADE20'])}

所有运动组来自Stage11B-2A冻结历史8帧标签，未用未来运动定义分组；frozen源SHA见protocol_frozen.json。类型与误差关联见mechanism_type_summary.csv、sample_level_mechanism.csv，均不能作因果归因。
''')
    answers=[
    ('原始热图支持','SUPPORTED',f"998目标均保存真实权重与支持。25px实际阈值后支持为0的目标{d['all_targets_zero_mass']}个；熵不代表校准。"),
    ('GT25px概率质量','SUPPORTED',f"均值{d['P_mass25_mean']:.8g}、中位数{d['P_mass25_median']:.8g}，逐目标见probability_mass.csv。"),
    ('10000抽样覆盖','SUPPORTED',f"{d['proposal_covered']}/998={d['proposal_coverage']:.6%}，来自真实抽样点。"),
    ('最终K20覆盖','SUPPORTED',f"{d['final_covered']}/998={d['final_coverage']:.6%}，旧缓存逐位一致。"),
    ('164失败中的原始命中','SUPPORTED',f"{d['final_fail_proposal_previously_covered']}/164；GT仅离线用于距离计算。"),
    ('最终失败与支持缺失','SUPPORTED',f"{d['final_fail_zero_mass']}/164共现，不等于原始Sigmoid为0或语义因果。"),
    ('原始命中后的覆盖丢失','SUPPORTED',f"{d['compression_loss_count']}目标；不证明KMeans实现有错。"),
    ('Softargmax补充作用','SUPPORTED',f"25px补救{d['softargmax_rescues_KMeans25']}目标。其多峰均值失效原因仍UNVERIFIED。"),
    ('deathCircle机制','PARTIALLY_SUPPORTED',f"主事件分布{d['recording_mechanisms']}；仅开发录像描述，不能泛化到全部SDD。"),
    ('高速/转向/变速关联','PARTIALLY_SUPPORTED','全部历史组见motion_mechanism_summary.csv；组均值有录像/窗口构成混杂，不证明运动状态导致覆盖损失。'),
    ('优先研究哪一环节',d['mechanism_evidence'],d['recommendation']+'；依据预先冻结的最终失败最大类及deathCircle/other4一致性，不是创新GO。'),
    ('语义不可靠直接证据','UNVERIFIED','没有受控语义干预或可靠语义GT，不能从当前热图推断。'),
    ('提出新方法的机制证据','PARTIALLY_SUPPORTED','仅具备本冻结链路的实证观察；新机制效果、可部署性与新颖性尚未验证，不能直接提出新算法。'),
    ('还需阅读哪些工作','UNVERIFIED','需复核Y-Net原TTST、TUTR聚类后处理替代、DLow集合多样性目标，以及受限候选预算下量化/覆盖代表点选择与概率保持方法。当前不是完整查重。'),
    ('下一阶段最低成本实验','PARTIALLY_SUPPORTED','若获批准，最低成本先使用当前缓存做簇级分配、簇半径与命中样点到最终中心距离的离线分解，并复核公开压缩/覆盖方法；同时区分已有相对阈值切除与网络分布低质量。不改变阈值、不重采样、不训练；有明确机制和查重后再冻结同K预算的独立对照。')]
    refs='''[Y-Net原生成源码](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/test.py)已明确使用大量抽样、KMeans和Softargmax；因此这些常规构件不能换名作为创新。[TUTR正式论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf)已讨论聚类后处理的代价及概率信息问题；[DLow作者论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540324.pdf)研究多样化潜变量集合，但原任务是3D人体运动，不能直接称为相同SDD候选压缩实验。来源与访问限制见source_references.json；本轮Y-Net CVF链接返回403，未将其误记为全文审读成功。'''
    write('06_RESEARCH_DECISION.md','# 机制与科研决策\n\n'+common+'\n\n'+'\n\n'.join(f'**Q{i+1} {title} — {state}**：{answer}' for i,(title,state,answer) in enumerate(answers))+'\n\n'+refs+f'\n\n最终机制建议 `{d["recommendation"]}`；算法创新UNVERIFIED；新算法开发/训练NO-GO，自动下一阶段NO-GO。停止，等待审查。')
    write('STAGE11B2C_REPORT.md',f'''# Stage11B-2C：Goal分布与候选生成机制审计

{common}

**完成998目标正序导出和全部998目标反序逐位回归。** 原probability/weights/proposal/goal/path五项SHA及20目标/20路径逐位一致，额外导出未改变Epoch30预测。模型状态SHA `{gen['model_state_SHA']}` 未变化，无训练步。

{table(allr,['radius','proposal_cover_count','proposal_cover_fraction','final_cover_count','final_cover_fraction','compression_loss_count','final_recovery_count','support_zero_count'])}

25px原始10000抽样覆盖{d['proposal_covered']}，最终K20覆盖{d['final_covered']}；最终失败164个中，零支持{d['final_fail_zero_mass']}、正支持但抽样未命中{d['final_fail_positive_mass_sampling_miss']}、抽样命中后最终失去覆盖{d['final_fail_proposal_previously_covered']}。不得将三组误差均值差解释成模块因果贡献。

GT25px概率质量均值{d['P_mass25_mean']:.8g}、中位数{d['P_mass25_median']:.8g}。Softargmax在25px补救KMeans19遗漏{d['softargmax_rescues_KMeans25']}个。最终反而命中而原样点未命中{d['final_recovery_without_proposal_hit']}个，本次没有观测到最终覆盖突破原始抽样覆盖的反例；算法定义仍不保证子集关系，不能把未观测当成数学约束。

机制优先建议 `{d['recommendation']}`；跨deathCircle与其他4录像的最大类一致性={d['same_dominant_in_deathCircle_and_other4']}。这是观察性链路审计，不能证明新算法可提升预测，也不能证明语义因果。

GPU新增总进程wall {d['GPU_seconds']:.3f}秒，包含启动、加载、采样、聚类、完整解码、张量导出、回归和退出，低于7200硬限/6600安全线。原精度大张量仅保存在忽略cache，Git不含原始数据或权重。模型/数据/原缓存及全部中间SHA、时间、显存与失败日志分别见input_export.json、tensor_manifest.csv、generation_execution.json、gpu_budget.json和cache_integrity.json。

报告01为原链路与GT隔离；02概率质量；03原始抽样；04压缩与Softargmax；05全录像及历史分组；06逐项科研问答与来源。图件PNG/SVG与全部冻结规则案例及sidecar身份/SHA保留，缺失类型不虚构案例。

验收记录见full_tests_final.txt/xml、history_after.json和delivery_audit.json。已有语义预训练与协议限制未解除；没有测试集评分、新训练、改采样或新模型。新算法开发NO-GO；仅可在后续批准后开展机制与文献独立复核。''')
    print('Actual-result reports generated',flush=True)
if __name__=='__main__':main()
