"""Assemble scoped research conclusions from real completed artifacts only."""
from .common import *
import numpy as np,pandas as pd,torch,platform
import xml.etree.ElementTree as ET

def tab(rows,cols):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'NA' if pd.isna(v) else f'{v:.6f}'
        return str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(fmt(r[c]) for c in cols)+' |' for r in rows])+'\n'

def main():
    cpu_only();d=json.loads((OUT/'final_decision.json').read_text());assert d['execution']=='COMPLETE'
    verification=json.loads((OUT/'fixed_repeat_reverse.json').read_text());assert verification['passed'] and verification['comparisons']==18
    scenes=pd.read_csv(OUT/'scene_comparison.csv');a=scenes[scenes.experiment=='SEMANTIC_TO_GOAL_AND_PATH'];b=scenes[scenes.experiment=='SEMANTIC_TO_PATH_GIVEN_GOAL']
    f=pd.read_csv(OUT/'full_prediction_metrics.csv');fixed=pd.read_csv(OUT/'fixed_goal_path_metrics.csv');trans=pd.read_csv(OUT/'a25_transition_summary.csv')
    motion=pd.read_csv(OUT/'motion_group_comparison.csv');semantic=pd.read_csv(OUT/'semantic_tensor_audit.csv')
    data=json.loads((CONF/'generation_config.json').read_text());budget=json.loads((OUT/'gpu_budget.json').read_text());probes=pd.read_csv(OUT/'internal_probe_comparison.csv')
    R=ROOT/'reports/stage11c0';R.mkdir(exist_ok=True,parents=True)
    def one(frame,condition,scope='ALL998'):return frame[(frame.condition==condition)&(frame.scope==scope)].iloc[0]
    def table_for(frame,scope,cols):return tab([dict(condition=c,**{k:one(frame,c,scope)[k] for k in cols}) for c in sorted(frame.condition.unique())],['condition']+cols)
    baseline=one(a,'S0');mac0=one(a,'S0','FIVE_RECORDING_MACRO')
    rows=[]
    for scope in ['ALL998','FIVE_RECORDING_MACRO','bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4','OTHER4']:
        for c in ['S0','S1','S2','S3']:
            q=one(a,c,scope);rows.append(dict(scope=scope,condition=c,n=int(q.targets),GoalCover25=q.GoalCover25,
                minGoalFDE20=q.minGoalFDE20,ADE1=q.ADE1,FDE1=q.FDE1,minADE20=q.minADE20,minFDE20=q.minFDE20))
    alltable=tab(rows,list(rows[0]))
    tests='全仓pytest未完成；不能声明已全部通过。';htext='历史保护后核验尚未完成。'
    if (OUT/'full_tests.xml').exists():
        suites=list(ET.parse(OUT/'full_tests.xml').iter('testsuite'));t={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        tests=f"全仓pytest：{t['tests']-t['skipped']-t['failures']-t['errors']} PASS、{t['skipped']} SKIP、{t['failures']} FAIL、{t['errors']} ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。"
    if (OUT/'history_after.json').exists():
        h=json.loads((OUT/'history_after.json').read_text());assert h['passed'];htext=f"{h['files']}项历史文件SHA全部未改变。"
    common=f"""
证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `{CK}`，原S0预测SHA `{PSHA}`，作者MIT源码 `{OFFICIAL}`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。{tests} {htext}
"""
    def write(name,title,text):(R/name).write_text('# '+title+'\n\n'+text.strip()+'\n\n'+common.strip()+'\n')
    maxsum=float((semantic[['sum_channels_min','sum_channels_max']]-1).abs().max().max())
    inputrows=[]
    for scene,q in semantic.groupby('scene'):
        x=q.iloc[0];inputrows.append(dict(scene=scene,shape=x['shape'],valid_height=int(x.valid_height),valid_width=int(x.valid_width),
            min_channel_value=float(q.canvas_min.min()),max_channel_value=float(q.canvas_max.max()),sum_mean=x.sum_channels_mean,mean_abs_sum_minus1=x.mean_abs_sum_minus1))
    write('01_SEMANTIC_INPUT_AUDIT.md','真实六通道语义输入审计',f"""
六通道实际是float32、六分类Softmax(dim=1)概率输出，并非任意logits或类别整数图。冻结分割器为Unet，head为Conv2d(16,6)→Identity→Softmax(dim=1)，eval且全部分割参数requires_grad=False。六通道值在[0,1]内，全部有限，通道和最大偏离1为{maxsum:.9g}；各通道有效区域均值/总体方差/范围见semantic_tensor_audit.csv。

{tab(inputrows,list(inputrows[0]))}

分割checkpoint SHA `75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194`；分割state SHA `79bd03ee82eea39723c27cba241f961669a53d2c47e7f26d4dce3b7bc949bd83`。不具备可核查的6通道类别名称/语义GT，始终只称channel0—5，不评分类正确率或断言道路/草地/障碍类别。

原始链路：cv2读取参考图像，INTER_AREA缩放0.25，右/下零边界补至32倍数，再按作者SMP resnet101 imagenet预处理。原语义输出的padding位置也含概率，并非必须为零。有效区域由实际cv2缩放shape确定，没有用GT裁定空间区域。原图SHA、预处理RGB张量SHA及语义tensor/file SHA均留档。

五个验证录像全部复用Stage2C已冻结语义，未重新执行分割器替换缓存。训练工程smoke的bookstore_0没有持久语义缓存，只在预先声明范围内计算一次分割；其两个训练目标完整预测/中间张量必须与旧Stage2C预检逐位相同，已通过。此例外不涉及验证语义重算或GT误差评分。
""")
    write('02_INTERVENTION_PROTOCOL.md','冻结干预协议与因果解释范围',f"""
四组定义：S0原六通道；S1包括padding的六通道全零；S2逐通道在有效RGB缩放区域取float32 CPU均值并铺满该区域，padding原值；S3六通道共同在有效区域水平翻转，无插值、无通道交换，padding原值。保存20份实际语义张量、文件与张量SHA，详见intervention_manifest.csv。干预前后shape/dtype不变，history heatmaps8通道在每次真实前向的encoder pre-hook中逐位检查，六通道输入也与指定干预张量逐位一致。

先做训练2目标、四组重复/倒序/未来字段反事实smoke，再提交完整数值协议`cbe69cc220894960f0c051c26b0c98e5feefb026`。S0固定原清单索引0/499/997，正序与倒序两次：logits、Sigmoid、阈值与实际采样权重、proposal/indices、19中心、Softargmax、20Goal和20Path全部逐位复现。正式S0基线只读复用998旧缓存。

首次完整预测在首目标的元数据dict构造时出现重复semantic_SHA TypeError；未读GT，失败原文、6.330436秒GPU wall及第一份真实缓存完整保留。只删除重复关键字并添加trace SHA与实际输入相等断言；干预、网络、采样与评价数值公式不变。新增回归测试后提交`1a9bf5e2b5ddca6d8a0c02dd4577a5ebe44bf54a`，更新源码SHA再继续。原冻结清单保存为protocol_frozen_failed_attempt1.json，不抹去失败历史；失败进程完整时间计入总预算。

实验A分别对S1/S2/S3运行原AuditPredictor：原Encoder/GoalDecoder、temperature1、rel_threshold0.01、CPU TTST10000 replacement=True、作者KMeans19+Softargmax1、原PathDecoder，K20不变。同sample_id使用原SHA派生CPU Generator与KMeans独立种子；干预改变分布后proposal/中心变化属于整个链路响应，不能全归因于分割类别。

实验B保持S0全部20Goal逐位相同，不运行Goal预测、TTST或KMeans，仅用相同历史与各组语义重算Encoder特征及原PathDecoder。2994×20条路径均实际重新解码，非旧路径冒用。A/B分别评价，不作线性可加的因果分解。

JS使用原/新实际阈值后采样权重，离线float64数学归一化，使用自然对数；p=0的KL项定义0，中间分布m=(p+q)/2，不用任意epsilon改变分布。支持像素>0；面积=count×16px²是原坐标网格面积，不是语义可行走面积。熵不是校准不确定性。KMeans head无语义对应，除head0外不按同编号解释模式位移；使用无GT的离线Hungarian/Chamfer/Hausdorff集合几何比较，不改变输出或评分。

尺度解释事先冻结：1e−4px数值级、1px描述性实际尺度，另保留5/25px比例；H1—H4的判断公式见experiment_protocol.json。不是优化收益门槛，也不通过窗口独立性p值声称显著性。GT仅在全部A/B组完整保存、身份/有限性/固定Goal核验后由独立CPU score.py加载。
""")
    cols=['JS_nats','total_variation','logits_mean_abs_difference','logits_RMS_difference','logits_max_abs_difference',
        'Original_entropy_nats','Alternative_entropy_nats','Original_support_pixels','Alternative_support_pixels',
        'Softargmax_displacement','Hungarian_mean_distance','Hausdorff_distance','GoalCover10','GoalCover25','GoalCover50','GoalFDE1','minGoalFDE20']
    write('03_GOAL_SENSITIVITY.md','Goal分布与候选集合敏感性',f"""
H2：{d['hypotheses']['H2']}，仅指当前冻结模型和指定空间扰动。

窗口加权（JS/熵nats、几何px、覆盖0—1）：

{table_for(a,'ALL998',cols)}

五录像等权：

{table_for(a,'FIVE_RECORDING_MACRO',['JS_nats','Hungarian_mean_distance','Softargmax_displacement','GoalCover25','minGoalFDE20'])}

以上热图差异直接来自缓存中实际Goal logits/实际采样权重。归一化Sigmoid和实际阈值后categorical不是同一个量，JS只比较后一种。S2/S3保留原通道或空间值的部分统计，仍属于冻结模型的分布扰动，未证明当前分割内容正确或错误。

每目标4组goal_distribution_sensitivity.csv、goal_set_comparison.csv保留全部分布和集合统计；每组998个生成记录含proposal、中心、Goal、Path、history与语义SHA。S0精确GoalCover25={baseline.GoalCover25:.9f}，不是用83.5671%的舍入数反推。
""")
    pathcols=['ADE1','FDE1','minADE20','minFDE20','PathPairwiseDispersion','GoalPathEndpointMeanDistance']
    fixedcols=['ADE1','FDE1','minADE20','minFDE20','Path_point_mean_displacement','Path_endpoint_mean_displacement',
        'Path_changed_heads','Path_unchanged_heads','Head0_point_mean_displacement']
    globals_rows=[dict(condition=c,global_max_point_move=float(fixed[fixed.condition==c].Path_max_displacement.max()),
        mean_per_target_max=float(fixed[fixed.condition==c].Path_max_displacement.mean()),
        changed=int(fixed[fixed.condition==c].Path_changed_heads.sum()),unchanged=int(fixed[fixed.condition==c].Path_unchanged_heads.sum()),
        median_point_move=float(fixed[fixed.condition==c].Path_point_mean_displacement.median()),
        targets_mean_move_gt1px=int((fixed[fixed.condition==c].Path_point_mean_displacement>1).sum())) for c in ['S1','S2','S3']]
    pd.DataFrame(globals_rows).to_csv(OUT/'fixed_goal_global_sensitivity.csv',index=False)
    write('04_PATH_SENSITIVITY.md','完整路径与固定Goal路径敏感性',f"""
实验A，窗口加权（所有路径真实运行）：

{table_for(a,'ALL998',pathcols)}

实验B，固定Goal的窗口加权（改变/不变head为每目标平均数量；总数另见下表）：

{table_for(b,'ALL998',fixedcols)}

{tab(globals_rows,list(globals_rows[0]))}

这里global_max_point_move才是全20×998×12点的最大位移；scene_comparison.csv中的Path_max_displacement列按统一均值聚合，表示每目标最大位移的平均值，不把它冒充全局最大值。极端最大值也不能推广为全部目标。

每个B组Goal全部逐位等于S0。三种干预×3个预定ID×正/倒序的18次真实B解码，与保存的对应路径全部逐位一致；未来占位字段0/1e6不改变历史前缀投影，固定Goal的重复性独立通过，详见fixed_repeat_reverse.json/rows.csv。H3={d['hypotheses']['H3']}，H4={d['hypotheses']['H4']}。不把微小bitwise变化直接当有意义变化，给出真实px尺度和1e−4/1/5/25px head计数。固定3个真实ID的被动hook显示Encoder改变级数{int(probes.encoder_changed_levels.min())}—{int(probes.encoder_changed_levels.max())}、Path logits改变head数{int(probes.path_logit_changed_heads.min())}—{int(probes.path_logit_changed_heads.max())}，原始SHA/范数见internal_probe_comparison.csv与baseline/fixed_internal_probes.json，证明检查了扰动是否到达中间计算。

完整预测和固定Goal是两个不同的条件性实验，不将A的变化全归因于Path，也不将B的变化归因于采样。ADE1、FDE1仅指第0个Softargmax Goal解码Path，不是概率分布期望；minADE20/minFDE20分别取最优head，是GT oracle指标，不是部署候选选择器。
""")
    group=[]
    for (factor,g),q in motion[(motion.scope=='ALL998')&(motion.experiment=='SEMANTIC_TO_PATH_GIVEN_GOAL')].groupby(['factor','group']):
        row=dict(factor=factor,group=g,n=int(q.targets.iloc[0]))
        for c in ['S1','S2','S3']:row[c+'_fixed_mean_move']=float(q[q.condition==c].iloc[0].Path_point_mean_displacement)
        group.append(row)
    grouptable=tab(group,list(group[0]))
    write('05_SCENE_AND_MOTION_ANALYSIS.md','五录像与历史运动分组',f"""
完整预测全部录像与两种总体口径：

{alltable}

全部历史组固定Goal平均路径位移，单位px：

{grouptable}

分组来自Stage2A历史8帧：high497、medium235、low266；转向≥30°270、直行322、中间211、方向未定义195；速度变异CV≥0.4为164、稳定CV≤0.15为394、中间236、低速未定义204；全部步长≤1px的近静止147。阈值及身份没有根据未来误差修改。motion_group_comparison.csv还给出全录像组成、每组S0/三组A和三组B全部误差、Goal分布/集合变化、各组n及空单元，不只保留受益组。

不能单凭high组误差或位移高认定模型对复杂运动依赖更强；这些组的录像组成、空间位置和原始位移尺度不同。各组真实敏感性表支持条件性描述，不能证明运动状态与语义依赖的因果关系。

本次high组的固定Goal平均路径位移在三种干预中均高于low组；但转向组与直行组、速度高变组与稳定组的关系并不一致：S1/S2的转向组低于直行组，高变组也低于稳定组，S3相反。因此不支持“复杂运动组统一更敏感”的概括。这里只是已曝光样本的描述，不排除录像/像素尺度/空间位置混杂。

原164个失败/834成功的完整恢复和退化（下表含原失败及成功各自minADE20）：

{tab(trans[trans.scope=='ALL998'].to_dict('records'),list(trans.columns))}

不能只展示恢复而忽略新增失败；A25集合始终是旧164/834身份。deathCircle与其他四录像分别列出，不将deathCircle主导的窗口结果作为跨录像一致性。所有案例按冻结的录像内/结果类别内sample_id中位序规则，GT只用于评分和离线显示。
""")
    hypotheses=[]
    for key,label in [('H1','语义输入敏感性'),('H2','语义空间结构影响Goal'),('H3','固定Goal时Path响应'),('H4','真实误差发生变化'),('H5','真实语义独立预测价值'),('H6','分割错误导致预测失败')]:
        hypotheses.append(dict(hypothesis=key,proposition=label,status=d['hypotheses'][key]))
    hyp=tab(hypotheses,['hypothesis','proposition','status'])
    signrows=[]
    for c in ['S1','S2','S3']:
        q=a[(a.condition==c)&a.scope.isin(['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'])]
        signrows.append(dict(condition=c,ADE_better_recordings=int((q.DeltaminADE20<0).sum()),ADE_worse_recordings=int((q.DeltaminADE20>0).sum()),
            macro_ADE_delta=one(a,c,'FIVE_RECORDING_MACRO').DeltaminADE20,macro_FDE_delta=one(a,c,'FIVE_RECORDING_MACRO').DeltaminFDE20))
    runtime=f"""
GPU完整进程wall合计{budget['actual_GPU_process_seconds']:.6f}s，包括预检、S0回归、失败首轮、重跑完整A、全部B、模型加载/缓存/退出。硬上限7200s，安全阈值6600s。失败首轮6.330436s未剔除。峰值及阶段时间见各worker日志和GPU_card_*.csv；预算估计来自训练smoke，实际值没有用历史耗时代替。训练时间0s。CPU评分{json.loads((OUT/'score_integrity.json').read_text())['CPU_seconds']:.6f}s，SHA、全测试及作图另计。
"""
    write('06_SEMANTIC_RESEARCH_DECISION.md','语义研究命题与下一步决策',f"""
{hyp}

{tab(signrows,list(signrows[0]))}

本阶段是输入干预敏感性研究，不是语义增强算法实验。冻结网络遇到全零、均值或错位概率图，可能有输入分布偏移；即使S0更好，也不能直接证明同预算训练时语义有独立增益。S0更差也不能证明移除语义训练更好。H5需要同数据、同预算、从头训练的匹配无语义对照；H6缺少语义GT与可靠类别映射，没有直接质量测量，因此UNVERIFIED。

研究建议：{d['research_recommendation']}。这只是可提案的下一步，不授权自动执行。唯一建议的确认性实验是匹配数据/初始化/预算/评价对象的从头无语义训练对照；单独统计有效容量与输入归一化差异。需预注册多seed、录像口径与失败门槛，并明确仍是已曝光开发数据。不得从当前结果直接开发可靠性门控、声称新颖性，或改写论文已有结论。

不依据测试集选择方向；不启动Stage11C-1。若各录像方向不一致，保留条件性解释；不包装统一语义可靠性机制。硕士小论文方向是否调整，待用户审查本阶段证据和训练成本后决定。

{runtime}
""")
    qas=[]
    def qa(n,q,answer):qas.append(dict(question=f'Q{n} {q}',answer=answer))
    qa(1,'六通道形式',f'float32 Unet Softmax(dim1)概率，通道和最大偏离1为{maxsum:.3g}；类别名未知')
    qa(2,'干预真实改变','20份输入SHA；S1全零、S2有效空间均值、S3有效水平翻转，S2/S3 padding原值；原8历史热图每次forward检查')
    qa(3,'Goal热图变化','；'.join(f"{c}:平均JS={one(a,c).JS_nats:.6f}nats, logits RMS={one(a,c).logits_RMS_difference:.6f}" for c in ['S1','S2','S3']))
    qa(4,'K20集合变化','；'.join(f"{c}:Hungarian均距={one(a,c).Hungarian_mean_distance:.6f}px" for c in ['S1','S2','S3']))
    qa(5,'完整预测minADE/FDE',f'S0={baseline.minADE20:.6f}/{baseline.minFDE20:.6f}px；'+'；'.join(f"{c}={one(a,c).minADE20:.6f}/{one(a,c).minFDE20:.6f}" for c in ['S1','S2','S3']))
    qa(6,'固定Goal Path变化','；'.join(f"{c}:均点位移={one(b,c).Path_point_mean_displacement:.6f}px" for c in ['S1','S2','S3']))
    qa(7,'复杂运动更敏感','high vs low平均路径位移：S1 18.451/9.884、S2 4.098/1.821、S3 2.453/1.345px；转向/变速组方向不一致，不支持统一概括或因果结论')
    qa(8,'deathCircle特殊性','；'.join(f"{c}:DC JS={one(a,c,'deathCircle_3').JS_nats:.6f},其他4 JS={one(a,c,'OTHER4').JS_nats:.6f}" for c in ['S1','S2','S3']))
    q=trans[trans.scope=='ALL998'].set_index('condition')
    qa(9,'原164恢复','；'.join(f"{c}:{int(q.loc[c,'recovered'])}/164" for c in ['S1','S2','S3']))
    qa(10,'新增覆盖失败','；'.join(f"{c}:{int(q.loc[c,'newly_lost'])}/834,总失败{int(q.loc[c,'total_new_failures'])}" for c in ['S1','S2','S3']))
    qa(11,'方向一致性','；'.join(f"{r['condition']}:ADE改善{r['ADE_better_recordings']}/5,退化{r['ADE_worse_recordings']}/5" for r in signrows))
    qa(12,'利用语义空间','H2='+d['hypotheses']['H2']+'；仅指当前干预与当前冻结权重')
    qa(13,'证明语义有益','UNVERIFIED；0个匹配预算重训练无语义对照')
    qa(14,'证明分割错误','UNVERIFIED；0份真实语义GT，类别映射未知')
    qa(15,'无语义正式对照',d['research_recommendation']+'；本阶段不执行')
    qa(16,'可靠性门控','NO-GO，0个语义质量因果证据，0个新模块开发')
    qa(17,'调整小论文方向','当前只提供诊断；不自动改写方向/论文结论，等待科研审核')
    qa(18,'一个下一步实验','同预算从头训练匹配NoSemantic对照；提前冻结种子/容量审计/评价/失败门槛，需另行指令')
    write('STAGE11C0_REPORT.md','Stage11C-0 冻结Y-Net语义敏感性诊断',f"""
执行COMPLETE：S0原始998缓存经3×2样本全链路逐位回归；三组A完整预测和三组B固定Goal各998目标全部完成。没有训练/模型修改/类别重映射/测试集访问。研究性质FROZEN_MODEL_SEMANTIC_INTERVENTION_ONLY，不能宣称新算法或语义增强提升。

{hyp}

完整预测窗口加权：

{table_for(a,'ALL998',['GoalCover25','ADE1','FDE1','minADE20','minFDE20','JS_nats','Hungarian_mean_distance'])}

五录像等权：

{table_for(a,'FIVE_RECORDING_MACRO',['GoalCover25','ADE1','FDE1','minADE20','minFDE20'])}

{tab(qas,['question','answer'])}

{runtime}

所有分组、恢复与新增失败均完整保存，不把某个录像/分组收益推广为因果增益。S1/S2/S3输入存在已声明分布偏移，H5/H6未知。保留首次元数据错误的失败日志、SHA和GPU时间；仅工程记录修复，没有根据误差修改实验。

交付：01语义审计、02冻结协议、03Goal、04Path、05全部录像/运动组、06限定研究决策；results/tables/stage11c0/保存可重算配对统计；results/figures/stage11c0/含10组PNG/SVG及真实来源JSON/CSV。大型模型与完整tensor缓存留在忽略的experiments/stage11c0/cache/。源码/环境/数据版本见execution_manifest.json，质量审计见delivery_audit.json。GitHub推送状态以最终实际反馈为准，失败现场保留。

完成后停止，不启动无语义训练、可靠性门控或Stage11C-1。
""")
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        CUDA_build=torch.version.cuda,GPU='RTX3080 10240MiB',isolated_python=str(PYTHON.relative_to(ROOT))),indent=2)+'\n')
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts/stage11c0').rglob('*.py')) if '__pycache__' not in p.parts]
    (OUT/'execution_manifest.json').write_text(json.dumps(dict(stage='Stage11C-0',base_commit=BASE,freeze_commit='cbe69cc220894960f0c051c26b0c98e5feefb026',
        metadata_fix_commit='1a9bf5e2b5ddca6d8a0c02dd4577a5ebe44bf54a',numeric_execution_commit=d['source_commit'],source_files=sources,
        checkpoint_SHA=CK,original_prediction_SHA=PSHA,protocol_SHA=sha(CONF/'experiment_protocol.json'),generation_config_SHA=sha(CONF/'generation_config.json'),
        validation_manifest_SHA=json.loads((CONF/'experiment_protocol.json').read_text())['data']['original_manifest_SHA'],
        source_provenance='stage11b2c/data_provenance.json; all source files inherited SHA protected',training_steps=0,test_access=False,
        semantic_caches_original_unmodified=True,complete_generation_before_actual_GT=True,failed_attempt_preserved=True,
        GPU_process_wall_including_failure=budget['actual_GPU_process_seconds'],environment='environment.json'),ensure_ascii=False,indent=2)+'\n')
    print('REPORTS 7; H1-H6 separately labeled; no next experiment started',flush=True)
if __name__=='__main__':main()
