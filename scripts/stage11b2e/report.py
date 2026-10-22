"""Evidence-only report assembly after complete generation and CPU scoring."""
from .common import *
import platform
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch

def table(rows,columns):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'NA' if pd.isna(v) else f'{v:.6f}'
        return str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(row[c]) for c in columns)+' |' for row in rows])+'\n'

def main():
    cpu_only();d=json.loads((OUT/'final_decision.json').read_text());assert d['status'] in ['CONTROL_GO','CONTROL_NO_GO']
    c=json.loads((OUT/'computational_cost.json').read_text());sel=c['CPU_selection'];g=c['GPU_decode'];reg=c['GPU_regression']
    f=pd.read_csv(OUT/'scene_comparison.csv').set_index('scope');r=pd.read_csv(OUT/'failure_recovery.csv')
    reps=pd.read_csv(OUT/'representative_selection.csv');p=pd.read_csv(OUT/'paired_path_metrics.csv')
    goal=pd.read_csv(OUT/'paired_goal_metrics.csv');a=f.loc['ALL998'];m=f.loc['FIVE_RECORDING_MACRO'];dc=f.loc['deathCircle_3'];other=f.loc['OTHER4']
    names=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'];R=ROOT/'reports/stage11b2e';R.mkdir(parents=True,exist_ok=True)
    cohorts=[]
    for name,q in r.groupby('original_category25'):
        cohorts.append(dict(original_category25=name,targets=len(q),original_covered25=int(q.OriginalGoalCover25.sum()),
            alternative_covered25=int(q.AlternativeGoalCover25.sum()),recovered=int((q.paired_outcome25=='recovered').sum()),
            newly_lost=int((q.paired_outcome25=='lost').sum()),original_minADE20=q.OriginalMinADE20.mean(),
            alternative_minADE20=q.AlternativeMinADE20.mean(),original_minFDE20=q.OriginalMinFDE20.mean(),
            alternative_minFDE20=q.AlternativeMinFDE20.mean(),median_local_draws25=q.LocalProposalCount25.median()))
    pd.DataFrame(cohorts).to_csv(OUT/'failure_cohort_summary.csv',index=False)
    identities=r.sample_id.str.rsplit(':frame',n=1).str[0]
    counts=dict(target_windows=len(r),unique_recording_person_IDs=identities.nunique(),overlapping_windows_not_independent=True)
    json_write=lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    json_write(OUT/'sample_dependence.json',counts)
    frozen=json.loads((OUT/'protocol_frozen.json').read_text());budget=json.loads((OUT/'gpu_budget.json').read_text())
    source=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in sorted((ROOT/'scripts/stage11b2e').rglob('*.py')) if '__pycache__' not in x.parts]
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,
        torch_CUDA_build=torch.version.cuda,CPU_analysis_CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        GPU='NVIDIA GeForce RTX 3080, 10240MiB; actual monitor samples in GPU_card_*.csv',
        isolated_python=str(PYTHON.relative_to(ROOT)),test_plugins='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1')
    json_write(OUT/'environment.json',env)
    json_write(OUT/'execution_manifest.json',dict(stage='Stage11B-2E',scope='CONTROL_EXPERIMENT_ONLY',base_commit=BASE,
        freeze_commit='d221b22279731d94655c1cba5134efc9876f2b0b',
        numeric_execution_commit=d['source_commit'],delivery_provenance='Final delivery commit in Git; source SHAs authoritative for report/plot postprocessing',
        official_YNet_commit=OFFICIAL,checkpoint_SHA=CK,original_prediction_SHA=PSHA,
        source_files=source,protocol_SHA=frozen['protocol_SHA'],generation_config_SHA=frozen['generation_config_SHA'],
        source_data_records=json.loads((CONF/'experiment_protocol.json').read_text())['sources'],
        new_goal_file_SHA=sel['new_goals_file_SHA'],new_path_file_SHA=g['alternative_paths_file_SHA'],
        score_identity='same998 history/GT_SHA in paired_goal_metrics.csv and paired_path_metrics.csv',
        original_tensor_files='all998 SHA in protected_files.csv.gz and stage11b2c/tensor_manifest.csv',
        GT_first_read='score.py only after all998 new paths saved and all head0 verified',
        training_steps=0,new_sampling=0,new_kmeans=0,test_data_access=False,environment='environment.json'))
    testfile=OUT/'full_tests.xml';testtext='全仓测试尚未完成，不能声明PASS。'
    if testfile.exists():
        suites=list(ET.parse(testfile).iter('testsuite'));tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        testtext=f"全仓pytest：{tests['tests']-tests['skipped']-tests['failures']-tests['errors']} PASS、{tests['skipped']} SKIP、{tests['failures']} FAIL、{tests['errors']} ERROR；原始记录full_tests.txt/XML。SKIP不算PASS。"
    after=OUT/'history_after.json';historytext='保护后SHA核验尚未完成。'
    if after.exists():
        h=json.loads(after.read_text());assert h['passed'];historytext=f"保护后{h['files']}项历史文件SHA全部未变（history_after.json）。"
    common=f"""
数据与结论范围：同一998个SDD开发验证目标窗口，5个既有录像，8帧历史/12帧未来、0.4秒，单位为SDD原参考像素px，resize=0.25。不是ETH/UCY，不是独立确认，不与米制结果排名。{counts['unique_recording_person_IDs']}个录像内行人ID；重叠窗口不能视为独立样本，本报告只给配对描述统计，不给窗口独立性p值。

原始模型SHA `{CK}`；原预测SHA `{PSHA}`；作者源码MIT版本 `{OFFICIAL}`。所有计算沿用已冻结协议；无训练、无新采样、无KMeans重跑、无测试集访问。GT只用于全部新预测保存后的离线评分和绘图。

原始证据目录：[results/tables/stage11b2e](../../results/tables/stage11b2e/)。{testtext} {historytext}
"""
    def write(name,title,text):
        (R/name).write_text('# '+title+'\n\n'+text.strip()+'\n\n'+common.strip()+'\n')
    scene_rows=[]
    for scope in ['ALL998']+names+['OTHER4','FIVE_RECORDING_MACRO']:
        z=f.loc[scope];scene_rows.append(dict(scope=scope,n=int(z.targets),oldCover25=z.OriginalGoalCover25,newCover25=z.AlternativeGoalCover25,
            oldADE=z.OriginalminADE20,newADE=z.AlternativeminADE20,oldFDE=z.OriginalminFDE20,newFDE=z.AlternativeminFDE20))
    scene_table=table(scene_rows,['scope','n','oldCover25','newCover25','oldADE','newADE','oldFDE','newFDE'])
    metrics=[]
    for key in ['ADE1','FDE1','GoalFDE1','minGoalFDE20','minADE20','minFDE20','GoalCover10','GoalCover25','GoalCover50',
        'GoalPairwiseDispersion','PathPairwiseDispersion','UniqueGoals','UniquePaths','Distortion19','Distortion20','FixedCellDistortion19',
        'ProbabilityUnion10','ProbabilityUnion25','ProbabilityUnion50','GoalPathEndpointMeanDistance']:
        metrics.append(dict(metric=key,original=a['Original'+key],alternative=a['Alternative'+key],delta=a['Delta'+key]))
    metrics_table=table(metrics,['metric','original','alternative','delta'])
    gate_rows=[];me=d['measurements']
    labels=[('五录像等权GoalCover25增加至少1pp',me['macro_GoalCover25_gain']*100,'macro_GoalCover25_gain_ge1pp'),
        ('五录像等权minADE20相对降低至少1%',me['macro_minADE20_relative_drop']*100,'macro_minADE20_drop_ge1pct'),
        ('五录像等权minFDE20相对退化不超过1%',me['macro_minFDE20_relative_degradation']*100,'macro_minFDE20_degradation_le1pct'),
        ('其他四录像合计GoalCover25严格增加',me['other4_GoalCover25_gain']*100,'other4_GoalCover25_strict_gain')]
    for label,val,key in labels:gate_rows.append(dict(condition=label,actual_pp_or_percent=val,result='PASS' if d['conditions'][key] else 'FAIL'))
    gate_table=table(gate_rows,['condition','actual_pp_or_percent','result'])
    audit=f"""
结论：缓存准入PASS；实验只检验一次普通代表点替换。基于Stage2D `{BASE}` 建立当前分支。冻结提交`d221b22279731d94655c1cba5134efc9876f2b0b`在998目标选择前成功；CPU选择执行SHA与GPU/评分执行SHA分别记录在selection_audit.json与generation_execution.json。工程准入提交`{d['source_commit']}`在GPU前成功。数值脚本、评价公式和四项门槛的文件SHA见protocol_frozen.json。

cache_audit.json确认998个sample_id唯一、原10000个proposal的下标/坐标/顺序SHA一致，原19中心、Softargmax、原20Goal与原20Path逐位一致，全部18962个分区非空，人口和每目标10000。没有用重新生成的随机样本替代缓存。保护清单configs/stage11b2e/protected_files.csv.gz覆盖92362项历史文件。

CPU选择接口只接收白名单sample_id、proposals、centers、Softargmax；实际GT尚未装载。原记录中未来GT和oracle/A25标签被替换的3个反事实样本，白名单输入和结果均逐位一致；全998倒序重算逐位一致。GPU配置为GT-free白名单历史、冻结语义、模型、已选Goal和3个旧预测回归参考；worker不载入GT/失败标签/已有误差。先完整保存(20,998,12,2)新Path，再由独立CPU score.py载入GT。score_integrity.json与逐目标history_SHA/GT_SHA核验通过。

当前使用既有纯历史缓存及五录像冻结语义，不新执行分割网络；原分割网络仍随模型严格加载并处于eval。大型tensor、模型、原图片只在忽略缓存，不提交。2D中的最终Voronoi不等同原KMeans最后迭代标签；本阶段不改变此定义。
"""
    write('01_PROTOCOL_AND_CACHE_AUDIT.md','冻结协议与缓存身份审计',audit)
    design=f"""
普通平方medoid，无原创算法主张。固定原10000次采样P与原19中心C，float64 CPU分配 a_j=argmin_k||P_j-C_k||²，精确等距取较小k。分区Z_k含重复采样的全部次数。

平方距离恒等式：sum_q||q-p||² = sum_q||q-mean(Z_k)||² + |Z_k| ||p-mean(Z_k)||²。因此选择真实原采样点中距离经验均值最近者；精确相同float64目标值取原全局proposal下标最小者。复制原float32坐标，不额外四舍五入。复杂度O(10000×19)，分块512计算，不构造大簇N×N矩阵；小规模暴力平方medoid人工测试通过。

Head0原Softargmax逐位保持，Head1..19严格对应原分区；空分区回退原中心并标注index=-1，实际空分区{int(reps.empty_fallback.sum())}，人口范围{int(reps.population.min())}—{int(reps.population.max())}。保留重复点，不补候选。实际新旧每个目标均有20个唯一Goal，无重复Goal；最终唯一Path平均均为{a.OriginalUniquePaths:.6f}（按整条12帧路径精确坐标去重）。

18962个非head0中{d['goal_heads_changed']}个发生变化；998个目标中K20 Goal整组完全不变为{d['unchanged20goal_targets']}。包含head0的平均位移{d['mean_goal_displacement_all20']:.9f}px；19中心平均位移{d['mean_goal_displacement_all20']*20/19:.9f}px。每head索引/人口/均值/坐标/固定分区SSD见representative_selection.csv；每目标proposal/center/index/新Goal SHA见goal_change_summary.csv，新Goal缓存文件SHA `{sel['new_goals_file_SHA']}`。

新旧Goal不等于全部解码图发生变化。作者trajectory_maps调用get_patch，在乘0.25后按np.round生成Goal图；此量化是旧代码原有计算，不是新算法额外取整。实际只{d['decoder_map_centers_changed']}个head改变图中心，对应{d['path_heads_changed']}条Path发生逐位变化；{d['unchanged20path_targets']}个目标的完整K20 Path逐位未变。全部20条均真实重算，没有复用旧Path伪装新推理。
"""
    write('02_REPRESENTATIVE_SELECTION.md','固定Voronoi平方medoid选择',design)
    decoder=f"""
解码工程准入PASS。固定3个sample_id（配置中已冻结，按原清单位置0/499/997），原20Goal走新入口，正序及倒序重复共6次；每次完整20×12×2路径与原缓存逐位一致，最大坐标差0。原始行级证据decoder_regression_rows.csv；模型全部参数stateSHA `{g['model_state_SHA']}`保持不变。

公式保持：原历史×0.25 → 原trajectory_maps与语义 → pred_features；原像素Goal×0.25 → 原trajectory_maps → 原多尺度avg_pool pyramid → concat共享特征 → pred_traj → 原softargmax → 除0.25恢复原像素。单元测试还核对作者内层循环AST。没有调用Goal Predictor、TTST或KMeans，没有改变网络/损失/参数。eval与no_grad固定，语义缓存SHA验证。

全998新路径shape=(20,998,12,2)，均有限，新文件SHA `{g['alternative_paths_file_SHA']}`，Path张量SHA `{g['path_SHA']}`。全部998个Head0 Goal和Path均逐位不变，ADE1/FDE1逐目标完全相同；不存在将Head0误差变化当收益的情形。新实际解码内容与原缓存相同也保留记录。

GPU回归进程完整wall {budget['processes'][1]['wall_seconds']:.6f}s，998解码进程完整wall {budget['processes'][0]['wall_seconds']:.6f}s；两进程返回码均0。完整进程wall含启动/加载/同步/保存/正常退出/监控线程退出，退出开销没有独立拆分，不用估值冒充实测。
"""
    write('03_FROZEN_DECODER_REGRESSION.md','冻结解码器逐位回归与全量推理',decoder)
    write('04_GOAL_AND_PATH_METRICS.md','Goal与完整Path配对指标',f"""
结论：窗口加权路径指标仅微降，五录像等权ADE/FDE均微升，25px覆盖减少。所有数值由同目标原/新缓存与同GT重新计算。minADE20、minFDE20各自取20候选最小，可来自不同head，属于GT oracle评价，不是在线选择能力。

{scene_table}

窗口加权详细指标（概率/覆盖用0—1；平方失真单位px²）：

{metrics_table}

Goal/PathPairwiseDispersion分别为190对Goal距离/190对整路径逐时平均距离。Distortion19/20为10000次原采样到相应候选集合的最近平方距离均值；评估时求最近距离不是重新聚类。FixedCellDistortion19保持原分区标签求SSD/10000。均保留原采样重复计数。原中心并非本次最终Voronoi经验均值的逐位重算，所以不假定所有单簇SSD只能增加；全局实际值如表。

ProbabilityUnionR使用原normalized_weights对应离散像素概率，在原坐标恢复后计算落入任一R半径球的并集质量，每网格只计一次；无GT参与此概率覆盖计算，也没改变阈值/采样分布。10px概率质量增加不等同25px真实终点覆盖改善，不能更换已冻结主门槛。

逐目标paired_goal_metrics.csv、paired_path_metrics.csv及19960行paired_head_metrics.csv给出全head身份、原/新GoalFDE、Path ADE/FDE、SHA与是否逐位变化；motion_group_comparison.csv保留Stage2A全部历史分组及所有录像，零样本组写n=0和空指标，不能填造0误差。
""")
    q=r[r.original_category25=='proposal_hit_final_miss'];z=r[r.original_category25=='zero_support_final_miss']
    cohorts_table=table(cohorts,list(cohorts[0]))
    write('05_FAILURE_RECOVERY_ANALYSIS.md','固定86失败恢复与原成功退化',f"""
原86身份保持不变：恢复{d['original86_recovered']}，仍失败{d['original86_still_fail']}；原834成功目标保留829，新增失覆盖{d['lost_originally_covered']}。总失覆盖164→167，覆盖净变化2−5=−3。不是只统计2个恢复忽略5个新失败。

{cohorts_table}

86组minADE20 {d['original86_minADE20_original']:.9f}→{d['original86_minADE20_alternative']:.9f}px；minFDE20 {d['original86_minFDE20_original']:.9f}→{d['original86_minFDE20_alternative']:.9f}px。局部原proposal25px次数与ADE变化的Spearman描述相关为{d['within_original86_local_count_ADEdelta_Spearman_descriptive']:.6f}，包含相同窗口行人的相关性，不能作因果解释或独立样本显著性。

77个原阈值后零支持目标覆盖仍为{d['zero77_final_covered']}/77。没有新采样或改阈值，故不能宣称恢复原离散分布不存在的概率支持。1个原正支持有限抽样未命中目标也继续失败。本次两例恢复来自原86，未重定义失败组。

deathCircle覆盖310/443→310/443，组内恢复和损失抵消；其他四录像524/555→521/555，下降{(other.AlternativeGoalCover25-other.OriginalGoalCover25)*100:.6f}pp。deathCircle占原86中的65/86，但不能由该录像微小路径改善推广到五录像。bookstore覆盖399/423→396/423，其余三录像覆盖数量保持，路径指标仍可能改变。

固定案例规则：四种配对结果分别按sample_id排序取floor((n−1)/2)，而非挑选最佳误差；case_manifest.csv与case_plot_data.json保留四个真实案例的完整20Goal/20Path/GT/SHA。图6同时展示恢复、退化、持续失败、持续覆盖。六图各PNG/SVG与源数据定位见figure_manifest.json。GT仅离线评分/绘图。
""")
    runtime=f"""
CPU首次998选择（含读取/核验）{sel['algorithm_seconds']:.6f}s，折合{sel['algorithm_seconds']/998*1000:.6f}ms/目标；其中分配{sel['assignment_seconds']:.6f}s、代表点计算{sel['representative_seconds']:.6f}s。倒序和反事实复核{sel['reverse_and_counterfactual_seconds']:.6f}s，新Goal保存{sel['save_seconds']:.6f}s，选择入口总计{sel['total_process_seconds']:.6f}s；新Goal缓存{sel['cache_bytes']}字节。独立CPU评分{c['CPU_score_seconds']:.6f}s，历史全量SHA与测试/作图另计，不冒充代表算法时间。

GPU全进程合计{c['GPU_process_seconds']:.6f}s / 3600s硬上限（3300s安全阈值），无中断；GPU正式解码模型加载{g['model_load_seconds']:.6f}s，冻结语义读取{g['semantic_cache_read_seconds']:.6f}s、H2D {g['semantic_H2D_seconds']:.6f}s，新分割前向0s，历史特征{g['feature_seconds']:.6f}s，998×20路径解码{g['decode20_seconds']:.6f}s，目标循环{g['all_target_loop_seconds']:.6f}s，最后缓存保存{g['cache_save_seconds']:.6f}s。allocated峰值{g['peak_allocated_bytes']/2**20:.3f}MiB，reserved {g['peak_reserved_bytes']/2**20:.3f}MiB；10秒采样整卡峰值{int(pd.read_csv(OUT/'GPU_card_decode.csv').memory_used_MiB.max())}MiB，采样峰值不等同连续整卡峰值。训练时间0s。
"""
    write('06_FINAL_RESEARCH_DECISION.md','普通代表点控制最终研究判断',f"""
最终：**{d['status']}**。四个冻结条件仅{sum(d['conditions'].values())}/4通过，未改成功标准。

{gate_table}

窗口加权minADE20降低{(a.OriginalminADE20-a.AlternativeminADE20)/a.OriginalminADE20*100:.6f}%，minFDE20降低{(a.OriginalminFDE20-a.AlternativeminFDE20)/a.OriginalminFDE20*100:.6f}%，不足以抵消五录像等权反向及覆盖退化；不能据此宣布H1成立。平方medoid微增几何分散度，却增加经验平方失真，未提供跨录像稳定工程收益。该结论否定本次冻结普通方案，不证明所有可能的候选选择方法都无效。

本方法是常规平方medoid和最近经验均值的恒等式实现，无可学习新机制；不具备将其直接作为SCI原创算法的证据。此前文献全文/版本审查见[Stage2D文献报告](../stage11b2d/04_LITERATURE_AND_NOVELTY_REVIEW.md)，DenseTNT、HYPER、DPP、NPSN、TUTR等已覆盖有限候选、质量与多样性取舍的相关问题。沿用2D结论，不声称本阶段重新完成系统检索或证明所有特定机制完全相同。

建议终止当前平方medoid代表点路线，不自动尝试FPS、其他阈值/seed/候选数/权重、不进入Stage11C、不训练。是否调整论文方向交由下一次科研决策。

{runtime}
""")
    qa=[
        ('Q1 Goal平均变化',f"全部20head平均{d['mean_goal_displacement_all20']:.9f}px；19中心{d['mean_goal_displacement_all20']*20/19:.9f}px"),
        ('Q2 变化head',f"{d['goal_heads_changed']}/19960；非head0为{d['goal_heads_changed']}/18962"),
        ('Q3 K20完全不变',f"Goal整组0/998；Path整组{d['unchanged20path_targets']}/998逐位未变"),
        ('Q4 Cover25',f"834/998({a.OriginalGoalCover25*100:.6f}%)→831/998({a.AlternativeGoalCover25*100:.6f}%)；未提高"),
        ('Q5 原86恢复',f"{d['original86_recovered']}/86恢复、{d['original86_still_fail']}/86仍失败"),
        ('Q6 新失败',f"{d['lost_originally_covered']}/834；总A25失败164→167"),
        ('Q7 minADE20',f"窗口{a.OriginalminADE20:.9f}→{a.AlternativeminADE20:.9f}px；宏平均{m.OriginalminADE20:.9f}→{m.AlternativeminADE20:.9f}px"),
        ('Q8 minFDE20',f"窗口{a.OriginalminFDE20:.9f}→{a.AlternativeminFDE20:.9f}px；宏平均{m.OriginalminFDE20:.9f}→{m.AlternativeminFDE20:.9f}px"),
        ('Q9 ADE1/FDE1',f"{a.OriginalADE1:.9f}/{a.OriginalFDE1:.9f}px完全相同；998/998 head0逐位不变"),
        ('Q10 五录像一致性','0/5录像Cover25增加；ADE在2/5微降、2/5微升、1/5相同，没有一致收益'),
        ('Q11 deathCircle',f"443/998目标，原86中65/86；Cover25 310→310，ADE {dc.OriginalminADE20:.6f}→{dc.AlternativeminADE20:.6f}，其他四录像Cover25 524→521"),
        ('Q12 分散度',f"Goal {a.OriginalGoalPairwiseDispersion:.6f}→{a.AlternativeGoalPairwiseDispersion:.6f}px；Path {a.OriginalPathPairwiseDispersion:.6f}→{a.AlternativePathPairwiseDispersion:.6f}px，微增不代表覆盖获益"),
        ('Q13 经验失真',f"19最近平方失真{a.OriginalDistortion19:.6f}→{a.AlternativeDistortion19:.6f}px²；固定原分区{a.OriginalFixedCellDistortion19:.6f}→{a.AlternativeFixedCellDistortion19:.6f}px²；20见04报告"),
        ('Q14 成本',f"CPU选择入口{sel['total_process_seconds']:.6f}s；CPU评分{c['CPU_score_seconds']:.6f}s；GPU全进程{c['GPU_process_seconds']:.6f}s，训练0s"),
        ('Q15 四条件',f"{sum(d['conditions'].values())}/4通过；只有FDE退化约束通过"),
        ('Q16 是否支持继续','本次普通方案NO-GO；不是对所有未知候选策略的普遍否定'),
        ('Q17 SCI新颖性','0个原创机制，普通平方medoid不能直接作为SCI创新点；2D相关先例风险HIGH'),
        ('Q18 下一步','终止本次路线，0个后续实验自动启动，等待用户科研决策')]
    qa_table=table([dict(question=x,answer=y) for x,y in qa],['question','answer'])
    write('STAGE11B2E_REPORT.md','Stage11B-2E 冻结分区代表点控制实验',f"""
**{d['status']} — CONTROL_EXPERIMENT_ONLY。** 全998目标CPU选择、真实冻结GPU解码与离线同目标GT评分完成。原834覆盖降至831；恢复2个原86失败但新损失5个。窗口加权路径指标略降，五录像宏平均路径指标略升。四项门槛仅1项通过；H1未得到工程门槛支持。无训练/采样/重新聚类/测试集评价。

{scene_table}

{gate_table}

用户要求的18项判断：

{qa_table}

{runtime}

工程通过不等于性能通过：18项GPU前工程测试通过、3样本原20路径×2次逐位回归、998个Head0逐位不变。18961个Goal发生变化，但原Goal热图的0.25倍缩放与np.round使仅761个head实际改变Path，518组Path完全不变；所有新Path仍逐条真实重算。无缓存冒用。六组真实PNG/SVG和案例规则保留，未只画恢复案例。

交付索引：01协议缓存、02选择公式/合法索引、03冻结解码回归、04全部指标与分组、05恢复与退化、06最终停止判断。详细表格为results/tables/stage11b2e/，图为results/figures/stage11b2e/，大缓存experiments/stage11b2e/cache/不入Git。协议与数值源码冻结SHA见protocol_frozen.json，完整执行环境/版本见execution_manifest.json和environment.json。GitHub推送结果以最终反馈的真实执行结果为准；网络或认证失败不改变实验判断，失败现场保留在本阶段忽略缓存。

不启动第二方法、不进入Stage11C，完成交付后停止。
""")
    print('REPORTS 7; decision',d['status'],flush=True)

if __name__=='__main__':main()
