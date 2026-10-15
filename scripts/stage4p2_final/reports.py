"""Evidence-only final acceptance reports. Nothing here trains or changes a model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,json
import numpy as np
from src.training.social_diagnostics import csv_new
ROOT=Path('results/tables/stage4p2_final');REPORT=Path('reports/stage4p2_final')


def read(name):return list(csv.DictReader((ROOT/name).open()))


def number(value):
    if value is None or value=='':return '空组/不适用'
    if isinstance(value,(bool,int)):return str(value)
    try:return f'{float(value):.6f}' if any(s in str(value) for s in ('.','e-','e+')) else str(value)
    except ValueError:return str(value)


def table(rows,columns):
    return '|'+ '|'.join(c[0] for c in columns)+'|\n|'+ '|'.join('---' for c in columns)+'|\n'+''.join('|'+ '|'.join(number(r.get(k)) for _,k in columns)+'|\n' for r in rows)


def main():
    protocol=json.loads(Path('configs/stage4p2_final/protocol.json').read_text());metrics=read('model_comparison.csv');pair=read('paired_comparisons.csv');group=read('all_group_results.csv')
    baselines=read('fresh_cv_metrics.csv');neural=[];resource=[];aux=[];convergence=[];gates=[]
    for fold in protocol['folds']:
        for kind in ('G0','G1'):
            summary=json.loads((ROOT/'extension'/fold/kind/'summary.json').read_text());neural.append(summary)
            resource.append(dict(fold=fold,model=kind,train_seconds=summary['additional_train_seconds'],eval_seconds=summary['additional_evaluation_seconds'],wall_seconds=summary['phase_wall_seconds']))
            rows=read(Path('extension')/fold/kind/'curve_1to100.csv');val=np.array([float(r['K1_distribution_mean_ADE']) for r in rows]);nll=np.array([float(r['NLL']) for r in rows]);train=np.array([float(r['train_NLL_window_mean']) for r in rows])
            convergence.append(dict(fold=fold,model=kind,best_ADE_epoch=summary['best']['epoch'],min_NLL_epoch=int(np.argmin(nll))+1,
                last_ADE=val[-1],last_NLL=nll[-1],train_NLL_90to94=float(train[89:94].mean()),train_NLL_96to100=float(train[95:100].mean()),
                val_ADE_90to94=float(val[89:94].mean()),val_ADE_96to100=float(val[95:100].mean()),
                NLL_down_ADE_up_steps=int(((np.diff(nll)<0)&(np.diff(val)>0)).sum()),max_preclip_gradient=max(float(r['gradient_norm_max']) for r in rows),
                peak_MiB=max(float(r['peak_allocated_MiB']) for r in rows),all_finite=bool(np.isfinite(val).all() and np.isfinite(nll).all() and np.isfinite(train).all())))
            for role,key in [('budget30_best','original_budget30_best'),('budget100_best','best'),('budget100_last','last')]:
                r=summary[key];aux.append(dict(role=role,fold=fold,model=kind,epoch=r['epoch'],**{k:r[k] for k in ('NLL','K1_distribution_mean_ADE','K1_distribution_mean_FDE','K1_first_sample_ADE','K1_first_sample_FDE','K20_independent_min_ADE','K20_independent_min_FDE')},checkpoint_sha256=r['checkpoint_sha256']))
        r0=next(r for r in metrics if r['role']=='budget100_best' and r['fold']==fold and r['model']=='G0');r1=next(r for r in metrics if r['role']=='budget100_best' and r['fold']==fold and r['model']=='G1')
        delta=float(r1['ADE_m'])-float(r0['ADE_m']);fdelta=float(r1['FDE_m'])-float(r0['FDE_m'])
        gates.append(dict(fold=fold,G1_minus_G0_ADE=delta,G1_minus_G0_FDE=fdelta,worthwhile=delta<=-max(.005,.01*float(r0['ADE_m'])),FDE_safe=fdelta<=max(.01,.02*float(r0['FDE_m']))))
    csv_new(ROOT/'auxiliary_neural_metrics.csv',aux);csv_new(ROOT/'convergence_100_audit.csv',convergence);csv_new(ROOT/'actual_extension_times.csv',resource)
    overall=table(metrics,[('预算/选择','role'),('录像','fold'),('模型','model'),('目标窗口','target_windows'),('不同ID','unique_people'),('ADE(m)','ADE_m'),('FDE(m)','FDE_m')])
    cvtable=table(baselines,[('录像','fold'),('解析模型','model'),('ADE','ADE_m'),('FDE','FDE_m'),('纯解析计算秒','inference_seconds'),('评价秒','evaluation_seconds')])
    pairedtable=table([r for r in pair if r['role']=='budget100_best' and r['model'] in ('G0','G1')],[('录像','fold'),('模型','model'),('对照','baseline'),('ΔADE','delta_ADE_m'),('ΔFDE','delta_FDE_m'),('逐目标ADE改善比例','fraction_ADE_improved')])
    (REPORT/'02_PAIRED_CV_COMPARISON.md').write_text(f'''# 完全配对CV与G0/G1验收

直接读取Stage4.2冻结缓存的ZARA2 921窗口/5833目标窗口、HOTEL301/1053，全体6886目标窗口均保留。每一行共同window_id、agent_id、原始float32真实未来，另存逐目标future/history SHA256；同一12步ADE/FDE定义。没有使用Stage3.8任何旧CV指标。

CV公式：Last2速度=(X7−X6)/0.4；OLS4/OLS8对最近4/8帧真实0.4秒时间坐标求线性斜率，均以X7为起点预测X7+v*k*0.4。完整8帧任务没有缺失速度回退或未来补齐。解析模型0参数，无拟合超参数。

{cvtable}

以上时间为CPU单线程向量化纯预测/距离评价实测，未含数据加载、写表，不能当端到端速度。神经预测位置保持原float32输出，与同一GT提升float64计算距离；聚合误差与冻结旧评价器核对差<=1e-6，不覆盖原指标。

## 全预算与末轮

{overall}

30与100分别用各自预算范围最低验证均值ADE、同一checkpoint FDE，完全同值更早轮；last100独立展示。100使用同开发集增加了选模次数，不是新的独立确认。CV在所有预算列完全相同，不能按录像事后混选CV作为一个新方法。

## 同目标配对差值

{pairedtable}

Δ为模型−对照，负值改善。完整目标配对明细`budget*_paired_targets.csv.gz`，不只保留成功目标。`all_group_results.csv`包含所有运动/人数/合法邻居组及空组；`per_person_errors.csv`按录像内行人ID聚合，`aggregate_comparison.csv`分开录像宏平均与目标窗口加权，重叠窗口不当作独立样本，不做iid p值。

完整20帧上下文人数包括目标且由离线完整筛选决定，不能混称历史可见邻居数；历史运动/密度分组沿用Stage4.2，并增加预定义静止细分。分组不读取未来误差；未来仅作为统一评价GT。ETH未评价。
''')
    budget30=read('budget30_diagnosis.csv')
    (REPORT/'03_TRAINING_BUDGET_AUDIT.md').write_text(f'''# 训练预算与收敛审查

唯一延长方案在930f92d提交后执行；四组同时满足预注册未收敛判据，不是看到G1差而单独延长。旧30轮结果/权重/曲线完整保留。由其各自epoch30模型、SGD状态与CPU/CUDA RNG恢复，31–100不重启；恢复时完整验证指标与旧30轮匹配。

{table(budget30,[('录像','fold'),('模型','model'),('训练NLL下降','train_NLL_drop'),('验证ADE下降','val_ADE_drop'),('旧最优轮','best_epoch'),('全有限','all_finite'),('延长判据通过','clearly_unfinished')])}

下降比较20–24与26–30均值，所有组训练NLL下降>=.01、ADE下降>=max(.005m,1%)且最优轮>=28。冻结预测追加训练+评价{protocol['extension']['estimated_train_eval_seconds']:.2f}秒；额外30%写盘等余量后约{protocol['extension']['estimated_minutes_with30percent_IO_margin']:.2f}分钟。没有使用ETH作预算依据。

## 实际追加成本

{table(resource,[('录像','fold'),('模型','model'),('追加纯训练秒','train_seconds'),('追加评价秒','eval_seconds'),('31–100含保存墙钟秒','wall_seconds')])}

保持SGD0.01、无momentum/decay/scheduler、clip1、batch16、64窗口累积、seed17+epoch窗口顺序，所有1553参数同旧模型。未改PTN epsilon/预测损失/数据/图，四组各追加70轮；只此一次，100后停止。

## 100轮后的边界与不稳定

{table(convergence,[('录像','fold'),('模型','model'),('minADE轮','best_ADE_epoch'),('minNLL轮','min_NLL_epoch'),('lastADE','last_ADE'),('90–94训练NLL','train_NLL_90to94'),('96–100训练NLL','train_NLL_96to100'),('90–94 ADE','val_ADE_90to94'),('96–100 ADE','val_ADE_96to100'),('NLL下降/ADE上升次数','NLL_down_ADE_up_steps'),('裁剪前最大梯度','max_preclip_gradient'),('峰值MiB','peak_MiB')])}

无NaN/Inf与数值稳健性不是同一判断；有限但振荡的ADE、裁剪前梯度和PTN扰动放大单独记录。逐步高斯NLL不等于累积位置ADE。minNLL轮只描述，不改变选模。所有辅助NLL/首采样/K20见`auxiliary_neural_metrics.csv`。

延长预算能够改变数值和局部模型排序；不能把30轮失败直接当作充分收敛结论。反之，100轮也不保证最终收敛，尤其最优点接近边界或末轮振荡时。本轮验收止于唯一固定100轮预算，不再为了超过CV继续延长。
''')
    # PTN and research decisions are rendered below from complete stress results.
    render_decision(protocol,metrics,group,gates,convergence)


def render_decision(protocol,metrics,groups,gates,convergence):
    stress=read('ptn_group_summary.csv');stress_execution=json.loads((ROOT/'ptn_execution.json').read_text())
    primary=[r for r in stress if r['graph_mode']=='fixed_history_graph' and float(r['epsilon_m'])==.001]
    coarse=[r for r in primary if r['group'] in ('all','stationary','low_speed','normal_walking','fast')]
    ptn_table=table(coarse,[('预算','role'),('录像','fold'),('模型','model'),('历史组','group'),('目标窗口','target_windows'),('不同人','unique_people'),('终点位移放大P50','final_gain_P50'),('P95','final_gain_P95'),('P99','final_gain_P99'),('最大','final_gain_max'),('>.1m且>100倍比例','high_risk_window_fraction'),('人均风险比例','person_macro_risk_fraction'),('重复风险判据','repeated_risk_evidence')])
    (REPORT/'04_PTN_SENSITIVITY_AUDIT.md').write_text(f'''# PTN固定权重全目标敏感性审查

不是单个极端案例外推：两录像全部6886目标窗口，G0/G1各自30与100预算best，扰动前固定历史运动分组。完整原始记录{stress_execution['raw_rows']}行；重叠窗口及两个方向仍相关，不把这些行称作独立样本。

预注册历史均速<=.2m/s静止、(.2,.8]低速、(.8,1.6]正常行走、>1.6快速；另列严格静止（最大步速<=.05且净位移<=.05m）和精确静止（相邻坐标差<=1e-8m）。静止仅历史定义，不读取未来真值。所有空组保留样本数0、数值留空。

## 扰动与度量

扰动幅度1e-5/1e-4/1e-3m，每个历史点二维方向由窗口ID/行人ID/rep键控，两个固定方向实现，同方向按幅度缩放。全体历史行人同时扰动；固定图为主，重算4m图为辅。分母记录实际float32坐标扰动RMS，分子为预测**相对最后观测的未来位移**变化；另存绝对位置变化。报告每目标两个方向各指标的最大值，再求组P50/P95/P99/max。该增益是共同观测扰动下的目标响应，不是全网络Lipschitz常数。

PTN方差常数epsilon_N=1e-5完全未改；它与以米计的输入扰动epsilon_X不同。各PTN钩子记录实际输入/输出变化比以及基线通道时间方差低于epsilon_N的比例。整个网络的扰动结果不能单独证明PTN的唯一因果责任，但局部PTN增益提供可定位的机制证据。

同扰动CV-Last2作为自然运动外推标定，其12步相对位移变化理论约束<=24倍单点幅度（含float32量化余量）；不把普通外推放大直接称故障。1mm下>.1m且>100倍、组内>=30不同人且>=5%窗口达到，预定义为重复风险迹象；仍不意味着所有目标不稳定，也不作iid显著性。

## 1mm固定图：全部主运动组

{ptn_table}

`ptn_group_summary.csv`还包含两种图、三种幅度、精确/严格静止、全部局部PTN增益和CV标定；原始逐目标文件在`ptn/`。局部比值与模型级位移增益是不同量，不能混用。固定图结果隔离了阈值换边；重算图记录每目标二值边变化，G0在两图模式预测一致性另行核验。

任何单次极端max均只作为反例；风险覆盖率、不同人数量和人均窗口风险率比单个max更重要。没有调整epsilon、重训归一化变体或按未来误差挑病例。
''')
    strong=all(r['worthwhile'] and r['FDE_safe'] for r in gates)
    improved=[r['fold'] for r in gates if r['G1_minus_G0_ADE']<0]
    model_index={(r['role'],r['fold'],r['model']):r for r in metrics}
    worse_all=all(float(model_index['budget100_best',f,m]['ADE_m'])>float(model_index['budget100_best',f,c]['ADE_m']) and float(model_index['budget100_best',f,m]['FDE_m'])>float(model_index['budget100_best',f,c]['FDE_m']) for f in protocol['folds'] for m in ('G0','G1') for c in ('CV-Last2','CV-OLS4','CV-OLS8'))
    cv_conclusion='两个录像上，G0/G1整体ADE和FDE均未超过三种重新配对的CV。' if worse_all else '神经模型与各CV的优势取决于录像/指标，完整对照表必须逐项判断，未建立跨录像一致优势。'
    social_conclusion=f'G1低于G0的录像：{", ".join(improved) if improved else "无"}；两录像预注册收益/FDE门槛同时通过：{strong}。'
    stop=not strong or worse_all
    recommendation='停止当前HTraj逐人PTN图家族及继续堆叠社会模块的开发，不采用为第四章正式基础。' if stop else '仍只有seed17和曝光录像的候选证据，暂不采用为第四章正式基础；本阶段不自动追加实验。'
    risks=[r for r in primary if r['repeated_risk_evidence']=='True']
    gate_table=table(gates,[('录像','fold'),('G1−G0 ADE','G1_minus_G0_ADE'),('G1−G0 FDE','G1_minus_G0_FDE'),('ADE达标','worthwhile'),('FDE达标','FDE_safe')])
    decision_groups=[r for r in groups if r['role']=='budget100_best' and r['model'] in ('G0','G1') and r['group'] in ('context_N2to4','context_N5to8','context_N9plus','speed_stationary','speed_slow','speed_walking','speed_fast','legal_neighbors_none','legal_neighbors_present','same_direction','opposing_direction','crossing_direction')]
    group_table=table(decision_groups,[('录像','fold'),('模型','model'),('组','group'),('目标窗口','target_windows'),('ADE','ADE_m'),('FDE','FDE_m'),('ΔADE对G0','delta_ADE_vs_G0'),('ΔADE对Last2','delta_ADE_vs_CV-Last2'),('ΔADE对OLS4','delta_ADE_vs_CV-OLS4'),('ΔADE对OLS8','delta_ADE_vs_CV-OLS8')])
    (REPORT/'05_FINAL_RESEARCH_DECISION.md').write_text(f'''# 最终验收决策

{cv_conclusion} {social_conclusion} {recommendation}

{gate_table}

## 八个核心问题

1. **是否超过同目标CV？** {cv_conclusion} 已重算CV而非借用旧队列。不能仅以G1超过G0替代强解析基线验收。
2. **训练预算是否改变判断？** 30轮确有未收敛迹象，全部统一续训后神经数值明显变化；结论以完整30/100/last100表为准。有限100轮不是充分收敛证明，但本次预先固定终点后不再延长，不能无限训练寻找超过CV的轮次。
3. **G1是否跨录像优于G0？** {social_conclusion} 单seed、两个历史曝光录像，不支持跨seed稳定声明。
4. **社会收益独立归因？** 相同1553参数、初始化、归一化、损失、decoder和数据控制了容量/架构混杂；图确有非零依赖已在Stage4.2验证。但独立训练的无邻居目标也可因共享权重更新而变，固定checkpoint换图又可能分布外；目前不能声称具体社会关系带来稳定独立预测优势。G2未训练，未补造证据。
5. **PTN系统风险？** 1mm固定图、预注册重复风险判据通过的预算/录像/模型/组条目数为{len(risks)}（条目相互相关，非独立实验数）。全量比例与不同人数量见04报告；未通过或空组同样列出。固定权重无NaN不等于稳健；局部PTN高增益和零/小时间方差提示可定位风险，但未做移除PTN因果消融，不将其当唯一原因。
6. **是否作为第四章基础？** {recommendation} 常规CV/归一化/等变性质均不包装成新创新。
7. **是否停止现有GNN路线？** 停止当前实现和在其上继续加高阶、关系GRU、注意力、gate的路线。此结论限定本家族/预算/两曝光录像，不等于证明所有社会信息或所有图网络没有价值，也不改动历史失败结论。
8. **下一公开模型？** 优先把EqMotion作为下一项**版本一致性与协议核查对象**，而非直接宣布新基础。其几何等变和不变交互推理有明确公开结构依据：[论文](https://arxiv.org/abs/2303.10876)、[作者官方代码](https://github.com/MediaBrain-SJTU/EqMotion)。但Stage4本地审计已发现权重strict不匹配、HOTEL/ZARA2官方脚本只有ego上下文、多头K20与本轮K1不直接对应。须先解决源码/权重版本、完整交互路径和数据协议，未解决不作为可信基线；不静默strict=False、不用官方ETH成绩作选择依据。本阶段未运行EqMotion或设计新方法，完整前证据见`../stage4/03_EQMOTION_AUDIT.md`。

## 所有重点运动和人数组

{group_table}

正值为退化，空组不推断。完整所有组/三个预算角色均在`all_group_results.csv`，没有按结果删除困难目标。不得把历史不足或上下文人数混成社会密度；本队列所有监督目标都有完整8/12。A离线完整上下文存在未来可见性筛选，仍不是官方loader完全复现或独立确认。

冻结边界：不换录像、不加seed、不读ETH预测指标、不再延长、不设计复杂模型。下一阶段是否审计EqMotion由用户决定，本阶段到此停止。
''')
    best_table=table([r for r in metrics if r['role']=='budget100_best'],[('录像','fold'),('模型','model'),('ADE(m)','ADE_m'),('FDE(m)','FDE_m'),('目标窗口','target_windows')])
    (REPORT/'FINAL_AUDIT_REPORT.md').write_text(f'''# Stage4.2最终验收总报告

{cv_conclusion} {social_conclusion} {recommendation}

## 唯一100轮预算下的完全配对结果

{best_table}

CV均本次重新计算，ZARA2 5833和HOTEL1053目标窗口共同GT/ID/窗口，数据hash和逐目标history/future SHA保存；旧Stage3.8指标未用于替代。新旧预算独立列出，验证均值ADE选模、FDE配对，同一规则覆盖四个神经run；K1首采样/K20/NLL仅辅助。

延长前930f92d已冻结方案，全部从原epoch30恢复31–100、共280个新checkpoint，无模型/损失/epsilon/学习率修改。100后停止。训练及PTN审查均有实际日志，不把计划算作结果。固定权重扰动记录{stress_execution['raw_rows']}行，全体队列及全部运动组覆盖，不将一个极端案例推广全部目标。

社会归因门槛：

{gate_table}

报告：01冻结协议，02同目标CV，03训练预算，04全量PTN敏感性，05最终研究决策。真实配置/曲线/配对误差/分组/耗时/数据与checkpoint SHA/测试在`results/tables/stage4p2_final/`。新checkpoint只存本地`experiments/stage4p2_final/extension/`；全部历史由SHA保护。

限制：seed17、两个反复曝光开发录像、重叠窗口非独立、100仍可能未充分收敛。ETH封存，未运行正式测试或额外种子/G2/EqMotion，不把本结果外推所有GNN。下一路线建议为EqMotion公开版本兼容性审查，依据[原论文](https://arxiv.org/abs/2303.10876)与既有Stage4失败记录，尚不具备直接用其开发新方法的条件。

运行必要测试并校验保护清单，证据见`results/tables/stage4p2_final/delivery_checks.json`。仅本地commit，不push；最终SHA以git log为准。用户可手动`git push -u origin feat/stage4p2-final-audit`。完成后停止。
''')
    Path(ROOT/'decision_summary.json').write_text(json.dumps(dict(stop_current_GNN_route=stop,CV_dominates_all_neural_ADE_FDE=worse_all,
        both_recordings_social_gate=strong,improved_recordings=improved,gates=gates,PTN_repeated_risk_entries=len(risks),next_candidate='EqMotion version/protocol audit only',
        ETH_evaluated=False,more_training_planned=False),ensure_ascii=False,indent=2)+'\n')
    print(cv_conclusion,social_conclusion,recommendation,flush=True)


if __name__=='__main__':main()
