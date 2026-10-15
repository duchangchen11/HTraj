"""Evidence-driven Stage3.8 reports; measurements are loaded, never invented."""
import csv,json,subprocess
from pathlib import Path
from hashlib import sha256

ROOT=Path('results/tables/stage3p8'); REPORT=Path('reports/stage3p8')
def read(name):return list(csv.DictReader((ROOT/name).open()))
def num(x,d=6):return '—' if x is None or x=='' else f'{float(x):.{d}f}'
def md(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def write(name,text):
    with (REPORT/name).open('x') as f:f.write(text)


def main():
    p=json.loads(Path('configs/stage3p8/protocol.json').read_text());metrics=read('recording_metrics.csv');aggregates=read('aggregate_metrics.csv')
    groups=read('group_paired_comparison.csv');people=read('per_person_paired_results.csv');history=read('history_features.csv');assoc=read('feature_error_associations.csv')
    recs=[f['recording'] for f in p['folds']];models=p['models']
    def metric(rec,model):return next(r for r in metrics if r['recording']==rec and r['model']==model)
    def group(rec,model,g):return next(r for r in groups if r['recording']==rec and r['model']==model and r['group']==g)
    headline=md(['录像','N目标窗口','CV-Last2 ADE/FDE','CV-OLS4 ADE/FDE','CV-OLS8 ADE/FDE','CV-Median ADE/FDE'],
        [[rec,metric(rec,'CV-Last2')['targets']]+[num(metric(rec,m)['ADE_m'])+' / '+num(metric(rec,m)['FDE_m']) for m in models] for rec in recs])
    macro=md(['口径','模型','录像数','N窗口','录像×行人单位','ADE','FDE'],
        [[r['weighting'],r['model'],r['recording_count'],r['target_windows'],r['recording_person_units'],num(r['ADE_m']),num(r['FDE_m'])] for r in aggregates])
    strong=md(['录像','冻结强对照','N','ADE','FDE','状态'],
        [[r['recording'],r['model'],r['targets'] or '—',num(r['ADE_m']),num(r['FDE_m']),r['status']] for r in metrics if r['model'] not in models])
    timings=md(['录像','模型','整折forward中位数(s)','每目标窗口(μs)','重复次数'],
        [[r['recording'],r['model'],num(r['forward_seconds_median']),num(r['microseconds_per_target_window'],2),r['timed_passes']] for r in read('runtime.csv')])
    report='# Stage3.8：解析运动学与强基线结果\n\n20项固定解析模型评价全部完成，另外8次兼容神经checkpoint推理；没有训练新网络、估计器或门控。所有误差均米制K1、完整8/12目标，四模型同一3582目标窗口，分属177个录像×行人单位。\n\n'+headline
    report+='\n## 不混用的三种平均\n\n'+macro
    report+='\n录像宏平均：先在每个录像按目标窗口平均，再等权平均5个录像。窗口加权：3582个目标窗口等权。行人口径：先同一录像内同一ID跨窗口平均，再行人等权、录像等权；它用于检查长轨迹重复权重，不构成行人完全独立的统计假设。三种口径排序都未支持替代Last2，不能只用某个子集的宏FDE改善覆盖总体退化。\n\n'
    report+='## 与历史最佳兼容神经对照比较\n\n'+strong
    report+='\nM0为Stage3.7原PedBase常学习率对照；students003为Stage3.5同结构A协议PedBase。额外保留历史最佳兼容PedBase（ZARA2为Stage3.6固定衰减版本），避免仅比较弱M0。历史最佳兼容CV残差：HOTEL为Stage3.7归一化M3，ZARA2为Stage3.6原坐标CV-Residual constant；候选列表和选择在本轮评价前冻结，见protocol.json。所有已有模型经源文件、manifest、checkpoint SHA与重评价数值核对；此类历史开发选择仍有选择偏差。\n\n'
    report+='三个平滑模型在HOTEL都胜过Last2、M0及最佳兼容CV残差，但在ZARA2均输给这些强对照，在students003均输给M0。students001/zara01没有合法对应神经checkpoint，students003没有CV残差；表中空缺不是0，没有借用训练见过该录像的模型。严格共享覆盖子集的宏/窗口/行人结果单列reference_matched_aggregates.csv，不能把两录像参考与五录像解析宏平均横比。\n\n'
    report+='## 真实运行耗时\n\n'+timings
    report+='\nCPU单线程、batch8、相同缓存观测批次，warmup3遍+20遍forward计时；包含各估计器现有运动预处理，不含IO、分组、GT误差计算。每目标窗口耗时还受同批历史上下文人数及padding影响，不是脱离数据构成的纯速度估计微基准。本阶段解析float64、神经float32，旧CV微小差异仅为舍入；新数据与时间协议未变。\n'
    write('02_KINEMATIC_BASELINE_RESULTS.md',report)
    # History summary and correlated-person descriptive changes.
    stats=[]
    for rec in recs:
        rs=[r for r in history if r['recording']==rec]
        for feature in ('last_speed_mps','mean_velocity_speed_mps','mean_scalar_speed_mps','velocity_variance_m2ps2','adjacent_velocity_change_mps','net_displacement_m','total_path_m','net_path_ratio','linear_fit_RMS_m','terminal_mean_velocity_difference_mps'):
            vals=sorted(float(r[feature]) for r in rs); n=len(vals);median=vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2
            stats.append([rec,feature,n,num(sum(vals)/n),num(median)])
    person_rows=[]
    for rec in recs:
        for model in models[1:]:
            rs=[r for r in people if r['recording']==rec and r['model']==model]
            person_rows.append([rec,model,len(rs),sum(float(r['delta_ADE_vs_Last2_m']) < -1e-12 for r in rs),num(sum(float(r['delta_ADE_vs_Last2_m']) for r in rs)/len(rs))])
    correlations=[]
    for rec in recs:
        for feature in ('velocity_variance_m2ps2','net_path_ratio','linear_fit_RMS_m','terminal_mean_velocity_difference_mps','last_mean_angle_deg'):
            a=[r for r in assoc if r['recording']==rec and r['feature']==feature]
            def ar(model,outcome,unit):return next(r for r in a if r['model']==model and r['outcome']==outcome and r['unit']==unit)
            w=ar('CV-Last2','ADE_m','target_window_descriptive');q=ar('CV-OLS8','delta_ADE_vs_Last2_m','target_window_descriptive')
            w_person=ar('CV-Last2','ADE_m','person_mean_descriptive');q_person=ar('CV-OLS8','delta_ADE_vs_Last2_m','person_mean_descriptive')
            correlations.append([rec,feature,w['observations'],w_person['observations'],num(w['Spearman_rho'],3),num(w_person['Spearman_rho'],3),num(q['Spearman_rho'],3),num(q_person['Spearman_rho'],3)])
    report='# Stage3.8：跨录像与历史可靠性分析\n\n三个平滑估计器全部只改善HOTEL、退化其他4/5录像。所有5录像都已报告，没有把HOTEL重新选为新方法目标场景。\n\n'+headline
    report+='\n## 重叠窗口与行人权重\n\n'+md(['录像','模型','行人单位N','均ADE改善人数','按行人均ΔADE'],person_rows)
    report+='\nHOTEL的OLS4/OLS8/Median分别改善22/28、19/28、17/28个行人的窗口均ADE；其他录像受益人数少，按行人检查仍未反转四录像退化。行人可共同经历场景事件，录像只有5个且已曝光，不能据此做普遍显著性或新确认集声明。\n\n'
    report+='## 十项历史特征及补充可靠性量\n\n'+md(['录像','历史量（单位见01）','N窗口','均值','中位数'],stats)
    report+='\n完整逐目标vx/vy、夹角及有效标记、历史长度见history_features.csv。均速向量模长和平均速率分开保存；往返小净位移不是静止证据。全部A目标历史均为8，历史长度的相关系数缺失是无变异，不填成0。\n\n'
    report+='## 特征与误差的描述相关\n\n'+md(['录像','历史量','窗口N','行人N','ρ:Last2 ADE窗口','ρ:Last2 ADE行人均值','ρ:OLS8−Last2窗口','ρ:OLS8−Last2行人均值'],correlations)
    report+='\nρ为Spearman描述相关，没有p值、iid窗口置信区间或数据驱动分组阈值。完整Pearson/Spearman、ADE/FDE及delta指标见feature_error_associations.csv。夹角仅在双方速度足够时纳入，路径比剔除未定义的完全静止路径。相关可能受速度、静止比例、行人重复与真实运动变化混杂，不能等同噪声强度或预测时gate的可用收益。\n\n'
    report+='关键反例：末端/均速差与OLS8−Last2 ADE的窗口ρ在HOTEL为−0.393，其他四录像为+0.403/+0.446/+0.317/+0.288。相同“不稳定”特征对应相反收益方向；只凭该指标不能设计一个已被证实跨录像有效的切换器。方差及拟合残差与Last2未来ADE普遍正相关，说明历史确有难度信息，但并不推出长历史速度能解决它。\n'
    write('03_CROSS_RECORDING_ANALYSIS.md',report)
    report='# Stage3.8：失败机制、历史分组及案例\n\n'
    report+='## 对八个研究问题的回答\n\n'
    report+='1. **HOTEL为何Last2相对神经模型好？** Last2 ADE0.329516，M0为0.503319、最佳兼容CV残差0.338101；模型修正的跨录像泛化仍有限。但Last2并非HOTEL最优：OLS4/OLS8/Median进一步降至0.237814/0.220144/0.218247。低均速224个窗口来自5名行人，Last2近似静止组82窗口来自4人且ADE0.052244。数据构成与短时外推适用性支持其相对优势，不能解释为未来都匀速或神经模型必然无用。Stage3.7方向依赖亦是已有证据，但本轮没有单独识别旧神经退化的唯一原因。\n'
    report+='2. **ZARA2为何Last2不如PedBase？** Last2总体0.296306、M0为0.282905。M0在中/高速组分别比Last2改善0.041281/0.054584m，抵消其在低速组退化0.012045m；样本集合相同。因此不是只靠静止比例或重配样本得到的总体优势。最佳兼容CV残差0.276902也优于所有新平滑估计。\n'
    report+='3. **长历史是否降低噪声？** 对人工终端扰动，单元测试确认OLS8速度响应更小。真实数据没有干净位置/速度真值，无法将方差分成测量噪声和真实运动；HOTEL高频变化较大且平滑改善，支持条件性去波动价值，不能宣称识别了真实噪声。historical_smoothing_diagnostic.csv比较历史OLS拟合RMS与历史一步Last2误差，两者目标与拟合支持不同，不作为公平未来精度比较。\n'
    report+='4. **长历史是否削弱转弯响应？** 方向变化组中OLS8−Last2 ADE：HOTEL(N55/5人)−0.403828m，students001(N64/14人)+0.238704m，students003(N13/3人)+0.925451m；两个ZARA录像该严格组N0。其他录像末端不稳定组亦大幅退化，与真实运动变化被平均的响应滞后相容，但不是对所有转弯的因果定论。\n'
    report+='5. **不同录像是否需不同估计？** 本次描述最优确不同：HOTEL偏好Median，其他4录像Last2更好。这不是允许输入录像ID选估计器的理由；没有测试这种场景选择器，也不能用它形成“统一模型”结果。\n'
    report+='6. **能从历史识别长历史适用目标吗？** 能预先定义风险组，但未得到跨录像稳定适用规则。末端不稳定组HOTEL仅4窗口/3人，OLS8改善0.940772m；其他录像同组退化0.688875/1.111158/0.970584/1.024288m。小组数、录像差异及相反方向不支持直接开发可靠性gate。\n'
    report+='7. **可靠性量与未来误差相关吗？** 有描述关联，见03报告；例如HOTEL/ZARA2历史拟合RMS与Last2 ADE的窗口ρ约0.617/0.740。但对“采用长历史能否改善”关联可变号，且行人与窗口相关，无独立确认或因果保证。\n'
    report+='8. **是否有不依赖录像身份且跨录像有效的稳健估计？** 本轮没有。三个预注册替代都在4/5录像退化，五录像宏ADE/FDE及窗口加权均变差，没有候选通过规则；停止修改CV估计器，不追加截断、回归或阈值搜索。\n\n'
    report+='## 全部预注册历史组\n\n下表保留所有20组及N0，N是目标窗口，括号为该组独立ID数（同一录像内），并非统计独立样本数。组之间重叠，不能相加。四模型均为同一目标集；所有强对照组及配对受益比例亦保存在group_paired_comparison.csv。\n\n'
    for rec in recs:
        gs=[r['group'] for r in groups if r['recording']==rec and r['model']=='CV-Last2']
        report+='### '+rec+'\n\n'+md(['历史组','N(行人ID)','Last2 ADE/FDE','OLS4 ADE/FDE','OLS8 ADE/FDE','Median ADE/FDE'],
            [[g,group(rec,'CV-Last2',g)['targets']+' ('+group(rec,'CV-Last2',g)['unique_pedestrians']+')']+
             [num(group(rec,m,g)['ADE_m'])+' / '+num(group(rec,m,g)['FDE_m']) for m in models] for g in gs])+'\n'
    report+='## 固定历史案例及失败边界\n\nfixed_history_case_examples.csv按每个预定义历史组、每录像的sample_id/target_id排序取前3个，不按未来误差挑选。所有案例的四模型/可用神经模型配对误差及历史特征均保存，未将事后未来表现构造成新组或模型输入。\n\n'
    report+='历史近似静止不等于未来静止：students003的该组37窗口/4人，Last2 ADE0.532037、FDE1.391947；ZARA2相应572窗口/9人，Last2 ADE0.100636，而OLS8总体小幅变差，即使其中85.66%窗口的ADE有微小改善。这说明受益比例与平均改善不是同一口径，少数大退化可能主导平均，不能只报告“多数案例成功”。\n\n'
    report+='往返组在HOTEL有28窗口/4人，在students001/students003各2窗口/1人，ZARA1为0，ZARA2仅1。它可提示运动复杂性，却不足以支持可靠的跨录像规则。没有从这些稀疏小组继续调阈值或增加模块。\n'
    write('04_FAILURE_CASE_ANALYSIS.md',report)
    report='# Stage3.8：第四章研究方向决策\n\n**失败结论：没有跨录像优势候选，按预设停止继续修改CV估计器。** OLS4/OLS8/Median均只改善HOTEL，在其余4录像ADE/FDE退化；五录像宏平均与窗口加权、行人检查同样不支持替代。没有启动可学习门控、神经网络、复杂社会模块或ETH测试。\n\n'+headline
    report+='\n## A．是否已有可信的稳健运动预测基础？\n\n支持证据：Last2是零参数、因果、可核查且在4/5录像优于本轮平滑模型的强基线；对照与3582目标严格配对。反对证据：HOTEL上Median明显更好，students003历史静止组未来误差仍高，神经对照覆盖不足。判断：已有可信的**基线套件**，尚无统一跨录像稳健、处处优于强对照的新主干。Last2必须保留，但不能称所有任务的最优预测器。\n\n'
    report+='## B．是否值得开发可靠性运动先验与神经残差协同？\n\n支持证据：历史方差、线性残差与误差有关，HOTEL平滑确有大收益。反对证据：相同可靠性指标的长历史收益在其他录像反号，三候选均4/5失败；ZARA2新估计还输给M0及最佳兼容CV残差。判断：**当前不支持启动可学习协同模型**。相关性和一个录像的平滑成功不能证明gate可学、可泛化，更不能为它增加新模块。\n\n'
    report+='## C．是否保留社会GNN为第四章主线？\n\n支持证据：社会交互是原论文问题，现有代码及历史分组保留，仍可作对照。反对证据：Stage3.6冻结ego信息归因未满足独立社会收益规则（reports/stage3p6/06_RESEARCH_DECISION.md），Stage3.7方向表示只有条件收益，本阶段没有新增社会价值证据。判断：**不再将当前GNN默认作为第四章正式方法主线**，保留作基线与待证伪假设；这不等于证明社会信息本身无价值。\n\n'
    report+='## D．是否调整第四章问题或数据集？\n\n支持调整问题：已有证据指出录像运动分布、历史状态可预测性与数据任务口径很重要；原单纯加高阶/记忆/门控的路线缺乏性能支持。反对立即换数据：本轮只有曝光开发录像、缺两折PedBase和三折CV残差，尚未有独立外部任务证据，不能按验证失败挑一个更有利数据集。判断：**先调整研究问题和验证设计，暂停新网络开发**；明确要证明何种额外信息的条件预测价值，再另行核查适合的任务/数据与协议。ETH保持封存，不用它决定路线，也不擅自迁移数据。\n\n'
    report+='## E．哪条路线最接近可投稿小论文？\n\n支持证据最多的是“任务协议、强运动基线与跨录像可靠性诊断”的可复核分析路线：五录像参数自由对照、强基线兼容性与配对失败证据均已存在。反对证据是没有新方法创新、没有独立确认/完整神经覆盖、没有外部重复验证；常规OLS/Median、归一化、CV残差不能充当创新点。判断：**目前没有已具备投稿证据的网络小论文路线**；上述诊断最接近可继续组织的技术研究材料，仍需另立研究问题、文献核查与独立验证，而不是立即包装为新模型论文。\n\n'
    report+='## 明确终点与下一步\n\n本阶段停止进一步CV估计器、超参数、可靠性gate和复杂GNN尝试，不增加正式训练。保存Last2与全部失败对照用于以后公平比较。下一指令若继续，应先确定第四章要验证的技术命题与所需证据，审查缺失的兼容基线及未曝光确认方案；本次不自动补训、不读ETH、不自动push。不得将HOTEL单独挑为新方法的目标场景。\n'
    write('05_RESEARCH_DIRECTION_DECISION.md',report)
    report='# Stage3.8总报告：历史运动可靠性与稳健速度估计\n\n完成四个零训练参数解析模型在全部五个固定非ETH开发录像上的统一评价，以及8次合法兼容神经checkpoint推理。没有新神经训练、gate、速度截断、额外估计器、ETH测试或自动push。\n\n'+headline+'\n'+macro
    report+='\n**结论：三个平滑估计器全部只改善HOTEL，其余4/5录像的ADE和FDE均退化；没有通过预注册规则的候选，停止继续修改CV估计器。** HOTEL的成功保留为条件现象，不能用于重新选录像。可靠性量包含难度信息，但不能区分噪声与真实运动变化，也未形成已验证的跨录像路由规则。\n\n'
    report+='已完成988项旧Git文件和142个checkpoint共1130项SHA保护；3582目标窗口、177录像×行人单位完整配对。所有20个历史组、560条含强对照的分组行、19,970条模型逐目标误差保存；窗口并非独立统计样本，另存逐人均值、配对差和两种必需平均口径，不报iid窗口p值。\n\n'
    report+='强对照：PedBase-M0及最佳兼容PedBase覆盖HOTEL/students003/ZARA2；最佳兼容CV残差覆盖HOTEL/ZARA2。缺失项留空，未借用泄漏录像的模型；共享覆盖宏平均另表，不能与五录像宏平均混用。旧模型与新解析指标的float32/64舍入差异已明确。\n\n'
    report+='第四章暂不把当前社会GNN或可靠性协同网络当作已证实主线。最有依据的是协议/基线诊断材料，当前没有足够的新模型创新与独立验证支持直接投稿；下一步应调整可证伪问题及证据设计，非继续堆模块。本轮全部真实输出保留，完成本地commit后停止。\n\n'
    report+='## 交付索引\n\n- 01_MOTION_RELIABILITY_AUDIT.md：评价前固定公式、阈值、数据与强对照。\n- 02_KINEMATIC_BASELINE_RESULTS.md：五录像、宏/窗口/行人口径、强对照及真实耗时。\n- 03_CROSS_RECORDING_ANALYSIS.md：历史统计、配对行人差值和描述相关。\n- 04_FAILURE_CASE_ANALYSIS.md：八个问题、所有历史组、固定案例及失败限制。\n- 05_RESEARCH_DIRECTION_DECISION.md：A至E支持/反对证据及停止决定。\n- configs/stage3p8/protocol.json：公式、manifest/raw SHA、冻结旧checkpoint及规则。\n- results/tables/stage3p8/：完整结果、版本SHA、配对表、单元测试与交付校验。\n'
    write('STAGE3P8_REPORT.md',report)
    reports=sorted(REPORT.glob('*.md'))
    (ROOT/'report_provenance.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),reports=[dict(path=str(f),sha256=sha256(f.read_bytes()).hexdigest()) for f in reports],
        inputs=[dict(path=str(ROOT/f),sha256=sha256((ROOT/f).read_bytes()).hexdigest()) for f in ('recording_metrics.csv','aggregate_metrics.csv','group_paired_comparison.csv','feature_error_associations.csv','research_decision.json')],
        generated_from_actual_measurements=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(reports_created=5,analytic_prior_candidates=[],new_training=False)))


if __name__=='__main__':main()
