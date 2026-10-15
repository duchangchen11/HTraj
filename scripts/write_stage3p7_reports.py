"""Render completed evidence tables; never creates experiment measurements."""
import csv
import json
import subprocess
from pathlib import Path
from hashlib import sha256

ROOT = Path('results/tables/stage3p7'); REPORT = Path('reports/stage3p7')


def rows(name): return list(csv.DictReader((ROOT/name).open()))
def number(v, digits=6): return '—' if v is None or v == '' else f'{float(v):.{digits}f}'
def table(headers, data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in data)
def write(name, text):
    with (REPORT/name).open('x') as f: f.write(text)


def main():
    models = rows('model_comparison.csv'); groups = rows('group_comparison.csv'); stability = rows('training_stability.csv')
    decision = json.loads((ROOT/'research_decision.json').read_text()); a = json.loads((ROOT/'analysis_provenance.json').read_text())
    p = json.loads(Path('configs/stage3p7/protocol.json').read_text()); recs = [f['recording'] for f in p['folds']]
    def get(rec, name): return next(r for r in models if r['recording'] == rec and r['model'] == name)
    def group(rec, name, g): return next(r for r in groups if r['recording'] == rec and r['model'] == name and r['endpoint'] == 'best' and r['group'] == g)
    summary = table(['录像','模型','参数','best轮','best ADE/FDE(m)','last ADE/FDE(m)','新训练耗时(s)'],
        [[r['recording'], r['model'], r['parameter_count'], r['best_epoch'] or '—',
          number(r['best_validation_ADE_m'])+' / '+number(r['best_validation_FDE_m']),
          number(r['last_validation_ADE_m'])+' / '+number(r['last_validation_FDE_m']),
          number(r['training_seconds'], 2) if r['new_training'] == 'True' else '—（无新训练）'] for r in models])
    differences = table(['录像','对照','Δbest ADE(m)','Δ配对FDE(m)'],
        [[rec, name+' − '+base,
          number(float(get(rec,name)['best_validation_ADE_m'])-float(get(rec,base)['best_validation_ADE_m'])),
          number(float(get(rec,name)['best_validation_FDE_m'])-float(get(rec,base)['best_validation_FDE_m']))]
         for rec in recs for name,base in [('M2','M0'),('M3','M0'),('M3','M2'),('M3','CVR-Unnormalized-reference'),('M2','CV'),('M3','CV')]])
    endpoint = table(['录像','模型','best训练ADE','best验证ADE','last训练ADE','last验证ADE'],
        [[r['recording'],r['model'],*[number(r[x]) for x in ('best_train_ADE_m','best_validation_ADE_m','last_train_ADE_m','last_validation_ADE_m')]]
         for r in models if r['model'] != 'M1'])
    model_result = '# Stage3.7：个体运动基线真实结果\n\n'
    model_result += '只使用固定A完整轨迹协议、两个历史曝光的录像隔离开发划分、seed17。M0/M2/M3各两次全新训练30轮，共6次/180轮；M1是M0等价别名，CV为解析基线。原坐标CV残差来自Stage3.6经源码、数据、优化器、初始化、预算及评价器核查的冻结结果，不计本阶段新训练。所有FDE与对应ADE使用同一checkpoint。\n\n'+summary
    model_result += '\nM0=M1；M2=历史朝向局部坐标直接位移；M3=同局部坐标CV残差。各神经版本参数量35,352，初始参数SHA相同。每个模型best/last均在完全相同目标上评价，不删除难例。数据源及manifest SHA见固定配置与各run的config.json。\n\n'+differences
    model_result += '\n## 同一endpoint的训练与验证误差\n\n'+endpoint
    model_result += '\n上述训练集误差来自训练完成后同一checkpoint的确定性推理，不等于每轮在线更新中累计的训练loss。逐轮在线loss、梯度范数及验证ADE/FDE见`all_epoch_curves.csv`；原始逐run结果与本地best/last checkpoint SHA在各模型summary.json。M0重复性见`M0_reproducibility.csv`。\n\n'
    model_result += f'本阶段实际神经训练耗时{a["training_seconds"]:.2f}s（{a["training_seconds"]/60:.2f}分钟），不含全部准备/评价/文件IO。实际optimizer更新{a["optimizer_updates"]}次。RTX3080，单CPU线程；没有延长训练、调学习率或额外seed。\n'
    write('03_BASELINE_RESULTS.md', model_result)
    # Full best-group table: no selective presentation of successful strata.
    group_table = table(['录像','历史组','N','模型','ADE','FDE','ΔADE vs M0','ΔADE vs CV'],
        [[r['recording'],r['group'],r['ADE_count'],r['model'],number(r['ADE_m']),number(r['FDE_m']),number(r['delta_ADE_m_vs_M0']),number(r['delta_ADE_m_vs_CV'])]
         for r in groups if r['endpoint'] == 'best'])
    rotation_rows = [r for r in rows('rotation_checks.csv') if r['rotation_degrees'] == '90' and r['model'] in ('M0','M2','M3')]
    rotations = table(['录像','模型','历史组','N','90°预测等变平均误差(m)','最大误差(m)'],
        [[r['recording'],r['model'],r['group'],r['targets'],number(r['equivariance_mean_error_m']),number(r['equivariance_max_error_m'])] for r in rotation_rows])
    sensitivities = table(['录像','模型','推理阈值(m/s)','切换目标数','N','ADE','FDE'],
        [[r['recording'],r['model'],r['inference_threshold_mps'],r['reliability_switch_targets'],r['ADE_count'],number(r['ADE_m']),number(r['FDE_m'])]
         for r in rows('threshold_sensitivity.csv') if r['group'] == 'all'])
    generalization = '# Stage3.7：泛化、坐标敏感性与完整历史运动分组\n\n'
    generalization += '训练前历史统计表明两个录像的运动分布并不一致。可靠朝向目标中，训练集的纵向运动比例在ZARA2留出划分约29.06%、HOTEL留出划分约24.92%；验证分别约0.68%与89.36%。此处纵向指|vy|>|vx|、平均历史速度≥0.2m/s，只分析历史；方向与录像相关但不是唯一可能混杂。原始八扇区计数及所有连续特征见history_distribution.csv。\n\n'
    generalization += '预设速度组使用当前连续历史段平均速度：低速<0.2，中速[0.2,1)，高速≥1m/s，无可用速度另列。HOTEL低速224/318，ZARA2低速648/1092。只评价A完整历史目标，因此部分历史组为空，仍显式保留。所有组会重叠，不能相加当作独立样本；窗口复用同一行人，N是目标窗口数而非独立行人数。\n\n'
    generalization += '## 坐标变化与优化的区别\n\n'+differences
    generalization += '\nM2−M0保持参数、初始化、数据、loss及训练策略相同，改变局部坐标的计算归纳偏置；M3−原坐标CV残差保持CV先验。M3−M2则主要检验CV主路径。这里坐标变换也会改变梯度方向及优化过程，不能从单seed、各自best checkpoint的误差完全排除“优化更容易”。需要同时看固定last30和同endpoint训练误差，不能把best提升直接等同独立方向因果证明。没有进行学习率搜索或添加更多训练。\n\n'+endpoint
    generalization += '\n## 合法全局旋转检查\n\n'+rotations
    generalization += '\n±90°、180°的全部数据见rotation_checks.csv。只对完成训练后的固定best checkpoint同步旋转历史输入与指标GT，检查预测应如何变换；GT仅用于误差计算。可靠方向组的M2/M3应在浮点误差范围等变，恒等低速回退不具有该保证。M0通常依赖全局轴方向。旋转指标不用于再选模型、阈值或checkpoint，也不是新增独立训练任务。\n\n'
    generalization += '## 阈值敏感性\n\n'+sensitivities
    generalization += '\n训练阈值固定0.2m/s；0.1和0.4只在同一best权重推理时切换。分组始终沿用原0.2定义，记录切换目标数。完整逐组结果见threshold_sensitivity.csv；这是一项接口干预，不能替代预注册的独立阈值重训练或用最低干预误差覆盖主结果。\n\n'
    generalization += '## 全部历史组主结果\n\n'+group_table
    generalization += '\n全部best/last分组及空组见group_comparison.csv，共560条；逐目标配对误差见paired_target_errors.csv，共14,100条，目标集合逐项核对完全一致。不能将只在某组改善推广为整体稳定收益，也不能仅凭单seed证明跨随机性稳定。\n'
    write('04_GENERALIZATION_ANALYSIS.md', generalization)
    candidate = decision['eligible_for_multiseed_proposal']
    decision_text = '# Stage3.7：研究决策与停止点\n\n'
    decision_text += ('预设双录像候选规则通过：'+', '.join(candidate)+'。这只是seed17候选迹象；可以提出下一轮seed42/2026建议，本阶段不启动。\n\n' if candidate else
                      '没有模型通过预设双录像候选规则：不提出立即进行三seed训练的结论，停止新增训练。单个录像或分组的改善仍保留，但不能宣称稳定通用优势。\n\n')
    decision_text += '规则：同一归一化版本在两个固定录像上均比M0减少ADE至少max(0.005m,0.5% M0 ADE)，且各自配对FDE不恶化。录像没有因结果重新选择。规则不等同优于CV，是否仍输给CV必须另报。\n\n'+differences
    decision_text += '\n## 回答六个研究问题\n\n'
    decision_text += '1. **跨录像泛化与坐标是否有关？** 原模型已平移等变，因此“没有去除绝对位置”不成立。方向表示存在明确结构依赖、也存在历史方向分布偏移；上表M2−M0给出本轮净效应。只有两个曝光录像、单seed和混合运动分布，不能归结为唯一原因。\n'
    decision_text += '2. **是否值得显式建模历史方向？** 可靠历史方向可消去全局轴依赖；是否有预测价值需同时看两录像、CV和低/高速组。低速恒等回退仍有轴依赖，因此不能把该实现称无条件旋转等变模型。它本身属于常规表示对照。\n'
    decision_text += '3. **CV残差是否更可靠？** 分别比较原坐标CV残差−M0以及M3−M2；不能混合归一化与CV先验后把全部收益归给CV。best与last、训练/验证误差均已给出；即便某一录像提升也不构成跨seed可靠性证据。\n'
    for rec in recs:
        for name in ('M2','M3'):
            harmed = []
            for g in ('history_speed_low','history_speed_medium','history_speed_high','history_CV_like','history_velocity_changing','heading_reliable','heading_fallback'):
                r = group(rec,name,g)
                if r['ADE_count'] != '0' and float(r['delta_ADE_m_vs_M0']) > 0:
                    harmed.append(f'{g}(N={r["ADE_count"]},ΔADE={float(r["delta_ADE_m_vs_M0"]):+.6f}m)')
            decision_text += f'4. **{rec}/{name}仍退化的历史组（相对M0）：** '+('；'.join(harmed) if harmed else '上述七个预设重点组未出现ADE退化；仍需检查CV及FDE')+'。完整全部组保留在04报告，不排除失败样本。\n'
    decision_text += '5. **是否值得立即重评社会GNN？** 本轮没有社会网络实验，不能声称社会收益已恢复。先确认个体主路径在两个录像上稳定、与CV及容量控制比较成立，再单独授权冻结ego条件下评估；不得直接复用归一化编码中的非prefix因果中间状态。当前不自动启动GNN。\n'
    decision_text += '6. **是否已有第四章创新路线？** 没有足够证据。坐标归一化、CV残差、GRU都不是本项目新创新。当前证据最多用于定位运动表示/录像分布问题与定义下一项可证伪实验，不能作为正式创新网络的性能主张。\n\n'
    decision_text += 'HOTEL的CV无需拟合其他录像的运动方向、静止比例和未来修正，神经模型则可能把训练域修正迁移到不匹配录像。即使训练loss下降仍可能验证退化。方向、速度变化、低速噪声与优化等假说并存，本实验没有唯一因果识别。若归一化仍不能同时击败CV，不继续堆社会模块；可提议明确区分可靠运动估计与残差适用条件的下一轮基础对照，需新指令。\n\n'
    decision_text += '停止点：六次有限训练、预设固定权重诊断与完整测试交付；不评价ETH，不启动第五章、多模态、高阶或关系GRU，不自动push。\n'
    write('05_RESEARCH_DECISION.md', decision_text)
    summary_text = '# Stage3.7总报告：运动表示与跨录像泛化\n\n'
    summary_text += '已完成源码审计、历史保护、六次seed17有限训练与预设补充诊断。原PedBase没有把绝对位置输入encoder，且已通过最后位置锚点实现平移等变；M1是M0的等价别名，没有伪造独立训练。实际新增对照为历史朝向局部坐标M2与同坐标CV残差M3，参数量均35,352，训练配置相同。\n\n'+summary
    summary_text += '\n'+('预设候选：'+', '.join(candidate)+'，仅建议后续多seed验证，本轮不追加训练。' if candidate else '没有通过预设双录像候选规则的模型，停止新增训练。')+'\n\n'
    summary_text += '不将坐标归一化、CV残差或GRU称为新创新；单seed、历史曝光录像和相关窗口的限制明确保留。可靠朝向旋转等变与低速恒等回退的例外分别测试。M0重训逐轮复核与Stage3.6相同，原坐标CV残差经兼容性核查作为冻结参考。\n\n'
    summary_text += f'实际6次/180epoch，{a["optimizer_updates"]}次更新，纯训练{a["training_seconds"]/60:.2f}分钟；新增12个checkpoint仅在本地忽略目录。所有数据原始来源hash、manifest SHA、训练源码commit及文件SHA均保存。保护973项旧成果（含130旧checkpoint），ETH封存，协议B未运行，无自动多seed或push。完整测试结果见unit_tests.xml，交付核验见DELIVERY_AUDIT.json。\n\n'
    summary_text += '## 文件索引\n\n- 01_MOTION_REPRESENTATION_AUDIT.md：训练前八项源码/数据审计。\n- 02_COORDINATE_NORMALIZATION.md：公式、张量、复杂度、回退及因果限制。\n- 03_BASELINE_RESULTS.md：best/last、配对FDE、参数/耗时、训练/验证。\n- 04_GENERALIZATION_ANALYSIS.md：全部分组、配对误差、旋转/阈值与优化限制。\n- 05_RESEARCH_DECISION.md：六项科研问题、失败/候选及下一步条件。\n- results/tables/stage3p7/：原始逐run配置/曲线/分组/指标/测试及聚合表。\n\n'
    summary_text += '本轮预注册模型及协议提交见各run config.json的source_commit。训练启动前的CV记录字段错误在0次optimizer更新时修复，原失败记录保留；没有神经训练重试或隐瞒失败。最终本地交付commit由Git记录，用户可自行正常push；不覆盖任何旧研究材料。\n'
    write('STAGE3P7_REPORT.md', summary_text)
    files = [REPORT/name for name in ('03_BASELINE_RESULTS.md','04_GENERALIZATION_ANALYSIS.md','05_RESEARCH_DECISION.md','STAGE3P7_REPORT.md')]
    (ROOT/'report_provenance.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), sources=[dict(path=str(ROOT/x),sha256=sha256((ROOT/x).read_bytes()).hexdigest()) for x in
        ('model_comparison.csv','group_comparison.csv','research_decision.json','analysis_provenance.json')],
        reports=[dict(path=str(f),sha256=sha256(f.read_bytes()).hexdigest()) for f in files], generated_from_actual_measurements=True),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(reports_created=4, candidate_proposal_only=candidate)))


if __name__ == '__main__': main()
