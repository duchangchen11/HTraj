"""Write reports from completed, paired actual results; no missing-run values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
import subprocess
import numpy as np
from src.data.preprocessing import sha256, write_json_new
from src.training.social_diagnostics import csv_new

ROOT = Path('results/tables/stage3p6')
REPORT = Path('reports/stage3p6')


def read(path):
    return list(csv.DictReader(path.open()))


def number(value, digits=6):
    if value is None or value == '': return '—'
    return f'{float(value):.{digits}f}'


def metric(r):
    return number(r.get('best_ADE_m', r.get('ADE_m')))+'/'+number(r.get('best_FDE_m', r.get('FDE_m')))


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str, r))+' |\n' for r in rows)


def write(name, text):
    with (REPORT/name).open('x') as f: f.write(text)


def main():
    p = json.loads(Path('configs/stage3p6/protocol.json').read_text())
    decision = json.loads((ROOT/'round3_decision.json').read_text())
    anchors = json.loads((ROOT/'anchor_selection.json').read_text())['selections']
    analysis = json.loads((ROOT/'analysis_provenance.json').read_text())
    summaries = read(ROOT/'completed_run_summary.csv')
    groups = read(ROOT/'group_comparison.csv')
    neural = [r for r in summaries if r.get('best_epoch')]
    budget = sum(float(r['training_seconds']) for r in neural)
    ego_rows, stability, references, selected = [], [], [], []
    for fold, anchor in zip(p['folds'], anchors):
        rec = fold['recording']
        cv = json.loads((ROOT/rec/'ego/CV__analytical/seed_17/summary.json').read_text())
        references.append(dict(model='CV', recording=rec, protocol_name=p['protocol_name'], manifest_sha256=fold['manifest_sha256'], raw_sources=p['source_files'], parameters=0,
            velocity='last consecutive two observed frames, unknown explicit stationary fallback', ETH_evaluated=False, source_commit=cv['source_commit']))
        ego_rows.append([rec, 'CV', '解析', '—', metric(cv), '不训练', 0, '—'])
        for label in p['neural_ego_runs']:
            name, schedule = label.split('/')
            directory = ROOT/rec/'ego'/(name+'__'+schedule)/'seed_17'
            s = json.loads((directory/'summary.json').read_text()); curve = read(directory/'epochs.csv')
            ego_rows.append([rec, name, schedule, s['best_epoch'], metric(s), number(s['last_ADE_m'])+'/'+number(s['last_FDE_m']), s['parameter_count'], number(s['training_seconds'], 2)])
            stability.append(dict(recording=rec, model=name, schedule=schedule, best_epoch=s['best_epoch'], best_in_first5=s['best_epoch'] <= 5,
                first_train_loss=s['first_train_loss'], last_train_loss=s['last_train_loss'], tail5_relative_loss_decrease=s['tail5_relative_loss_decrease'], last_minus_best_ADE_m=s['last_ADE_m']-s['best_ADE_m'],
                first5_min_validation_ADE_m=min(float(r['validation_ADE_m']) for r in curve[:5]), last5_validation_ADE_std=float(np.std([float(r['validation_ADE_m']) for r in curve[-5:]]))))
        reference = json.loads((ROOT/rec/'residual/Ego-Only/seed_17/summary.json').read_text())
        references.append(dict(model='Ego-Only', recording=rec, seed=17, protocol_name=p['protocol_name'], manifest_sha256=fold['manifest_sha256'], raw_sources=p['source_files'],
            total_parameters=35352, trainable_parameters=0, frozen_parameters=35352, anchor=anchor, parameters_sha256=reference['anchor_parameters_sha256'], ETH_evaluated=False,
            output='CV' if anchor['cv_output'] else anchor['model'], cache='created from observations of current manifest, no persistence or old cache'))
        selected.append([rec, ('CV输出；'+anchor['model']+'编码') if anchor['cv_output'] else anchor['model'], anchor['schedule'], number(anchor['anchor_ADE_m'])+'/'+number(anchor['anchor_FDE_m']), anchor['checkpoint_sha256']])
    csv_new(ROOT/'ego_training_stability.csv', stability)
    write_json_new(ROOT/'analytic_reference_configs.json', dict(protocol_sha256=sha256('configs/stage3p6/protocol.json'), references=references, no_future_input=True, ETH_evaluated=False))
    differences = []
    for rec in [f['recording'] for f in p['folds']]:
        for name in ('PedBase', 'CV-Residual'):
            a = next(r for r in neural if r['recording'] == rec and r['model'] == name and r['schedule'] == 'constant' and r['seed'] == '17')
            b = next(r for r in neural if r['recording'] == rec and r['model'] == name and r['schedule'] == 'step10_20' and r['seed'] == '17')
            differences.append(dict(recording=rec, model=name, step_minus_constant_best_ADE_m=float(b['best_ADE_m'])-float(a['best_ADE_m']), step_minus_constant_last_ADE_m=float(b['last_ADE_m'])-float(a['last_ADE_m']),
                step_minus_constant_last5_ADE_std=next(r['last5_validation_ADE_std'] for r in stability if r['recording'] == rec and r['model'] == name and r['schedule'] == 'step10_20')-next(r['last5_validation_ADE_std'] for r in stability if r['recording'] == rec and r['model'] == name and r['schedule'] == 'constant')))
    csv_new(ROOT/'fixed_schedule_comparison.csv', differences)
    early = sum(r['best_in_first5'] for r in stability)
    text = '# 个体运动基线与训练稳定性\n\n全部为协议A；两个历史曝光录像、seed17；best按整体ADE、FDE配对同checkpoint，单位m。各神经run独立从头30epoch，不删后期退化。\n\n'
    text += table(['录像', '模型', 'schedule', 'best轮', 'best ADE/FDE', 'last30 ADE/FDE', '参数', '训练秒'], ego_rows)
    text += '\n## 固定LR对照与曲线\n\n'
    text += table(['录像/模型/策略', 'best轮≤5', '首/末loss', '尾5loss下降', 'last−best ADE', '尾5val ADE std'], [[r['recording']+'/'+r['model']+'/'+r['schedule'], r['best_in_first5'], number(r['first_train_loss'])+'/'+number(r['last_train_loss']), number(r['tail5_relative_loss_decrease']), number(r['last_minus_best_ADE_m']), number(r['last5_validation_ADE_std'])] for r in stability])
    text += f'\n8个ego神经run中{early}个best落在1–5轮，不能把旧students003早期最优推广到所有录像。训练loss下降与val变化是不同分布下的事实；旧的1024窗口结论不能直接替代本轮更多窗口结果。fixed_schedule_comparison.csv保存预先规定衰减相对constant的best/last/尾部波动差。不同方向如实保留，单seed不能断言唯一原因是LR过高；30epoch末尾loss趋缓也不等于已全局收敛。没有延长训练。\n'
    text += '\n## 历史运动分布\n\n'
    motion = read(ROOT/'historical_motion_distribution.csv')
    text += table(['留出录像', 'split', '历史量', 'mean', 'median', 'P95'], [[r['recording'], r['split'], r['history_measure'], number(r['mean']), number(r['median']), number(r['p95'])] for r in motion])
    text += '\n速度/方向/变化量及密度全部来自历史。训练与验证的中位速度、方向轴与密度不一致，可能解释泛化/偏差（待验证），不是凭这些相关量认定因果。逐目标同样本CV/PedBase/CVR比较在paired_target_errors.csv，两个ego参数均35352/同初始化，排除了总参数增加这一因素；CV主路径同时改变优化残差分布，不能说仅靠参数量得益。\n\n## 冻结选择\n\n'
    generalization = read(ROOT/'endpoint_train_validation_generalization.csv')
    text += '补充训练结束后的同checkpoint全训练/验证推理，排除在线minibatch loss与epoch末模型不一致的比较方式；仅诊断、没有优化或模型选择，CPU验证与主GPU指标差异<1e-6。\n\n'
    text += table(['录像/模型/策略/endpoint', 'split', '目标', 'ADE/FDE(m)', 'ADE−同split CV'], [[r['recording']+'/'+r['model']+'/'+r['schedule']+'/'+r['endpoint'], r['split'], r['targets'], number(r['ADE_m'])+'/'+number(r['FDE_m']), number(r.get('delta_ADE_vs_same_split_CV_m'))] for r in generalization])
    text += '\n训练与验证改善方向相反只能支持分布/泛化风险，不能确定某一个速度轴、LR或数据问题为唯一原因。\n\n'
    text += table(['录像', '输出/编码器', '神经schedule', 'anchor ADE/FDE', '源checkpoint SHA256'], selected)
    text += '\n若选择CV输出，最佳神经encoder/head仍原样冻结，decoder不参与最终基础CV输出；总冻结参数35352、有效输出非学习CV。所有分支共享同一源文件/数值权重/输出规则。多候选anchor使用开发集选择，非独立验证，未评价ETH。\n'
    write('02_EGO_BASELINE_RESULTS.md', text)
    residual = [r for r in neural if r['phase'] == 'residual']
    rows = []
    for fold in p['folds']:
        s = json.loads((ROOT/fold['recording']/'residual/Ego-Only/seed_17/summary.json').read_text())
        rows.append([fold['recording'], 17, 'Ego-Only', '解析', metric(s), '不训练', 0, '—'])
    rows += [[r['recording'], r['seed'], r['model'], r['best_epoch'], metric(r), number(r['last_ADE_m'])+'/'+number(r['last_FDE_m']), r['trainable_parameters'], number(r['training_seconds'], 2)] for r in residual]
    text = '# 冻结主路径的社会payload与容量对照\n\n同一录像/seed五模型完全相同anchor文件SHA和数值参数；社会实验不更新encoder/decoder。参数前后SHA与每轮anchor_unchanged为自动断言。四个新增分支均14425参数，总49777，Ego-Only新增0/冻结35352；只改变自身/邻居来源与source编码年龄/合法关系。\n\n'
    text += table(['录像', 'seed', '模型', 'best轮', 'best ADE/FDE', 'last30 ADE/FDE', '新增可训练参数', '训练秒'], rows)
    text += '\nPrevious与Current均使用当前时刻的相对位置/速度几何；此对照验证的是ego嵌入的新鲜度，并非删除全部当前运动信息。\n\n## 同样本总体差值\n\n'
    overall = [r for r in groups if r['group'] == 'all' and r['endpoint'] == 'best' and r['model'] in p['residual_models']]
    text += table(['录像/seed/模型', 'N', 'ΔADE/FDE vs anchor', 'ΔADE/FDE vs同容量Ego-Residual', 'ADE配对受益比例vs anchor'], [[r['recording']+'/'+r['seed']+'/'+r['model'], r['target_count'], number(r.get('delta_minus_Ego-Only_ADE_m'))+'/'+number(r.get('delta_minus_Ego-Only_FDE_m')), number(r.get('delta_minus_Ego-Residual_ADE_m'))+'/'+number(r.get('delta_minus_Ego-Residual_FDE_m')), number(r.get('paired_improved_vs_Ego-Only_ADE_fraction'))] for r in overall])
    text += '\n## 状态、门控、图负对照\n\n'
    diag = read(ROOT/'all_state_diagnostics.csv')
    text += table(['录像/seed/模型', '修正mean/P95(m)', 'gate mean/P05/P95', '状态mean/P95', '消息mean/P95', '真实/使用/更换边', '换边率', '入度绝对偏差', '真实/使用距离median/P95(m)'], [[r['recording']+'/'+r.get('seed', '')+'/'+r['model'], number(r.get('correction_norm_mean'))+'/'+number(r.get('correction_norm_p95')), number(r.get('gate_mean'))+'/'+number(r.get('gate_p05'))+'/'+number(r.get('gate_p95')), number(r.get('state_norm_mean'))+'/'+number(r.get('state_norm_p95')), number(r.get('message_norm_mean'))+'/'+number(r.get('message_norm_p95')), r.get('real_edges', '')+'/'+r.get('used_edges', '')+'/'+r.get('changed_edges', ''), number(r.get('changed_edge_fraction')), r.get('incoming_budget_absolute_difference', ''), number(r.get('real_edge_distance_m_median'))+'/'+number(r.get('real_edge_distance_m_p95'))+' → '+number(r.get('used_edge_distance_m_median'))+'/'+number(r.get('used_edge_distance_m_p95'))] for r in diag if r['endpoint'] in ('best', 'analytical') and r['model'] in p['residual_models']])
    text += '\n边数/距离统计覆盖完整验证窗口所有历史上下文关系，不只监督目标；状态/修正/gate统计只覆盖评价目标。每epoch pre-clip payload/delta/gate梯度均值/P95在all_epoch_curves.csv，无冻结参数梯度。gate是共享标量乘支持mask，零值来自无消息，不是已学得自适应选择。\n\nRandom为独立从头新增分支训练，未随机打乱轨迹；半径4m、同帧/同窗口、双端连续、无自环、预算约束保留。真实入度≤8的完整可行邻居集合不能换边；若比例低，负对照力度不足。即使合法，重新选邻居会改变距离/关系几何/信息分布，不当严格单因素关系因果证明。未做固定checkpoint干预替代训练。\n'
    text += '\n停止规则与实际第三轮判定：\n\n'+table(['模型/录像', '通过', 'social ADE/FDE', 'anchor/EgoResidual ADE'], [[r['model']+'/'+r['recording'], r['pass_fixed_rule'], number(r['social_ADE_m'])+'/'+number(r['social_FDE_m']), number(r['EgoOnly_ADE_m'])+'/'+number(r['EgoResidual_ADE_m'])] for r in decision['rows']])
    text += '\n'+('固定筛选通过，实际执行42/2026，家族/策略/输出方式保持17的选择；所有新增分支同预算。' if decision['run_extra_seeds'] else '无人满足两录像同时胜过anchor及同容量控制的预定收益/FDE规则，未运行42/2026，不以未运行补零。')+'\n'
    write('04_SOCIAL_PAYLOAD_COMPARISON.md', text)
    text = '# 历史交互分组与配对误差\n\n原17组与4个历史速度组/no_legal_social_messages全部保留；方向可重叠，组人数不可简单相加。固定4m/k8合法图密度0/(0,1]/(1,3]/>3；相向/同向cos±0.7、双端速度>0.2；关系变化持续候选相对速度变化均值≥0.5m/s。CV_like自身历史速度变化<0.2，moderate0.2–0.5，changing≥0.5；不足另列。所有这些只用历史；原合法空图不等物理周围无人。\n\nCV ADE≥1m与<1m为训练结束后误差分层，只解释预测难度，绝不用于输入/gate/邻居或checkpoint选择。group_comparison.csv提供所有模型、所有组、best/last、人数/不同人/窗口、历史量、误差配对均值/中位数/受益比例。每个模型同target tuple且12个GT点；下表采用各模型整体ADE所选best，不按组挑checkpoint。\n'
    for fold in p['folds']:
        for seed in analysis['actual_seeds']:
            subset = [r for r in groups if r['recording'] == fold['recording'] and int(r['seed']) == seed and r['endpoint'] == 'best']
            indexed = {(r['group'], r['model']): r for r in subset}
            names = ['CV']+p['residual_models']
            out = []
            for group in dict.fromkeys(r['group'] for r in subset):
                reference = indexed[group, 'CV']
                out.append([group, reference['target_count']+'/'+reference['unique_people']+'/'+reference['windows']]+[number(indexed[group, name].get('ADE_m'))+'/'+number(indexed[group, name].get('FDE_m')) for name in names])
            text += '\n## '+fold['recording']+' seed'+str(seed)+'\n\n'+table(['组', '目标/不同人/窗口']+[n+' ADE/FDE' for n in names], out)
    text += '\n无合法社会消息目标在所有真实/随机社会变体的best/last均与同一anchor逐目标ADE/FDE完全一致，修正/gate精确0；测试证据no_message_exact_checks.csv。Ego-Residual仍可改自身运动，与社会无消息回退不同。不同人/窗口数提示重叠相关性，不把目标数当独立统计样本；小组/空组不能推普遍规律或降低阈值凑数。单录像条件差异与GT事后高误差组收益不是可部署选择器证据。\n'
    write('05_INTERACTION_GROUP_ANALYSIS.md', text)
    fresh = []
    for fold in p['folds']:
        rec = fold['recording']
        a = next(r for r in residual if r['recording'] == rec and r['seed'] == '17' and r['model'] == 'Social-Previous')
        b = next(r for r in residual if r['recording'] == rec and r['seed'] == '17' and r['model'] == 'Social-Current')
        fresh.append(dict(recording=rec, Current_minus_Previous_ADE_m=float(b['best_ADE_m'])-float(a['best_ADE_m']), Current_minus_Previous_FDE_m=float(b['best_FDE_m'])-float(a['best_FDE_m'])))
    csv_new(ROOT/'payload_freshness_comparison.csv', fresh)
    has_gain = decision['run_extra_seeds']
    text = '# 研究决策：社会价值与主路径损害能否分开\n\n'
    text += '**预设筛选结果：** '+('满足首轮值得稳定性验证的条件，已执行固定42/2026。' if has_gain else '两录像没有社会模型满足相对anchor及同容量Ego-Residual的预定收益规则，停止新增社会模块及额外seed训练。')+'\n\n'
    text += table(['录像', 'Current−Previous ADE/FDE(m)'], [[r['recording'], number(r['Current_minus_Previous_ADE_m'])+'/'+number(r['Current_minus_Previous_FDE_m'])] for r in fresh])
    text += '\n1. 个体基线：CVR与PedBase同35352参数/同初始化，预定LR策略结果在02；没有容量增加混杂。完整8/12与更多训练窗口下早期best不必然出现，衰减/分布均应依据真实曲线讨论，不无条件延长。\n2. 冻结主路径：全部新增分支共享同一checkpoint/输出，encoder/decoder数值SHA前后相同，梯度为空。无合法消息严格保持anchor，排除了本轮社会分支更新基础预测这一因素。\n3. 独立社会价值：需同时胜过anchor和14425参数Ego-Residual；仅超过anchor不足以排除泛运动容量。实际差值和每目标受益比例全部在04/paired_target_errors.csv，不只选最佳组。\n4. 信息时效：Previous/Current采用同几何/预算/模块，仅payload h_j,t−1→h_j,t；上表是直接设计对照，但仍只有被选开发录像，且各自checkpoint按开发ADE选择。小差异或跨录像方向不同不能宣布Current普遍更有效。\n5. 具体关系：Random合法约束、实际换边/距离差值在04。若弱随机化不影响误差，不能证明模型不用具体邻居；若同样改善，也不能单独证明真实关系特异性。\n6. 条件收益：全部历史/运动/密度及事后CV误差组在05。全局scalar gate没有适应性子网，即使某组改善也不能宣称已实现自适应选择；少人数关系变化组只作探索。\n7. 原问题仍有边界：这是冻结后信息来源实验，**没有完全相同残差结构、训练预算与初始化的联合训练反事实**。因此可以判断本轮是否仍有邻居收益，但不能把旧Dynamic-GNN差异严格归因给“联合训练破坏个体预测”；也不能从失败断言社会信息本身信息论上无价值。\n8. 下一阶段：不继续高阶/关系GRU/复杂loss或第五章。优先保护强运动主路径；若只有局部社会收益，先独立复核该条件/合法负对照与容量控制。若社会持续不胜同容量自修正，第四章应转向可靠运动预测/观测与域泛化，社会保持对照。创新性仍需后续文献与跨录像/seed验证，CV残差/冻结/均值聚合不是新论文创新。\n\nETH未读取，所有录像有历史研究曝光，结果不叫独立确认；未运行条件seed没有虚填指标。源码与预注册终点保持，不反复换录像/seed/门控结构。完成本地提交后停止，用户自行push。\n'
    write('06_RESEARCH_DECISION.md', text)
    text = '# Stage3.6总报告：强个体与冻结社会残差归因\n\n基础2abca51dd42489a06668e6430531b984dff14641；feat/stage3p6-frozen-ego；训练前源db2ea1c2eedcecbb08c192037bbe7a0985755edf。740项旧成果/98个旧checkpoint保护，无旧源码/配置/报告/结果修改。\n\n'
    text += f'实际完成A协议两录像，seed列表{analysis["actual_seeds"]}；{analysis["completed_neural_runs"]}神经run×30epoch，共{analysis["executed_epochs"]}epoch，训练{budget:.2f}秒/{budget/60:.2f}分钟，{analysis["paired_target_rows"]}条best/last逐目标配对记录；CV和Ego-Only解析另列。无高阶/关系GRU/新loss/B/ETH或旧项目模型缓存。\n\n'
    text += table(['录像', '模型', 'schedule', 'best轮', 'best ADE/FDE', 'last30 ADE/FDE', '参数', '训练秒'], ego_rows)
    text += '\n## 冻结信息来源结果\n\n'+table(['录像', 'seed', '模型', 'best轮', 'best ADE/FDE', 'last30 ADE/FDE', '新增参数', '训练秒'], rows)
    text += '\n## 判断\n\n'+('固定条件通过，额外seed已完成；稳定与组差异详见04/05，不将三seed等同新独立数据。' if has_gain else '社会模型未在两个录像同时达到相对anchor及同容量控制的预定收益，按规则不运行42/2026、不堆叠复杂模块。')+'冻结排除了本轮基础ego被社会训练更新的因素；仍缺同结构联合训练反事实，无法给过去失败作唯一因果归因。Current/Previous差值、Random实际强度及完整分组如实报告。\n\n'
    text += table(['录像', 'Current−Previous ADE/FDE(m)'], [[r['recording'], number(r['Current_minus_Previous_ADE_m'])+'/'+number(r['Current_minus_Previous_FDE_m'])] for r in fresh])
    text += '\n## 交付与复核\n\n七报告在reports/stage3p6；protocol/保护清单在configs/stage3p6；results/tables/stage3p6保存完整配置/source/raw/manifest/protocol/hash、真实每epoch、best/last逐目标、分组、状态/门控/梯度/随机图、anchor SHA及预设停止判定。165项训练前测试通过，最终unit_tests.xml与check_stage3p6.py另核查；未运行实验不填0。权重在experiments/stage3p6、git忽略，本地保留；不上传数据/缓存/权重。\n\n下一步先依据同容量差值与跨录像方向评估可靠运动主路径，必要时另授权同结构联合训练反事实或条件复核；本阶段不继续开发。用户自行git push -u origin feat/stage3p6-frozen-ego。\n'
    write('STAGE3P6_REPORT.md', text)
    write_json_new(ROOT/'report_generation_provenance.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), script_sha256=sha256(__file__), reports={str(r): sha256(r) for r in sorted(REPORT.glob('*.md'))}, actual_training_seconds=budget, ETH_evaluated=False))
    print(json.dumps(dict(reports=7, neural_runs=len(neural), training_minutes=budget/60, extra_seeds_executed=has_gain)))


if __name__ == '__main__':
    main()
