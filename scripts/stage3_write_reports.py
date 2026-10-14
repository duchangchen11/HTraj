"""Render Stage3 conclusions from completed artifacts; never invent results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json

ROOT=Path('results/tables/stage3');REPORTS=Path('reports/stage3')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'


def write(name,text):
    with (REPORTS/name).open('x',encoding='utf-8') as f:f.write(text)


def main():
    runs=list(csv.DictReader((ROOT/'seed17_comparison.csv').open()));by={r['model']:r for r in runs}
    groups=list(csv.DictReader((ROOT/'all_groups.csv').open()));g={(r['model'],r['group']):r for r in groups if (r['seed']=='17' and r['endpoint']=='best') or r['model']=='CV'}
    cap={r['model']:r for r in csv.DictReader((ROOT/'parameter_capacity_audit.csv').open())}
    selection=json.loads((ROOT/'round2_selection.json').read_text());assert not selection['selected_models'],'Reports implement the preregistered failure exit; adapt only if actual candidates qualify'
    cv=json.loads((ROOT/'CV/summary.json').read_text());names=['PedBase','Dynamic-GNN-RN','Receiver-Fix','Relation-Routing','Routing-Current','Routing-NoUpdate','Routing-Uniform','Routing-Random']
    comparison=table(['模型','名义/扣已知死参数','best epoch','best ADE/配对FDE(m)','last30 ADE/FDE(m)','训练秒（旧冻结单列）'],
        [[n,f"{cap[n]['nominal_parameters']}/{cap[n]['remaining_parameters']}",by[n]['best_epoch'],f"{float(by[n]['best_ADE_m']):.6f}/{float(by[n]['paired_FDE_m']):.6f}",
          f"{float(by[n]['last_ADE_m']):.6f}/{float(by[n]['last_FDE_m']):.6f}",f"{float(by[n]['training_seconds']):.2f}"+('（旧）' if n in ('PedBase','Dynamic-GNN-RN') else '')] for n in names])
    cohort_names=['history_full8','history_1','history_2to4','history_5to7','new_entry_with_neighbors','new_entry_without_neighbors','full8_high_density','full8_low_density','full8_medium_density','full8_no_neighbors']
    cohorts=table(['历史分组','ADE/FDE目标数','CV','PedBase','Dynamic-RN','Receiver-Fix','History-Routing'],
        [[k,f"{g[('PedBase',k)]['ade_count']}/{g[('PedBase',k)]['fde_count']}",*[f"{float(g[(n,k)]['ADE_m']):.4f}/{float(g[(n,k)]['FDE_m']):.4f}" for n in ['CV','PedBase','Dynamic-GNN-RN','Receiver-Fix','Relation-Routing']]] for k in cohort_names])
    curves=[]
    for path in sorted((ROOT/'new').glob('*/seed_*/epochs.csv')):
        for row in csv.DictReader(path.open()):curves.append(dict(model=path.parent.parent.name,**row))
    from src.training.social_diagnostics import csv_new
    csv_new(ROOT/'all_new_epoch_curves.csv',curves)
    write('04_EXPERIMENT_RESULTS.md',f'''# Stage3真实有限规模实验

训练源c8d151c8c53367ef361d23d30569c9ea5ff8cf52。新模型六次seed17从头30epoch，共180epoch/23040 optimizer updates；不是六模型三seed。PedBase和Dynamic-GNN-RN为原Stage2.5完全兼容冻结结果，三seed/best-last共12个checkpoint已核实源码/超参数/manifest/hash/CPU重评；主表只用seed17配对。旧三seed记录不当作新模型稳定性验证。CV无训练/参数，ADE/FDE={cv['ADE_m']:.6f}/{cv['FDE_m']:.6f}m。

## 固定协议与完整主指标

1024训练/96开发，d64、obs8/pred12、Adam0.001（betas0.9/0.999、eps1e-8、weight_decay0）、batch8、clip1、原masked per-track L2、K1米制ADE/FDE。best最小开发ADE/最早严格平局，配对FDE取相同epoch；last30另列，未挑FDE最佳epoch或改终点。全部ADE1202目标×窗口、末帧FDE839、cutoff可见1226。数据支持重叠窗口不能当独立1202人。source/protocol SHA在各run/config；共同Adam细节在shared_training_configuration.json。

{comparison}
全部新模型社会消息结构只含直接源ego，不增加阶数。扣死参数不是函数空间维数，NoUpdate循环weight_hh还失效1728；原run/config的scorer计数不包含该项，权威容量审计为parameter_capacity_audit.csv。Timing为真实训练循环耗时，评价另计total_run_seconds；冻结耗时来自旧run，系统状态可能不同，不能作为严格硬件效率排名。

## 所有要求的历史分组（ADE/FDE，m）

{cohorts}
完整8帧/1帧/2–4/5–7按总可见帧数划分，覆盖全部1202个ADE目标，不删除难例。new_entry按cutoff可见但不连续，本子集恰好与1帧相同；有邻居按Receiver-Fix历史候选资格定义，不使用未来坐标。完整历史密度沿用Stage2.5的平均连续帧入度：无0、低(0,1]、中(1,3]、高>3。不是改成新图密度以改善结果。

每模型/best-last的全部21个分组、各组可见目标数/窗口数/ADE-FDE有效样本数，以及对PedBase和Dynamic-RN差值均在all_groups.csv（含原9个场景/链组）。不只报告改善组。CV不足两个连续点时明确stationary fallback，不用未来速度。

## 主要观察

Receiver-Fix一帧有邻居ADE比PedBase降低{float(g[('PedBase','new_entry_with_neighbors')]['ADE_m'])-float(g[('Receiver-Fix','new_entry_with_neighbors')]['ADE_m']):.6f}m（约0.35%），History降低{float(g[('PedBase','new_entry_with_neighbors')]['ADE_m'])-float(g[('Relation-Routing','new_entry_with_neighbors')]['ADE_m']):.6f}m（约1.27%），远小于预设10%。两者完整8帧相对PedBase略好，但2–4与5–7帧更差，整体ADE均差于两个基础。FDE仅略好于PedBase，均差于Dynamic-RN。

History比Current稍好，但比同名义参数NoUpdate更差；不能据Current结构/容量不同就证明历史记忆贡献。Uniform ADE也略优于History，FDE较差。Random完整结果见主表，约边数/半径/后端混杂见relation_legality.json，不能写成严格同分布证明。

## 状态、权重与来源

每epoch曲线all_new_epoch_curves.csv及各new/model/seed_17/epochs.csv；config/summary包含参数、时间、源commit、manifest/protocol SHA、best/last checkpoint SHA。权重本地experiments/stage3/new/model/seed_17/checkpoints，不进Git。冻结曲线与配置原路径在frozen_baselines.csv。

all_diagnostics.csv及各diagnostics_best/last记录社会消息计数、unknown receiver消息、before/source/message/carry/after范数mean/P95、关系state范数、残差gate P05/P50/P95及修正。epochs还记录projection/关系/encoder/fusion/decoder梯度。按观察分组的gate/范数/alpha熵在cohort_state_diagnostics.csv，不用GT掩码挑低误差目标。

没有ETH指标、正式五场景、模型B、增加高阶或第二轮新训练。没有合格候选而按预设退出，不反复选择seed。独立确认窗口0；详见06_RESEARCH_DECISION.md。
''')
    inference=list(csv.DictReader((ROOT/'fixed_checkpoint_interventions.csv').open()))
    it=table(['模型','best固定推理case','ΔADE(m)','ΔFDE(m)','预测改变均值L2(m)'],[[r['model'],r['case'],f"{float(r['delta_ADE_m']):+.6f}",f"{float(r['delta_FDE_m']):+.6f}",f"{float(r['prediction_change_L2_m']):.6f}"] for r in inference])
    state=list(csv.DictReader((ROOT/'cohort_state_diagnostics.csv').open()));st=table(['模型/分组','可见目标','gate均值/P95','state范数均值/P95','修正范数均值/P95','归一熵均值'],
        [[r['model']+'/'+r['group'],r['observed_targets'],f"{float(r['gate_mean']):.4f}/{float(r['gate_p95']):.4f}",f"{float(r['state_L2_mean']):.3f}/{float(r['state_L2_p95']):.3f}",f"{float(r['correction_L2_mean']):.3f}/{float(r['correction_L2_p95']):.3f}",f"{float(r['normalized_entropy_mean']):.4f}"] for r in state if r['endpoint']=='best' and r['model'] in ('Receiver-Fix','Relation-Routing','Routing-NoUpdate') and r['group'] in ('all','history_full8','new_entry_with_neighbors','new_entry_without_neighbors')])
    h1={r['group']:r for r in csv.DictReader((ROOT/'supplementary_H1/groups_best.csv').open())}
    ht=table(['组','旧H1 best ADE/FDE','RF best ADE/FDE','RF−H1 ADE/FDE'],[[k,f"{float(h1[k]['ADE_m']):.6f}/{float(h1[k]['FDE_m']):.6f}",f"{float(g[('Receiver-Fix',k)]['ADE_m']):.6f}/{float(g[('Receiver-Fix',k)]['FDE_m']):.6f}",f"{float(g[('Receiver-Fix',k)]['ADE_m'])-float(h1[k]['ADE_m']):+.6f}/{float(g[('Receiver-Fix',k)]['FDE_m'])-float(h1[k]['FDE_m']):+.6f}"] for k in ('all','history_full8','history_1','history_2to4','history_5to7','new_entry_with_neighbors')])
    write('05_ABLATION_ANALYSIS.md',f'''# 关系选择贡献：独立训练与固定checkpoint分开

## 从头训练的主要证据

无社会信息：PedBase独立旧训练经完全兼容核查后冻结引用；其名义参数较少，不能单独排除通用修正容量。当前权重：Receiver-Fix（旧alpha）和Routing-Current（新特征MLP）从头训练。历史：Relation-Routing GRU从头训练。删除更新：NoUpdate同GRU每步前状态0，从头训练。Uniform同候选均匀权重从头训练。Random可靠同场景源、约入度匹配、放宽4m，从头训练。六种要求都具备训练/冻结训练控制，不用推理关闭代替重训。

History−NoUpdate best ADE/FDE={float(by['Relation-Routing']['best_ADE_m'])-float(by['Routing-NoUpdate']['best_ADE_m']):+.6f}/{float(by['Relation-Routing']['paired_FDE_m'])-float(by['Routing-NoUpdate']['paired_FDE_m']):+.6f}m；名义参数73355相同，NoUpdate另有1728循环权重死参数。History没有赢得额外循环容量的收益。History−Current={float(by['Relation-Routing']['best_ADE_m'])-float(by['Routing-Current']['best_ADE_m']):+.6f}/{float(by['Relation-Routing']['paired_FDE_m'])-float(by['Routing-Current']['paired_FDE_m']):+.6f}m，但Current编码/参数不同，非独立记忆证明。History−Uniform={float(by['Relation-Routing']['best_ADE_m'])-float(by['Routing-Uniform']['best_ADE_m']):+.6f}/{float(by['Relation-Routing']['paired_FDE_m'])-float(by['Routing-Uniform']['paired_FDE_m']):+.6f}m，ADE/FDE方向分开报告。

History−Random={float(by['Relation-Routing']['best_ADE_m'])-float(by['Routing-Random']['best_ADE_m']):+.6f}/{float(by['Relation-Routing']['paired_FDE_m'])-float(by['Routing-Random']['paired_FDE_m']):+.6f}m。Random实际43529边vs真实43708，差0.410%；替换65.579%、59.900%控制边超4m，CPU/CUDA随机图249条对称差边。它改变空间/连续性分布，不能称严格关系单因素或仅容量对照。即便真实图好也不足独立证明，若随机更好同样如实记录。

## Receiver-Fix单阶原结构对照

旧H1 seed17最佳/最后checkpoint按Stage3分组CPU重评，与旧主指标误差<1e-6m，源码与共同数据/损失/预算未变，记录supplementary_H1/audit.json。不是又训练一次旧H1。

{ht}
接收资格及必需未知速度语义是结构差异。各模型独立训练/按整体ADE分别选epoch，不能把某分组差值解释为所有已知目标的同权重必然改善。无邻居targets的预测也因共享训练参数改变，故有邻居组小改善不能全归于邻居信息。

## 固定best checkpoint辅助干预（CPU、不是独立训练）

{it}
no_messages精确绕过全部social历史；uniform改权重；reset_relation每帧置0边先前状态，Receiver-Fix该case是明确no-op；scene_random放宽距离规则。比较各自CPU full，预测变化仅用历史cutoff可见目标，ADE/FDE只用于事后评价，不用于选邻居或确认集调参。OOD/共适应效应不能替代上方从头训练结果。

## 状态与门控按历史分组

{st}
范数为学习特征单位，不是速度或米。熵是cutoff alpha熵除log(degree)，degree≤1记0，所以跨组比较受degree/无邻居比例影响；Uniform有多邻居时等于1。全部21组/best-last/六模型保留252行诊断，未过滤高误差targets。无邻居新进入者support/gate/correction为0，精确回到训练模型自己的ego，不意味着等于另一个PedBase训练权重的预测。

人工测试在current几何、current速度和ego全部固定时，改变较早本对关系历史会改变History路由，而NoUpdate不变；证明计算依赖结构。已训练checkpoint敏感性与梯度说明模型计算使用消息/记忆的程度；从头消融却没有建立稳定精度优势。不能把非零梯度、非零gate、注意力模式或删消息退化直接写成新机制有效。

有效容量未严格排除、Random有分布偏移、仅一seed且22个一帧有邻居目标。当前结论是所测原型没有独立有效选择收益，不是证明所有关系信息理论上无用。
''')
    decision=table(['候选','整体1%门槛','完整8安全','短历史10%门槛','整体安全','记忆0.5%门槛','入选'],[[r['model'],r['primary_ADE_1pct'],r['full8_safe'],r['targeted_ADE_10pct'],r['overall_safe'],r['memory_vs_NoUpdate_0p5pct'],r['qualified']] for r in selection['decisions']])
    write('06_RESEARCH_DECISION.md',f'''# Stage3研究决定与停止边界

第一轮六个新模型全部完成后，按训练前configs/stage3/protocol.json和00_PREREGISTRATION.md执行筛选。候选为主要网络Receiver-Fix/Relation-Routing；Current/NoUpdate/Uniform/Random是归因对照，不扩充搜索候选以挑指标。资格及具体布尔结果见round2_selection.json。

{decision}
**没有候选入选，停止第二轮。** 不训练新模型seed42/2026，不重抽seed，不改筛选阈值，不延长epoch、不启动B/高阶/完整五场景/ETH。旧基础三个seed冻结记录不能当作新模型跨seed稳定性证据。

## 用户十个问题的回答

1. 连续性是否阻断有效消息？是：开发27个一帧目标22个有可靠邻居，65条cutoff关系被阻断；训练314/384个和1485条。结构测试已证明修复能接收；有效接收不等于预测收益。
2. RF是否改善短历史？一帧有邻居相对PedBase仅0.35% ADE改善、相对旧H1见05；2–4/5–7帧退化。没有达预设10%目标，不宣称可靠改善。
3. 是否使完整历史退化？seed17整体完整8组比PedBase略好，仍未超越Dynamic-RN；其他密度/ADE/FDE方向见04，不能写无任何组退化。
4. 历史关系比逐帧有效？比Current稍好，但同名义参数NoUpdate更好，且Uniform主ADE更好。独立记忆收益未建立。
5. 新模型比PedBase好吗？RF与History整体ADE更差，FDE略好；未建立整体优势。
6. 比Dynamic-RN好吗？两个主要候选的整体best ADE/FDE都更差。
7. 跨seed稳定吗？未验证。无第一轮候选达到预设值得复核门槛，按规则没有第二轮；不把一次失败推断为所有seed失败。
8. 收益来自具体关系吗？人工和固定checkpoint证明存在计算依赖，但独立选择/记忆收益不足；近参数对照也没有支持。Random的空间/预算混杂和不完美有效容量匹配不允许因果强断言。
9. 与文献结构有区别吗？有：本对因果记忆只调alpha，直接source ego消息、接收/carry分离、原ego残差；不复制UniEdge patch双图/Hodge/Transformer。更近的Social Attention已研究边RNN+注意力，结构不同不代表已有概念新颖。来源和边界见03。
10. 值得作为第四章正式网络吗？当前不值得把History-Routing作为已验证正式贡献；保留接收契约修复和可复核失败原型。第四章不预写正向结论。

## 确认性验证不可构造

原fold_ETH中非测试验证548窗口，开发96被多轮使用。按完整20帧支持、同recording、预先8秒隔离核查，候选0；即使取消隔离，448个剩余窗口也全有支持重叠。confirmation_manifest.json逐窗口审计、check_stage3.py直接验证。没有确认指标，没有将未选窗口称为独立验证，更没有开启ETH。未来应准备新的时间隔离验证/额外非测试记录或预留另一套划分，且在方案冻结后一次性确认；此前开发报告仍保留重复使用限制。

## 失败后替代方向（本阶段未实现）

停止继续叠加pair记忆、高阶或新模块。优先重新界定可辨识问题：单个历史点没有自身方向，局部邻居也不一定代表其意图。可在下一次授权后单独测试“可靠源当前可计算运动先验”与延迟h_prev payload，明确缺失/不确定性、保留ego、同容量泛运动修正对照；只改消息内容/新鲜度，不同时加GRU、改融合或删难例。

若独立确认和多seed下仍无至少约1%整体ADE（FDE安全）或明确短历史收益，停止将当前局部社会路由当第四章主要精度贡献，转向观测完整性、运动不确定性/可辨识性或可核查的场景约束问题。新的数据或任务必须先确认可用性，不在本阶段直接开发第五章。接收合法性修复可以保留为工程组件，不能用其正确性代替科研效果。
''')
    write('STAGE3_REPORT.md',f'''# Stage3总报告：接收条件修复与历史关系路由原型

项目/home/lrj/HTraj；分支feat/stage3-relation-routing，基础2d786785813ee298af1cb148999e13a21ffab0c6。保护440项旧源码/配置/报告/结果/测试/文献和权重（其中78个checkpoint），未覆盖旧A0/A-R/H1/H2/H3。新模型与实验独立目录，仅本地提交、不自动push。

## 已执行与主结果

六个新社会变体seed17独立30epoch，180epoch/23040更新；1024/96原manifest、原loss/Adam/batch/clip、原ego encoder/残差/decoder；无额外阶数。冻结PedBase/Dynamic-RN的三seed best/last完全审计，主表仅配对seed17；CV历史速度不足两点时stationary回退，CV ADE/FDE={cv['ADE_m']:.6f}/{cv['FDE_m']:.6f}m。

{comparison}
整体1202 ADE/839 FDE目标，全部21历史/密度/链分组和差值保留。140项预训练测试通过后开始训练，最终142项完整测试通过；人工验证新target接收、carry reset、缺失源禁发、不伪造速度、ego精确回退、GT隔离、padding/置换、pair前缀依赖及NoUpdate失效循环权重。训练源c8d151c8c53367ef361d23d30569c9ea5ff8cf52；所有权重保持本地ignored，config/summary保存best/last SHA。

## 归因判断

开发27个一帧目标22个有可靠邻居、65条潜在接收边被旧条件阻断；其中36条邻居关系完整8帧（55.38%），22目标全都有至少一个已知运动的上一帧payload。RF解决结构阻断，但短历史有邻居ADE改善仅约0.35%；History约1.27%，均远低预设10%。两个主要候选完整8相对PedBase略好，2–4/5–7更差，整体ADE不及两个基础，FDE仅略好PedBase而弱于Dynamic。

历史pair平均活跃4.778帧，旧alpha已有运动输入/ego历史；历史重复提供动机，不证明信息价值。History虽稍优于当前小MLP，却弱于同名义参数NoUpdate，Uniform主ADE也更好。人工依赖、非零梯度、固定checkpoint删消息或改alpha不能代替独立训练收益。NoUpdate额外1728循环权重失效；初始run死参数字段只数scorer，完整容量审计另表，不声称完全相同有效容量。

真实/随机关系从头训练对照保持同场景/可靠源/无自环重复，实际边预算少0.41%、约59.90%超4m，存在分布偏移；不能强称关系因果验证。另保存10条明确CPU固定checkpoint辅助干预及252条历史分组状态/门控/熵诊断。

## 研究决定

全部首轮结束后按预提交门槛筛选，0个候选合格，停止第二轮，不盲跑42/2026、不选幸运seed。新模型跨seed稳定性**未验证**。原548验证窗口余下448全与开发96支持重叠，隔离确认候选0，不能称独立验证，ETH继续封存。

当前History-Routing不建议作为第四章正式性能贡献；保留Receiver-Fix契约与失败原型。UniEdge结构与此不同，但Social Attention（ICRA2018）已使用边RNN历史与注意力，不能将新增GRU称创新。下一阶段如获授权，优先单独核查当前可靠源运动先验/消息新鲜度和观测不确定性，再用独立数据确认；不继续无限叠模块，未启动B/五场景/第五章。

## 交付与溯源

七份报告：01_PROBLEM_ANALYSIS.md、02_RECEIVER_FIX_DESIGN.md、03_RELATION_ROUTING_DESIGN.md、04_EXPERIMENT_RESULTS.md、05_ABLATION_ANALYSIS.md、06_RESEARCH_DECISION.md、本报告；00_PREREGISTRATION.md保存训练前规则。结果results/tables/stage3/包含逐run配置/30行曲线/分组/状态/summary、汇总比较、完整容量、负对照、manifest/源码/protocol SHA、pretraining_tests.xml/unit_tests.xml和确认划分。

data manifest SHA c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced；subset SHA 4b4a50ce47b8d8a2fc574a814f80982b602fed237939f7d7f019492be6dc4be1；protocol SHA 8d84a0c8ad96574f95e2a03db8ca5c315bb91d29218be4e4fbaec4d286fcee10。冻结源90d987799d06bef4e213952342da231e93792a41。各分析另存运行commit与脚本SHA。

用户自行git push -u origin feat/stage3-relation-routing。阶段工作结束，只完成本地提交并停止。
''')


if __name__=='__main__':main()
