"""Generate completed Stage3.5 reports from real saved metrics, not estimates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,subprocess
import numpy as np
from src.training.social_diagnostics import csv_new
from src.data.preprocessing import sha256,write_json_new

R=Path('results/tables/stage3p5');D=Path('reports/stage3p5')


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'
def val(x):return f'{float(x):.6f}' if x not in ('',None) else '—'
def write(name,text):
    with (D/name).open('x',encoding='utf-8') as f:f.write(text)


def main():
    p=json.loads(Path('configs/stage3p5/protocol.json').read_text());runs=[];cv=[];curves=[];budget=[]
    for view in p['protocols']:
        cv.append(json.loads((R/'pilot'/view/'CV/summary.json').read_text()))
        for n in ('PedBase','Dynamic-GNN-RN'):
            path=R/'pilot'/view/n;s=json.loads((path/'summary.json').read_text());assert s['status']=='completed';runs.append(s)
            epochs=list(csv.DictReader((path/'epochs.csv').open()));curves.extend(dict(model=n,protocol_name=view,**r) for r in epochs)
            loss=[float(r['train_loss']) for r in epochs];s['tail5_train_loss_relative_decrease']=(loss[-5]-loss[-1])/abs(loss[-5]);s['last_minus_best_ADE_m']=s['last_ADE_m']-s['best_ADE_m']
            s['first_train_loss']=loss[0];s['last_train_loss']=loss[-1]
            for epochs_planned in (30,60,100):
                all_n=2471 if view.startswith('A_') else 2778
                budget.append(dict(protocol=view,model=n,training_windows=all_n,epochs=epochs_planned,estimated_training_seconds=s['training_seconds']*all_n/s['training_windows']*epochs_planned/30,
                    estimate_not_execution=True,basis='measured pilot per-window rate, same d64/Adam; no validation/IO, composition/epoch changes unmeasured'))
    csv_new(R/'pilot_neural_summary.csv',runs);csv_new(R/'pilot_all_epoch_curves.csv',curves);csv_new(R/'updated_full_budget_estimate.csv',budget)
    groups=list(csv.DictReader((R/'pilot_group_comparison.csv').open()));legacy=list(csv.DictReader((R/'legacy_seed_direction_consistency.csv').open()))
    summary=table(['协议','模型','参数','best epoch','best ADE/FDE(m)','last30 ADE/FDE(m)','训练秒'],
        [[c['protocol_name'],'CV',0,'解析',val(c['ADE_m'])+'/'+val(c['FDE_m']),'不训练',0] for c in cv]+
        [[s['protocol_name'],s['model'],s['parameter_count'],s['best_epoch'],val(s['best_ADE_m'])+'/'+val(s['best_FDE_m']),val(s['last_ADE_m'])+'/'+val(s['last_FDE_m']),f"{s['training_seconds']:.2f}"] for s in runs])
    rows=table(['协议/历史组','ADE/FDE人数','CV ADE/FDE','PedBase ADE/FDE','Dynamic ADE/FDE','Dynamic−Ped ADE/FDE','配对ADE改善比例'],
        [[r['setting']+'/'+r['group'],r['ADE_count']+'/'+r['FDE_count'],val(r.get('CV_ADE_m'))+'/'+val(r.get('CV_FDE_m')),val(r.get('PedBase_ADE_m'))+'/'+val(r.get('PedBase_FDE_m')),
          val(r.get('Dynamic_ADE_m'))+'/'+val(r.get('Dynamic_FDE_m')),val(r.get('delta_Dynamic_minus_PedBase_ADE_m'))+'/'+val(r.get('delta_Dynamic_minus_PedBase_FDE_m')),
          val(r.get('paired_ADE_improvement_fraction'))] for r in groups if r['endpoint']=='best'])
    leg=table(['旧B开发组','ADE/FDE人数','三seed平均ΔADE(Dyn−Ped)','ADE更好seed数','FDE更好seed数'],
        [[r['group'],r['ADE_targets']+'/'+r['FDE_targets'],val(r['mean_delta_Dynamic_minus_PedBase_ADE_m']),r['ADE_direction_better_seeds']+'/3',r['FDE_direction_better_seeds']+'/3'] for r in legacy])
    inter=list(csv.DictReader((R/'pilot_fixed_graph_interventions.csv').open()))
    it=table(['协议/case','ΔADE','ΔFDE','变动边比例'],[[r['protocol']+'/'+r['case'],val(r['delta_ADE_m']),val(r['delta_FDE_m']),val(r['changed_or_removed_fraction'])] for r in inter])
    pairs=list(csv.DictReader((R/'cross_protocol_common_targets.csv').open()));common=[]
    for name in ('CV','PedBase','Dynamic-GNN-RN'):
        subset=[r for r in pairs if r['model']==name];common.append(dict(model=name,targets=len(subset),A_ADE_m=np.mean([float(r['A_ADE_m']) for r in subset]),B_ADE_m=np.mean([float(r['B_ADE_m']) for r in subset]),
            A_FDE_m=np.mean([float(r['A_FDE_m']) for r in subset]),B_FDE_m=np.mean([float(r['B_FDE_m']) for r in subset])))
    csv_new(R/'cross_protocol_common_target_summary.csv',common)
    ct=table(['同706个完整目标','A ADE/FDE','B ADE/FDE'],[[r['model'],val(r['A_ADE_m'])+'/'+val(r['A_FDE_m']),val(r['B_ADE_m'])+'/'+val(r['B_FDE_m'])] for r in common])
    write('02_BASELINE_COMPARISON.md',f'''# 两套协议的真实基础对照

训练源5771f31df0ae863daf70307a9f9b6c73f232ee68，students003原验证区70窗口，其它四recording原训练区；所有新神经模型seed17重新初始化，没有旧权重迁移。A2058、B2365有效训练窗口，不是再跑原1024/96。Adam0.001/batch8/clip1/d64/原per-track L2/30epoch统一；每协议自己的开发ADE选best，同epoch配对FDE与last30都保存。

{summary}
A评价706个完整12步目标，B1464个有效future目标/FDE986个，二者平均值**不能直接横比模型优劣**。参数PedBase35352、Dynamic102235（其中单channel scorer8321已知不活跃），容量未匹配，GNN优于Base也不自动证明独立社会价值。

CV用最后两个连续历史位置差分，缺失速度明确stationary回退。米坐标/0.4秒一致。所有神经run/config、30行epochs、best-last targets/groups/summary与本地ignored权重SHA保存。GPU index_add浮点不逐位稳定，无重新挑seed/epoch口径。

## 同一完整目标的人群控制

{ct}
用tuple(recording,sample_id,agent_id)对齐A与B，共同完整目标每人future_points=12。CV相同输入自身完整历史，结果一致；其它模型分别训练、上下文政策也不同，此比较排除评价人群差异但仍不隔离训练目标/上下文差异，不称A协议独立提升预测。原B全部难例继续作为主结果，未为提升指标删掉部分历史/未来。

## 外部公开参考

Social-STGCNN正式名称、MIT许可、固定源码/数据loader、高斯输出/20样本评价已核查（01_PROTOCOL_AUDIT.md、external_reference_sources.json及supplement/limitations）。官方processed逐文件split因API限流未完整核对；本阶段**没有训练、评价、官方权重运行或公平比较**。它的SGD250epoch、完整20筛选/至少2人/全帧列表与K20输出均不同，不能以这里数字代替其官方结果。

## 旧结果严格分开

B_legacy_1024_96_frozen_analysis目录是旧PedBase/Dynamic三个seed最佳权重的兼容重评：源码逐字节、manifest/subset/checkpoint SHA及主指标<1e-6m核查通过。0新增训练。它只提供旧开发组的seed方向线索，不进入上表，也不冒充新录像隔离实验。
''')
    write('03_INTERACTION_VALUE_ANALYSIS.md',f'''# 社会交互价值：历史分类与同样本配对

全部分类在预测前仅用observed/mask/timestamps计算，算法/阈值已训练前冻结（00计划和protocol.json）。标签GT只评价误差，未来可用性只用于协议目标资格；没有用未来运动定义相向、交叉或关系变化。运动类别可重叠，人数不能相加当总人数。“关系稳定”限定为持续候选对相对速度变化均值<0.5m/s，不等于距离/拓扑不变；没有有效持续pair另列。每组不管改善或退化均保留，空组显示0。

## 新录像隔离pilot best（所有组，m）

{rows}
每组记录同一批目标的ADE/FDE、观察点数量/可计算motion步数、邻居密度、GT有效标签点数和唯一行人数，详见pilot_group_comparison.csv与各targets.csv。Dynamic−Ped负值为改善，配对ADE改善比例是样本比例，非显著性；平均改善可能集中在少数难例。4340条best/last配对保留，不能用不同模型人群平均之差伪称同样本改善。

## 旧B重用开发集的三seed方向（与新pilot分开）

{leg}
ZARA2录像两项方向3/3改善，平均ADE收益仅0.00354m，是值得新划分复核的弱线索；UNIV students001有较大平均收益但ADE只有2/3。交叉运动FDE3/3更好、ADE2/3；同向FDE3/3但ADE1/3，不能笼统称稳定改善。全体ADE1/3更好。无邻居/2–4帧/关系运动变化组均3/3 ADE和FDE更差，后者仅16个目标×窗口，不足泛化结论。

无邻居已有融合fallback仍可能差于独立PedBase：两模型的ego encoder/decoder接受不同联合训练，fallback到GNN自己的ego并不恢复PedBase权重。不能把无边组差异当作收到无效边；它暴露主路径共适应/容量与优化混杂。

## 固定checkpoint辅助消息依赖（不是从头消融）

{it}
no_messages删图使社会support为空并回到该GNN自己训练的ego；radius_random保持同场景/两端连续/4m/无自环/同入度，来源为已有合法图干预代码，记录实际换边比例。随机化力度有限时，无明显变化不证明不用关系。删除/随机介入可能OOD或共适应，不能单独证明独立预测收益；缺少新fold从头近容量无社会控制。

## 重叠与“稳定”限制

原70验证窗口高度时间重叠。metadata-only完整20帧支持再加8秒隔离的抽稀保存在nonoverlap_sensitivity.json；对应最后一组单列，样本很少，**仍不是独立确认**。同一人多个窗口相关，不能将706/1464当独立行人，不能用普通目标bootstrap声称精确置信区间。一个recording一个seed只能提供可行性/条件差异，稳定性需预先固定其它recording/seed复核；旧三个优化seed也不是三份新数据。

当前不能证明普遍或独立社会收益；可以识别“哪些条件值得复核”以及“哪些已有代价需要保护ego”。不因论文需要只保留改善组。
''')
    convergence=table(['协议/模型','best epoch','train首/末loss','尾5loss相对下降','last−best ADE','peakMiB'],[[s['protocol_name']+'/'+s['model'],s['best_epoch'],val(s['first_train_loss'])+'/'+val(s['last_train_loss']),val(s['tail5_train_loss_relative_decrease']),val(s['last_minus_best_ADE_m']),val(s['peak_cuda_allocated_MiB'])] for s in runs])
    total=sum(s['training_seconds'] for s in runs)
    write('04_TRAINING_FEASIBILITY.md',f'''# 训练收敛、可用目标与RTX3080预算

00_PRETRAINING_PLAN.md与pretraining_budget_estimate.csv在训练前提交：根据旧真实RTX3080时间按窗口数线性估计首轮约26分钟训练，随后实际执行四个run；不是先跑结果再写估算。

## 旧30epoch不是收敛证明

审计43个Stage2.5/2.6/3真实30epoch曲线，6个best在末epoch，37个last ADE弱于各自best；4个尾5训练loss相对下降仍≥2%。其余尾部趋缓也可能是优化饱和/欠拟合/过拟合，不能据tail<2%断言全局收敛。固定预算避免挑epoch/seed，尚缺独立训练量/学习率/长预算控制。

## 新pilot实际曲线与成本

{convergence}
合计实际训练{total:.2f}秒（{total/60:.2f}分钟），评价/加载另计各summary.total_seconds。RTX3080 10GiB上实测峰值列于表，当前d64基线内存可行；群体人数和稠密图成本会随split变化，不能按参数量单独估时。

A train2058完整窗口/19034目标/228408点，B2365/31458 ADE、23423 FDE目标/328649点；新30epoch分别7740和8880更新，比旧1024的3840多。best早于last的真实表现表明加训练步数并不保证录像域外改善。不能把新旧验证平均差当训练量的单因素结论，二者人群/recording都变了。

## 全部原训练窗口可否使用

原2781都可作为已定义训练pool；A只有2471窗口含合法完整目标（28022目标/336264点），B2778窗口有效（46287 ADE、34575 FDE/484422点）。无合法标签窗口不虚计为训练目标。**如果在留一recording开发中使用所有2781，会重新把验证录像混入训练，破坏录像隔离**；本次必须舍弃students003的413原训练窗口。其它原548验证窗口仍不能未经重新定义就进入训练。

根据实测每窗口时间更新全pool30/60/100预算：updated_full_budget_estimate.csv。它是将来协议合法调整后的数学估算，未执行全pool/60/100；不在当前LORO实际使用被舍弃录像。未来五recording×三seed×两协议×两神经模型是60个run，按本次粗估数小时（约{total/3600*15:.2f}小时训练），并非立即启动；validation/IO/群体差异和正式场景级协议另核查。公开Social-STGCNN额外环境/SGD250预算没有实测，不给假耗时。

下一步预算应先固定新开发fold/近容量控制/更强motion基线和选择规则，再做少量长预算诊断；当前不能直接把30epoch写成“所有网络均充分收敛”。ETH不参与选择。
''')
    a=[s for s in runs if s['protocol_name'].startswith('A_')];b=[s for s in runs if s['protocol_name'].startswith('B_')]
    delta={}
    for label,rs in [('A',a),('B',b)]:
        by={s['model']:s for s in rs};delta[label]=[by['Dynamic-GNN-RN']['best_ADE_m']-by['PedBase']['best_ADE_m'],by['Dynamic-GNN-RN']['best_FDE_m']-by['PedBase']['best_FDE_m']]
    write('05_CHAPTER4_RESEARCH_DECISION.md',f'''# 第四章问题选择：证据、停止条件与下一步

## 十个研究判断

1. 当前任务有明显社会收益吗？旧B全体平均GNN ADE略差、只有1/3seed更好。新recording pilot A的GNN−Ped ADE/FDE={delta['A'][0]:+.6f}/{delta['A'][1]:+.6f}m，B为{delta['B'][0]:+.6f}/{delta['B'][1]:+.6f}m；不能证明普遍或独立收益。参数/共同主路径学习尚混杂。
2. 哪些场景收益最大？旧开发UNIV students001平均ADE−0.00766m，ZARA2−0.00354m且两指标3/3方向一致；它们是旧窗口上的候选证据，不是新recording复核。首轮只验证students003，不能把其中一个录像结果推广到整个UNIV。
3. 哪些场景个体已足够？旧HOTEL GNN两指标3/3退化、无邻居也3/3退化，故当前运行选择更支持PedBase。不能证明信息理论上不需要社会。新A的完整运动输入下强PedBase与CV比较见02，没有稳定证据要求更复杂社会模块。
4. 小规模限制评价吗？是重要限制：旧1024只覆盖原2781中的部分且窗口重叠、验证96反复选择；但新更多窗口/更新并未自动保证更好。训练量效果尚缺同一新验证/同一目标的1024-vs-full独立对照，不能定量归因。
5. 标准完整任务更值得研究交互吗？它更适合作为与公开研究接轨的主任务；本次A并未证明GNN优势，因此不能说完整轨迹必然带来社会收益。完整协议、官方split/K指标与其它录像复核优先于结构堆叠。
6. 部分观测是否保留？保留B作为独立鲁棒性任务，并明确partial-future ADE/FDE不同分母。后续可另定义完整future标签的观测缺失鲁棒性实验以隔离输入不足，必须新协议命名并保留原B主结果，不能偷删难例。
7. 第四章继续Dynamic-GNN吗？保留为标准社会控制和诊断工具，不直接当已验证正式创新模型。也不回到增加TOHI阶数。
8. 自适应协同值得研究吗？是有条件的待验证假设：少数录像/交叉FDE受益，而部分历史/无邻居/部分关系变化受损，提示选择社会修正及保护独立运动主路径。不能只加一个gate就宣布创新。
9. 哪些问题有实验依据？完整与部分future人群/分母差异；录像隔离后的早期best与后期退化；GNN自己ego共适应导致无边组劣于独立Base；收益依场景/历史信息而变。edge-memory与高阶收益没有支持，停止继续扩展。
10. 下一阶段实现什么？优先**强运动学主路径（CV先验/轻量残差）与可撤回社会修正的分阶段验证**：先固定/冻结强ego，再单独训练小型社会残差，门控仅依历史置信度/邻居可靠性，无消息时严格保持同一anchor预测；对照联合训练、同有效容量泛运动修正及合法关系重训。已有残差/门控/kinematic思想须归属已有工作，未确定新颖性，当前不实现。

## 实验先行与停止条件

下一次授权后先在预定多个非ETH recording的录像隔离协议A上复核seed17基线，检查早期best/长预算/强CV残差能力，再决定三个seed与公开模型复现。没有按历史定义的适用组稳定收益，就停止把当前local社会GNN当第四章主精度贡献；优先观测不确定性/泛化与运动先验问题。近容量+关系重训+独立开发不足时不宣称“社会独立贡献”。

本阶段没有新复杂网络、方案B、ETH指标、正式五场景、其它四fold训练或新模型三seed；其它fold只有真实数据统计。原数据都被研究用过，recording隔离开发不能改名为全新独立确认，需额外未用记录或另有数据。保持失败与空组，下一步研究方案等待用户指令。
''')
    write('STAGE3P5_REPORT.md',f'''# Stage3.5：数据协议核查与社会交互价值总报告

项目/home/lrj/HTraj；分支feat/stage3p5-benchmark-audit；基础46ecb91169eeadb5ba7ec28b7c648dd0ace837c6。606项Stage1–3源码/脚本/配置/报告/结果/权重SHA保护，90个旧checkpoint未改变；未使用旧项目自定义模型或缓存，只本地提交。

## 实际完成

明确A_complete8x12_history_context完整8/12任务与B_partial_mask_v1原部分观测任务；world米/0.4s/K1，不混分母。标准公开Social-STGCNN/SGAN采样/筛选/划分及MIT代码已核查；固定源码与作者稿SHA保留，外部模型**未复现**、没有虚填其结果或公平排名。processed逐文件split仍未完整核对。

五个非ETH录像隔离候选都有真实窗口/目标/历史/密度/运动统计；首轮只students003原val70窗口，train仅其它四recording原train，held原train413舍弃，原val未加入训练。A2058训练窗口/19034目标→706验证目标；B2365/31458 ADE、23423 FDE训练目标→1464 ADE/986 FDE验证目标。与旧96有11个相同窗口且旧研究看过这些录像，所以不是独立确认。

四个神经baseline seed17从头30epoch/120epoch，加两套CV解析评价；训练源5771f31df0ae863daf70307a9f9b6c73f232ee68，旧B三seed只完全兼容冻结重评，不算新训练。148项测试训练前通过，最终测试记录在unit_tests.xml。

{summary}
源manifest SHA a7698e867ac7b7163dee035382eab0ebccb94a733e51955c4e3461d3ea5a3cf5；旧manifest SHA c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced；protocol SHA 10fefd17291f655b8a004f0560cce1feb9b7f0a4803315f9a3f26aa046b3558b。各config含源码/data/index SHA、参数、checkpoint SHA与环境。

## 归因与第四章判断

新pilot全部历史分组和同target误差配对保存，不只报改善。A GNN−Ped ADE/FDE={delta['A'][0]:+.6f}/{delta['A'][1]:+.6f}m，B={delta['B'][0]:+.6f}/{delta['B'][1]:+.6f}m；单recording/seed，且容量不匹配，不能证明普遍独立社会价值。旧B ZARA2两指标3/3方向受益，交叉FDE3/3但ADE2/3；无邻居、2–4帧、变化关系组3/3退化。此为待新开发复核假设，非独立验证。

模型无边时回到自己的ego，却不等独立PedBase权重，主路径共同优化是关键混杂；固定checkpoint删消息/合法随机图只是依赖证据。标准A应保留作主研究任务，B保留作独立鲁棒设置；暂不开发复杂社会结构，建议下一步先验证强运动先验与冻结主路径的小型、可撤回社会修正，已有思想不改名宣称创新。

旧30epoch与新更多训练步数都不证明收敛，早期best/后续val退化真实保留。首轮实际训练{total/60:.2f}分钟；全pool/60/100/多个fold预算仅估算，RTX3080实测基线内存可行。没有ETH调参、其它fold训练、独立确认或公开模型训练。

## 文件

reports/stage3p5的01_PROTOCOL_AUDIT.md、02_BASELINE_COMPARISON.md、03_INTERACTION_VALUE_ANALYSIS.md、04_TRAINING_FEASIBILITY.md、05_CHAPTER4_RESEARCH_DECISION.md及本报告；00_PRETRAINING_PLAN.md保留预提交设计。results/tables/stage3p5保存所有真实配置、split/group counts、30epoch曲线、best-last逐目标误差与分组、配对/共目标对比、历史三seed审计、图介入、hash、测试、预算及保护清单。未运行实验明确为估算/未复现，不填假结果。

用户自行git push -u origin feat/stage3p5-benchmark-audit；完成本地commit后停止，等待下一指令。
''')
    write_json_new(R/'report_generation_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),fresh_neural_runs=4,epochs=120,ETH_evaluated=False))


if __name__=='__main__':main()
