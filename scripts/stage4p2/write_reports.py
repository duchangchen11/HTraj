"""Render Stage4.2 reports only from completed real artifacts."""
import sys,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
ROOT=Path('results/tables/stage4p2');REPORT=Path('reports/stage4p2');FOLDS=('crowds_zara02','biwi_hotel')


def read(name):return list(csv.DictReader((ROOT/name).open()))


def table(rows,cols):
    text='|'+ '|'.join(c[0] for c in cols)+'|\n|'+ '|'.join('---' for c in cols)+'|\n'
    for row in rows:
        values=[]
        for _,key in cols:
            value=row.get(key)
            if value is None or value=='':values.append('未运行/空组')
            elif isinstance(value,float):values.append(f'{value:.6f}')
            elif isinstance(value,str):
                try:values.append(f'{float(value):.6f}' if any(s in value for s in ('.','e-','e+')) else value)
                except ValueError:values.append(value)
            else:values.append(str(value))
        text+='|'+'|'.join(values)+'|\n'
    return text


def main():
    summaries={}
    for fold in FOLDS:
        for kind in ('G0','G1'):
            summaries[fold,kind]=json.loads((ROOT/fold/kind/'deterministic_scan/summary.json').read_text())
    decision=json.loads((ROOT/'first_round_decision.json').read_text());rows=[];aux=[];states=[];selection=[]
    for (fold,kind),s in summaries.items():
        best=s['best'];last=s['last']
        rows.append(dict(fold=fold,model=kind,epoch=best['epoch'],ADE=best['K1_distribution_mean_ADE'],FDE=best['K1_distribution_mean_FDE'],
            last_ADE=last['K1_distribution_mean_ADE'],last_FDE=last['K1_distribution_mean_FDE'],NLL=best['NLL'],params=s['parameters'],seconds=s['training_seconds']))
        aux.append(dict(fold=fold,model=kind,NLL=best['NLL'],K1_ADE=best['K1_first_sample_ADE'],K1_FDE=best['K1_first_sample_FDE'],K20_ADE=best['K20_independent_min_ADE'],K20_FDE=best['K20_independent_min_FDE']))
        states.append(dict(fold=fold,model=kind,ego=best['ego_norm_mean'],neighbor=best['neighbor_norm_mean'],correction=best['correction_norm_mean'],P95=best['correction_norm_p95'],ratio=best['relative_correction_mean']))
        curves=read(Path(fold)/kind/'deterministic_scan/curves.csv');minn=next(r for r in curves if int(r['epoch'])==s['min_NLL_epoch'])
        selection.append(dict(fold=fold,model=kind,ADE_epoch=best['epoch'],NLL_epoch=s['min_NLL_epoch'],minNLL_ADE=float(minn['K1_distribution_mean_ADE']),best_ADE=best['K1_distribution_mean_ADE'],first_NLL=float(curves[0]['NLL']),last_NLL=last['NLL'],first_train=float(curves[0]['train_NLL_window_mean']),last_train=last['train_NLL_window_mean']))
    main_table=table(rows,[('验证录像','fold'),('模型','model'),('ADE选中轮','epoch'),('best ADE','ADE'),('配对FDE','FDE'),('last ADE','last_ADE'),('last FDE','last_FDE')])
    aux_table=table(aux,[('录像','fold'),('模型','model'),('NLL','NLL'),('K1首采样ADE','K1_ADE'),('K1首采样FDE','K1_FDE'),('minADE@20','K20_ADE'),('minFDE@20','K20_FDE')])
    resource_table=table(rows,[('录像','fold'),('模型','model'),('参数','params'),('30轮训练秒','seconds')])
    selection_table=table(selection,[('录像','fold'),('模型','model'),('minADE轮','ADE_epoch'),('minNLL轮','NLL_epoch'),('minNLL轮ADE','minNLL_ADE'),('选中ADE','best_ADE'),('训练首轮NLL','first_train'),('训练末轮NLL','last_train')])
    actual=read('group_comparison.csv');groups=[r for r in actual if r['checkpoint_role']=='best' and r['mode']=='K1_distribution_mean']
    selected_names=('all','context_N2to4','context_N5to8','context_N9plus','legal_neighbors_none','legal_neighbors_present','legal_degree_low','legal_degree_mid','legal_degree_high',
        'speed_stationary','speed_slow','speed_walking','speed_fast','motion_smooth','motion_variable','path_turning_or_return','path_straight_or_low_speed',
        'observed_neighbors_none','observed_neighbors_present','observed_neighbors_but_no_model_edge','same_direction','opposing_direction','crossing_direction','relations_changing','relations_stable')
    group_table=table([r for r in groups if r['group'] in selected_names],[('录像','fold'),('历史/上下文组','group'),('目标窗口','target_windows'),('不同ID','unique_people'),('G0 ADE','G0_ADE_m'),('G1 ADE','G1_ADE_m'),('ΔADE','delta_ADE_m'),('G0 FDE','G0_FDE_m'),('G1 FDE','G1_FDE_m'),('ΔFDE','delta_FDE_m')])
    inter=read('checkpoint_graph_interventions.csv')
    inter_table=table(inter,[('录像','fold'),('G1固定权重输入图','intervention'),('均值ADE','K1_distribution_mean_ADE'),('均值FDE','K1_distribution_mean_FDE'),('邻居状态范数','neighbor_norm_mean'),('修正范数','correction_norm_mean')])
    sensitivity=[r for r in read('history_position_sensitivity_with_norm.csv') if r['epsilon_m']=='0.001']
    sensitivity_table=table(sensitivity,[('录像','fold'),('模型','model'),('源位置微扰(m)','epsilon_m'),('接收者终点位移(m)','receiver_final_prediction_shift_m'),('历史编码源低方差通道比例','motion_norm_sender_fraction_variance_below_eps'),('传播后接收者低方差通道比例','post_norm_receiver_fraction_variance_below_eps')])
    perm=read('trained_permutation.csv');max_pos=max(float(r['max_position_difference_m']) for r in perm);max_param=max(float(r['max_parameter_difference']) for r in perm)
    jac=read('trained_jacobian.csv');jac_table=table(jac,[('录像','fold'),('模型','model'),('源Jacobian最大','sender_jacobian_max'),('零图预测参数变化','zero_graph_distribution_difference')])
    aggregate=[r for r in read('aggregate_comparison.csv') if r['checkpoint_role']=='best' and r['mode']=='K1_distribution_mean']
    aggregate_table=table(aggregate,[('平均口径','scheme'),('G0 ADE','G0_ADE_m'),('G1 ADE','G1_ADE_m'),('ΔADE','delta_ADE_m'),('G0 FDE','G0_FDE_m'),('G1 FDE','G1_FDE_m'),('ΔFDE','delta_FDE_m')])
    context_notes=[]
    for fold in FOLDS:
        for name in ('context_N2to4','context_N5to8','context_N9plus'):
            r=next(r for r in groups if r['fold']==fold and r['group']==name)
            context_notes.append(f"{fold}/{name}：n={r['target_windows']}，ΔADE={float(r['delta_ADE_m']):+.6f}m、ΔFDE={float(r['delta_FDE_m']):+.6f}m。" if int(r['target_windows']) else f"{fold}/{name}：无样本，不能判断。")
    context_text='\n\n'.join(context_notes)
    reductions=decision['folds'];improved=[r['fold'] for r in reductions if r['ADE_reduction_m']>0];worse=[r['fold'] for r in reductions if r['ADE_reduction_m']<=0]
    if len(improved)==0:
        conclusion='两个录像G1整体ADE均未优于G0。当前受控图家族的独立预测收益验证失败，不支持继续堆叠图网络模块。'
    elif len(improved)==1:
        conclusion=f"G1仅在{improved[0]}整体ADE改善，在{worse[0]}未改善；属于单seed、录像相关的条件性现象，不支持跨录像稳定社会收益。"
    else:
        conclusion='G1在两个录像均有ADE改善，但单seed不足以证明稳定优势；是否达到预注册幅度与FDE要求另见门槛判定。'
    action='仅建议下一阶段预注册seed42/2026，不在本阶段自动训练。' if decision['recommend_seeds42_2026'] else '本阶段停止追加seed42/2026，不增加注意力、高阶传播或关系记忆。'
    verdict_table=table(reductions,[('录像','fold'),('ADE降低(m)','ADE_reduction_m'),('最低要求','ADE_required_m'),('ADE达标','ADE_worthwhile'),('FDE增加(m)','FDE_increase_m'),('FDE允许','FDE_allowed_m'),('FDE达标','FDE_safe')])
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'04_TRAINING_RESULTS.md').write_text(f'''# Stage4.2实际训练结果

四个seed17 run各完成30轮。每个run先5轮有限性、非零梯度、checkpoint精确恢复检查，通过后继续6–30轮。没有根据验证成绩改变学习率、种子、录像、模型结构或训练终点。此前CUDA积分中止尝试另存，见`00_RUNTIME_COMPATIBILITY_NOTE.md`；其没有完成指标，不作为模型失败或零结果。

## 主指标：K1高斯均值，单位m

{main_table}

验证ADE最低、完全相同取更早epoch；FDE取同一checkpoint。best/last即使同轮，也保留明确角色。训练代码提交`{summaries[FOLDS[0],'G0']['source_commit']}`，训练前协议提交`daf1e55975f94c04392a1a68c912131766e72d38`。数据hash/模型初始权重hash/入口与源码hash记录于每run的`run_config.json`；checkpoint SHA记录于每轮JSON与`checkpoint_inventory.csv`（120个）。

## 辅助指标：仍取ADE选择的同checkpoint

{aux_table}

K1首采样与K20使用同一固定逐窗口随机库。minADE@20和minFDE@20分别选最小，不要求同一候选；不能与确定性K1横向混排。NLL为目标窗口加权，曲线另存等窗口NLL；训练为等窗口损失。

## NLL与ADE的不同最优点

{selection_table}

NLL下降不保证位置均值误差下降。Stage4.1是NLL选模，本轮ADE选模；新表与旧Full/NoSocial不能作为同规则直接性能排名。所有30轮指标均保存，未挑选不同规则展示各模型最好数字。

四个run的ADE和NLL最优点均在第30轮预算边界；训练NLL仍降低，因此不能宣称充分收敛或全预算结论。本阶段遵守30轮终点，不据此追加训练寻找最低ADE。差异仅对本固定预算成立。

例如ZARA2 G1的NLL略优于G0（-1.540301 vs -1.539940），但ADE明显更差（0.713563 vs 0.661059），说明即便去除BN运行统计，概率似然和均值位置指标仍不等价。

## 容量、时间与资源

{resource_table}

统一1553参数、SGD0.01、64窗口累计、mini-batch16、clip1、无scheduler。所有初始参数hash相同。每轮实际显存峰值、窗口/目标吞吐、梯度均值/P95/最大、优化器更新数及验证耗时见`all_training_curves.csv`。上表为GPU同步测得纯训练时间，未包含准备、启动、验证、报告和checkpoint写盘，不将其当端到端时间。

G2独立训练未运行：预注册换边门槛未满足。seed42/2026未运行，ETH未评价，协议B未训练；均无虚构指标。

## 汇总口径

{aggregate_table}

录像宏平均与目标窗口加权分别保留；窗口重叠，不能把6886个目标窗口视作6886个独立人或独立证据。每录像unique ID、同人窗口均值另存；两个录像均历史曝光。
''')
    (REPORT/'05_INTERACTION_ANALYSIS.md').write_text(f'''# 全组交互与配对归因

{conclusion}

## 全部分组结果

下表为ADE选中的G0/G1在**完全相同目标**上的K1均值误差。Δ=G1−G0，负值为改善。所有组均展示，空组误差留空，非零样本才计算均值。完整表另含best/last、首采样与K20三模式，见`group_comparison.csv`。

{group_table}

`context_N*`为完整20帧离线上下文人数（含目标），不等于历史邻居密度。`legal_degree_*`为模型完整20帧上下文中4m真实历史边的8帧平均度；`observed_neighbors_*`来自原录像每历史帧的全部可见行人，包括不具备完整未来者。另保存完整8帧历史邻居数量。这三个集合不能混称社会密度。全体评价目标均有8帧历史。

速度分组均值阈值0.2/0.8/1.6m/s；加速度阈值0.5m/s²；转弯/回返为均速>0.2且净位移/路程<0.8；方向组由历史末步两人速度>0.2及cos±0.7定义，关系变化由历史边集合变化定义。仅历史特征参与分组，方向组可重叠。分组规则在训练前冻结，没有按未来误差选择病例。

同目标配对明细`paired_target_errors.csv.gz`；按录像内行人ID聚合`paired_person_means.csv`。不做重叠窗口iid显著性检验，组内构成不同不得冒充同样本效果。

## 中等/高人数的具体检查

{context_text}

Stage4.1旧Full相对NoSocial的“中等人数改善、高人数退化”仅是旧混杂结构下的观察。本轮ZARA2中等与高人数均退化，因此不复现其中的中等人数优势；高人数退化方向仍在，但不能证明同一机制。HOTEL没有9人以上完整上下文，不能外推高人数表现。

HOTEL无合法邻居组也改善，而ZARA2无邻居组也退化。G0/G1独立训练后权重不同，无边目标的模型间差异可来自训练时其他样本的图消息改变共享参数，不能解释为该目标当前邻居贡献。HOTEL历史低速/静止组虽ADE改善，FDE反而退化，不能只展示其ADE。

## 消息强度与图依赖

{table(states,[('录像','fold'),('模型','model'),('ego范数均值','ego'),('聚合邻居范数','neighbor'),('0.25邻居修正范数','correction'),('修正P95','P95'),('修正/ego均值','ratio')])}

上表为图传播处特征范数，不是轨迹空间残差或可学习gate。本阶段无可学习图gate，邻居行质量固定0.25、自身系数1。全部P95与逐目标范数已保存。非零消息或Jacobian只证明关系被使用，不证明使用有益。

{jac_table}

## 辅助推理干预：不是独立训练对照

{inter_table}

同一G1 best checkpoint分别换真实图、随机图、零图，仅评估固定权重的响应。换图是潜在分布外干预；不能据此声称G2训练性能或独立关系选择优势。随机图保持接收度/权重多重集，却改变距离与发送者列度，见`graph_control_statistics.csv`和`03_GRAPH_ONLY_ABLATION.md`中的度方向说明。G2未独立训练，其预测性能栏不存在；表中control是G1固定权重推理。

HOTEL G1零图ADE仍为1.192900m，优于独立训练G0的1.227714m；恢复真实图进一步到1.180636m，但FDE从零图2.132961m变为2.136477m。因此总体收益不能全部归于具体邻居。ZARA2去图ADE0.704924m略好于真实图0.713563m，但仍差于G0。真实图优于随机图只说明固定权重更适配训练关系，不构成独立训练的社会收益证明。

## 排列诊断

四个训练后best checkpoint分别检验前64个验证窗口，共256个窗口比较；最大高斯参数差{max_param:.9g}、最大恢复后位置差{max_pos:.9g}m，处于浮点误差。训练前2/3/5/8/12人还检查了输入与参数梯度。相比旧TXP明显节点索引混合，新逐人decoder消除了该结构依赖；归因结果不再被旧索引卷积路径混杂。

## 数值敏感性：不因结构测试通过而忽略反例

HOTEL首个有边窗口的跨人Jacobian明显较大。为解释该现象，固定两模型checkpoint，对每录像**首个有边窗口的首条边**进行仅历史位置微扰；选择的是终点x对源历史位置导数最大的一维，未读取未来GT来选病例。每次重算历史距离图，均未改变二值边；不进行训练、调参或选模。

{sensitivity_table}

HOTEL静止发送者/接收者在历史编码与图后PTN输入时间方差均为0；1mm源历史坐标扰动使G1接收者终点预测变化0.242211m。G0对他人微扰为精确0；ZARA2该例G1仅变化0.002361m。详细1e-5/1e-4/1e-3m扰动、Jacobian和各层方差见`history_position_sensitivity_with_norm.csv`。

小方差PTN可能放大微小历史扰动，是待验证的数值机制，不可凭此单例宣称解释全部录像误差。该规范化结构由G0/G1共同使用，本阶段不修改它，也不为某一模型单独调整epsilon。公平归因不等于稳健性成立，此反例限制正式方法开发；不能以消除排列敏感性掩盖新的数值敏感性。
''')
    (REPORT/'06_RESEARCH_DECISION.md').write_text(f'''# 研究决策与停止条件

{conclusion} {action}

## 预注册门槛的真实判定

{verdict_table}

两录像均达标：{decision['both_worthwhile']}；G2关系强度门槛：{decision['relation_control_gate']}；G2独立训练：未运行；额外种子：未运行。未在看完指标后放宽换边25%或修改“值得验证”定义。

## 回答本阶段11个问题

1. 逐人时序decoder正常运行：四次5+25轮训练、Gaussian/NLL、恢复、有限梯度全部通过；120个新checkpoint保留。
2. 节点索引敏感性在该受控家族中消除：2/3/5/8/12人前向/位置/梯度测试和256个训练后窗口恢复检查通过。排列等变是既有数学性质，不是新贡献。
3. NLL与ADE不一致会影响选模，旧Full29/30权重×BN统计交换复现了这一问题。本轮统一用ADE，NLL仅辅助；逐步概率目标与累计位置均值目标不同，没有据辅助指标改主结论。
4. 独立图收益只能由本轮匹配G1/G0回答，不能用旧Full/NoSocial替代。{conclusion}
5. G1相对G0的具体ADE/FDE和幅度见上表与`04_TRAINING_RESULTS.md`；未满足的录像不能以有利K20替换主指标。
6. 仅seed17、两个曝光录像，不足以证明跨seed稳定；即使同方向，也仍需下一阶段预注册。
7. 收益是否集中于交互，需结合**全部组**与目标数看。非零Jacobian、边干预差异与某组改善均不是整体或跨录像收益证明；小组和重复窗口不作强统计结论。
8. 新G0/G1均1553参数、同初始权重、同PTN及相同预算，可排除本对照的额外名义参数量与归一化实现差异。网络非线性对图消息的数值响应仍存在，不能外推为所有图架构的结论。
9. G1真实邻居Jacobian与固定checkpoint换图证明输入关系对预测有实际影响；G0为零。关系“被使用”和“提供预测价值”明确分开。
10. 本阶段不构成可直接写为新方法的第四章网络。排列等变、逐人归一化、普通RBF图均不包装成创新；可信的贡献首先是受控基线及失败/条件性证据。
11. 后续优先据失败类型重新冻结问题与公开基线协议，不继续增加复杂注意力、高阶或关系GRU。若仅条件性组收益，先设计独立预算的多录像验证，不宣称自适应交互机制已有效。

## 证据边界与下一步

本新网络不是维护版Social-STGCNN原架构：共享decoder、规范化和固定RBF图均为全体受控模型的共同设计。此实验不能直接否定原论文，也不能把新模型数字与Stage4.1 NLL选出的模型直接排行。A协议仍有未来存在性离线上下文筛选；history-only协议B的因果任务需另行预注册。本轮未评价ETH，不启动正式五场景训练。

数值诊断还发现静止历史PTN的高敏感反例（1mm邻居位置扰动对应0.242m终点变化）。在证明稳健性、完整训练预算和跨录像优势之前，不应把当前家族作为第四章正式网络；此处只提出待验证原因，不新增模块或修改本轮已冻结模型。

如果没有跨录像收益，停止本家族社会模块扩展；保留原公开Full作为参考与当前G0作为匹配基线。第四章暂不立新网络创新题目。下一阶段由用户决定是否审查更成熟的等变公开网络、history-only任务或其他有实证依据的问题，本阶段不自动执行。
''')
    (REPORT/'STAGE4P2_REPORT.md').write_text(f'''# Stage4.2社会图归因总报告

{conclusion}

{main_table}

四个独立seed17 run均完成30轮；先5轮smoke再从相同checkpoint继续，G0/G1各1553参数，同逐人归一化、相同初始化、相同解码/损失/训练预算，唯一主要差异为0.25强度显式邻居聚合。ETH封存，旧Stage1–4.1全部保留。G2只完成结构/图统计与G1固定权重辅助换图；未独立训练。seed42/2026未启动。

旧Full29/30端点已重算；固定权重交换BN统计使ADE显著变化，说明不能把NLL下降等同均值预测改善。新decoder不沿节点索引卷积；训练前排列/梯度/Jacobian测试通过，训练后256窗口位置恢复最大误差{max_pos:.9g}m。模型确实能够使用图关系，但非零消息本身不能证明收益。

本轮按最低验证高斯均值ADE选checkpoint，Stage4.1按NLL选，两阶段不能宣称相同选择协议。辅助首采样与K20各自报告，K20两个最小值独立选择。

{verdict_table}

{action} 不把常规图、等变性或规范化当作论文创新。不据曝光录像或重叠窗口宣称独立确认；本阶段只证明/否证当前小型受控家族在固定设置下的收益。

## 交付索引

- `01_PROTOCOL_AND_DIAGNOSIS.md`：预注册与旧结果审计。
- `02_ORDER_INVARIANT_DECODER.md`：公式、张量、参数、复杂度与等变边界。
- `03_GRAPH_ONLY_ABLATION.md`：匹配图对照及G2合法性/混杂。
- `04_TRAINING_RESULTS.md`：best/last、三类指标、预算与耗时。
- `05_INTERACTION_ANALYSIS.md`：全部组、同目标配对、消息强度与推理干预。
- `06_RESEARCH_DECISION.md`：11个科研问题、预注册门槛与停止条件。
- `00_RUNTIME_COMPATIBILITY_NOTE.md`：最初确定性CUDA积分兼容修复及保留的中止记录。

真实结果`results/tables/stage4p2/`包含4×30轮曲线、逐epoch JSON、最佳/最后逐目标压缩CSV、配对/分组/逐人表、120个checkpoint SHA、数据hash、配置、环境、原始运行日志压缩包和测试XML。权重仅本地`experiments/stage4p2/`，不进入Git。

训练前冻结提交`daf1e55975f94c04392a1a68c912131766e72d38`；有效训练代码提交`{summaries[FOLDS[0],'G0']['source_commit']}`。最终交付commit以`git log -1`为准，避免自引用SHA。完整测试、训练代码hash和历史SHA保护结果见`results/tables/stage4p2/final_delivery_checks.json`，前次检查`delivery_checks.json`同时保留。

仅本地commit，不自动push。用户手动执行`git push -u origin feat/stage4p2-graph-attribution`。完成后停止，等待下一阶段决策。
''')
    print(conclusion,action)


if __name__=='__main__':main()
