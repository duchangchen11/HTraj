# Stage3真实有限规模实验

训练源c8d151c8c53367ef361d23d30569c9ea5ff8cf52。新模型六次seed17从头30epoch，共180epoch/23040 optimizer updates；不是六模型三seed。PedBase和Dynamic-GNN-RN为原Stage2.5完全兼容冻结结果，三seed/best-last共12个checkpoint已核实源码/超参数/manifest/hash/CPU重评；主表只用seed17配对。旧三seed记录不当作新模型稳定性验证。CV无训练/参数，ADE/FDE=0.462669/1.146167m。

## 固定协议与完整主指标

1024训练/96开发，d64、obs8/pred12、Adam0.001（betas0.9/0.999、eps1e-8、weight_decay0）、batch8、clip1、原masked per-track L2、K1米制ADE/FDE。best最小开发ADE/最早严格平局，配对FDE取相同epoch；last30另列，未挑FDE最佳epoch或改终点。全部ADE1202目标×窗口、末帧FDE839、cutoff可见1226。数据支持重叠窗口不能当独立1202人。source/protocol SHA在各run/config；共同Adam细节在shared_training_configuration.json。

| 模型 | 名义/扣已知死参数 | best epoch | best ADE/配对FDE(m) | last30 ADE/FDE(m) | 训练秒（旧冻结单列） |
|---|---|---|---|---|---|
| PedBase | 35352/35352 | 24 | 0.441765/1.075548 | 0.447649/1.087694 | 107.43（旧） |
| Dynamic-GNN-RN | 102235/93914 | 24 | 0.437891/1.050755 | 0.442564/1.071946 | 252.50（旧） |
| Receiver-Fix | 73627/65306 | 19 | 0.443099/1.074098 | 0.447179/1.082299 | 276.57 |
| Relation-Routing | 73355/65034 | 20 | 0.442996/1.073757 | 0.451941/1.085896 | 297.60 |
| Routing-Current | 70931/62610 | 20 | 0.443987/1.080292 | 0.447394/1.082781 | 285.20 |
| Routing-NoUpdate | 73355/63306 | 24 | 0.441905/1.073008 | 0.448323/1.090936 | 290.53 |
| Routing-Uniform | 73627/56537 | 30 | 0.442805/1.081710 | 0.442805/1.081710 | 245.77 |
| Routing-Random | 73355/65034 | 23 | 0.442443/1.074205 | 0.444914/1.077513 | 322.16 |

全部新模型社会消息结构只含直接源ego，不增加阶数。扣死参数不是函数空间维数，NoUpdate循环weight_hh还失效1728；原run/config的scorer计数不包含该项，权威容量审计为parameter_capacity_audit.csv。Timing为真实训练循环耗时，评价另计total_run_seconds；冻结耗时来自旧run，系统状态可能不同，不能作为严格硬件效率排名。

## 所有要求的历史分组（ADE/FDE，m）

| 历史分组 | ADE/FDE目标数 | CV | PedBase | Dynamic-RN | Receiver-Fix | History-Routing |
|---|---|---|---|---|---|---|
| history_full8 | 999/665 | 0.3671/0.9339 | 0.3561/0.8883 | 0.3511/0.8608 | 0.3535/0.8773 | 0.3546/0.8793 |
| history_1 | 27/24 | 2.9020/5.5295 | 2.7820/5.2930 | 2.7887/5.3157 | 2.7716/5.2750 | 2.7440/5.2138 |
| history_2to4 | 96/85 | 0.6567/1.3459 | 0.5858/1.2091 | 0.6018/1.2331 | 0.6246/1.2729 | 0.6124/1.2581 |
| history_5to7 | 80/65 | 0.5996/1.4383 | 0.5488/1.2594 | 0.5321/1.1812 | 0.5580/1.2768 | 0.5675/1.2936 |
| new_entry_with_neighbors | 22/19 | 2.7030/5.2382 | 2.6067/5.0544 | 2.6130/5.0727 | 2.5975/5.0357 | 2.5735/4.9837 |
| new_entry_without_neighbors | 5/5 | 3.7776/6.6364 | 3.5533/6.1997 | 3.5619/6.2387 | 3.5379/6.1846 | 3.4942/6.0883 |
| full8_high_density | 791/556 | 0.3687/0.9068 | 0.3593/0.8714 | 0.3520/0.8392 | 0.3590/0.8645 | 0.3572/0.8598 |
| full8_low_density | 51/38 | 0.5869/1.5018 | 0.5021/1.1965 | 0.5147/1.2451 | 0.5008/1.1979 | 0.5101/1.1951 |
| full8_medium_density | 151/65 | 0.2933/0.8868 | 0.2968/0.9008 | 0.2986/0.8685 | 0.2812/0.8401 | 0.2940/0.9039 |
| full8_no_neighbors | 6/6 | 0.1550/0.3575 | 0.1796/0.3693 | 0.1562/0.3448 | 0.1946/0.4315 | 0.2089/0.4182 |

完整8帧/1帧/2–4/5–7按总可见帧数划分，覆盖全部1202个ADE目标，不删除难例。new_entry按cutoff可见但不连续，本子集恰好与1帧相同；有邻居按Receiver-Fix历史候选资格定义，不使用未来坐标。完整历史密度沿用Stage2.5的平均连续帧入度：无0、低(0,1]、中(1,3]、高>3。不是改成新图密度以改善结果。

每模型/best-last的全部21个分组、各组可见目标数/窗口数/ADE-FDE有效样本数，以及对PedBase和Dynamic-RN差值均在all_groups.csv（含原9个场景/链组）。不只报告改善组。CV不足两个连续点时明确stationary fallback，不用未来速度。

## 主要观察

Receiver-Fix一帧有邻居ADE比PedBase降低0.009192m（约0.35%），History降低0.033235m（约1.27%），远小于预设10%。两者完整8帧相对PedBase略好，但2–4与5–7帧更差，整体ADE均差于两个基础。FDE仅略好于PedBase，均差于Dynamic-RN。

与结构更匹配的原H1比较：RF一帧有邻居ADE仅改善0.000416m，全部一帧ADE反而差0.000455m；完整8帧ADE/FDE差0.002020/0.002576m。不能用相对PedBase略好代替接收修复独立有效的证据。相同旧H1 checkpoint只切接收条件的辅助推理，该一帧有邻居组ADE/FDE反而增加0.009547/0.025356m，整体稍好−0.000171/−0.001763m；没有直接证明新进入者获益。

History比Current稍好，但比同名义参数NoUpdate更差；不能据Current结构/容量不同就证明历史记忆贡献。Uniform ADE也略优于History，FDE较差。Random完整结果见主表，约边数/半径/后端混杂见relation_legality.json，不能写成严格同分布证明。

## 状态、权重与来源

每epoch曲线all_new_epoch_curves.csv及各new/model/seed_17/epochs.csv；config/summary包含参数、时间、源commit、manifest/protocol SHA、best/last checkpoint SHA。权重本地experiments/stage3/new/model/seed_17/checkpoints，不进Git。冻结曲线与配置原路径在frozen_baselines.csv。

all_diagnostics.csv及各diagnostics_best/last记录社会消息计数、unknown receiver消息、before/source/message/carry/after范数mean/P95、关系state范数、残差gate P05/P50/P95及修正。epochs还记录projection/关系/encoder/fusion/decoder梯度。按观察分组的gate/范数/alpha熵在cohort_state_diagnostics.csv，不用GT掩码挑低误差目标。

diagnostic_semantics.json明确定义：消息数是结构edge-time事件，非非零payload数量；统一wrapper的relation_before/memory_inherited在Current/Legacy/Uniform只是缓存/连续资格日志，只有History/Random实际把其读入GRU，不能把该字段非零当作当前帧模型使用记忆。每run新增capacity_audit.json给出完整已核实死参数，保留原训练捕获config/summary及checkpoint不变。

没有ETH指标、正式五场景、模型B、增加高阶或第二轮新训练。没有合格候选而按预设退出，不反复选择seed。独立确认窗口0；详见06_RESEARCH_DECISION.md。
