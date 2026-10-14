# Stage2.6总报告：阶数、时间机制与关系归因

日期：2026-10-14。项目/home/lrj/HTraj，本地分支feat/stage2p6-order-verification，从与远程一致的Stage2.5 fb8373997134c476f98a1912cf82fbceb536ac39安全建立。旧代码、配置、报告、表格和权重222项SHA256保护，不覆盖原A0/A-R或任何旧实验。用户只要求本地commit，本阶段不自动push。

## 已实际完成

H1/H2/H3/NoOrder及两个近参数量控制，六个新模型各17/42/2026三个seed、30epoch，18次独立训练/540epoch/69120更新。另6次PedBase/Dynamic-RN旧结果经源代码、共同协议、数据/评估器、best/last权重和CPU重评核实后冻结引用。并非本次重训24次。

所有模型固定1024/96窗口、d64、Adam0.001、batch8、clip1、原masked L2、K1米制ADE/FDE、best最小开发ADE与配对FDE、last30。新网络从完整随机H3模板删除额外阶，共同模块同seed同初值；没有旧训练权重迁移。GPU只读等价性审计用了旧A-R权重核对数值，0优化步，与独立训练分开。没有ETH测试、五场景正式实验、B或第五章模型、额外loss/augmentation/decoder。

完整测试118项通过；对应16项结构测试及此前87项共103项预训练通过后才启动训练。正序/逆序/同帧/无高阶/多路径、前缀/梯度、padding/置换/回退、原A-R同权重CPU数值/梯度逐位一致和同预算均已验证。后续关系、身份计数、统计、单channel容量测试均通过，JUnit在results/tables/stage2p6/。

## 三种子best结果，mean ± sample std，m

| 模型 | 实际参数 | best ADE | 配对FDE |
|---|---:|---:|---:|
| PedBase | 35352 | 0.439398 ± 0.003578 | 1.070541 ± 0.008332 |
| Dynamic-GNN-RN | 102235 | 0.440075 ± 0.002073 | 1.063977 ± 0.011466 |
| TOHI-H1 | 73627 | 0.441364 ± 0.000896 | 1.071762 ± 0.002066 |
| TOHI-H2 | 94812 | 0.441229 ± 0.001363 | 1.077324 ± 0.010186 |
| TOHI-H3 | 115997 | 0.444220 ± 0.002895 | 1.076493 ± 0.009294 |
| TOHI-NoOrder | 115997 | 0.445767 ± 0.004754 | 1.083143 ± 0.014830 |
| TOHI-H1-Cap | 115995 | 0.439586 ± 0.003318 | 1.070475 ± 0.008745 |
| TOHI-H2-Cap | 116060 | 0.442316 ± 0.003106 | 1.070669 ± 0.014919 |

PedBase主ADE均值仍最好，Dynamic-RN配对FDE均值最好。H1-Cap接近PedBase，但ADE只一个seed优于它。所有种子保留，未更换验证集、预算或best主终点。last及逐seed全部另列，不用最后epoch中某项较好改变预定结论。

## 十个研究判断

| 问题 | 判断 | 实验依据与限制 |
|---|---|---|
| 1. H2优于H1吗 | 未建立稳定独立收益 | Δbest ADE=-0.000135m但FDE=+0.005562m，仅1/3同时改善；H1-Cap均值更好 |
| 2. H3优于H2吗 | 主ADE负结果，暂不保留三阶为精度贡献 | 3/3 ADE更差，均值差+0.002991m；FDE均值微降但方向混合；对近参数H2-Cap 3/3 ADE/FDE都更差 |
| 3. 严格有序优于NoOrder吗 | 均值稍好，跨seed不稳定 | H3-NoOrder平均ADE/FDE差-0.001547/-0.006650m，但seed17两者更差；放宽规则同时改变walk数量、年龄、尺度 |
| 4. 优于Dynamic-RN吗 | 没有稳定整体优势 | H2/H3/NoOrder best均值ADE/FDE都更差；H3 ADE三个seed均更差；Dynamic还有GRU/LN结构差异 |
| 5. 高密度明确改善吗 | 未证明高阶优势 | full8×high的H2/H3 ADE都弱于Dynamic-RN；相对PedBase H2仅很小ADE下降/FDE稍差，H3 ADE更差 |
| 6. 改善跨seed稳定吗 | 否 | Native H2整体仅1/3同时改善H1；时间对照方向混合；容量对照也有seed依赖 |
| 7. 能排除额外容量吗 | 不能完整排除 | H1-Cap接近PedBase且优于H2/H3均值；H2-Cap与H3近参数而更好；宽delta与递推记忆不同，单channel scorer还有8321惯性参数 |
| 8. 真用邻居关系吗 | 确认计算依赖，未证明高阶独立泛化价值 | 强重连/路径删除改变预测和误差，社会分支非零；但本次负对照只为固定checkpoint的分布外敏感性 |
| 9. 保留A吗 | 保留可审计原型与一阶R控制，暂停三阶核心贡献叙述 | 结构正确、数据与机制可验证；性能证据不足，二阶也只作待验证候选 |
| 10. 第四章如何重设计 | 从可靠接收/有效信息选择开始，非继续加阶 | 新进入目标连续性mask阻断消息、walk时间重复、共享alpha与跨阶payload不匹配、状态尺度是优先假设；全部待后续单因素验证 |

## 时间计算图与身份计数

NoOrder允许q>1的source从old low-state变为old low-state + current low-message，q1仍读相同h_prev，carry/图/读出不变。仍是因果非严格升时t1≤t2≤t3，不打乱轨迹帧。每帧每阶一次projection+aggregate，与H3参数/投影次数相同，但额外张量加法且有效walk总数不同，不能宣称完全相同信息预算。

严格验证二阶walk87.69%含3名不同身份，12.31%返回起点；80.97%的不同端点对有多个中间人。三阶29.40%含重复身份，70.60%含四个不同身份。任意三阶覆盖95.35%，四身份三阶88.66%，纯重复身份支持82个目标/6.69个百分点。因此回返抬高了覆盖但不能解释全部失败；相同身份三元组在多个时间出现，唯一身份三元组仅相当于二阶时间实例的8.22%，时间实例数不等独立信息。统计只用历史，waiting不增加事件，不是因果社会事件。

## 状态尺度与历史混杂

记录每阶before/source/message/carry/after的mean/P95、beta/gate、实际残差修正和投影梯度。H3三阶best raw mean45.63–75.56，高于一阶11.15–19.23，梯度有限非零，没有NaN/Inf。H2二阶beta随seed约0.012/0.899/0.037，不宜把权重当稳定贡献比例。

H3 raw norm与ADE总体Spearman略负，full8高密度却约+0.10–0.12，说明历史与密度混杂。相关性不证明尺度造成误差，未运行独立尺度控制。无邻居/部分历史含难预测的新进入行人；所有社会模型partial组平均指标都弱于PedBase。当前代码同时用continuity约束carry与新消息接收，单历史点目标不能获得社会先验，这不是已实现的修复，而是明确待验证设计点。

## 加强关系干预：96条真实记录

每个H2/H3/NoOrder/Dynamic-RN的三个seed、best checkpoint、八种case（含full），共96条CPU推理记录。图介入前后encoder/decoder/权重不改，不使用未来构图。每项完整记录实际边数、替换率、入/出度差异及超4m比例。

| 模型 | local最大换边 ΔADE/m | scene随机 ΔADE/m | scene最大换边 ΔADE/m | 入出度重连 ΔADE/m | 删间接路径 ΔADE/m | 匹配随机删边 ΔADE/m | 全删消息 ΔADE/m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic-GNN-RN | +0.001495 | -0.001961 | +0.001513 | +0.002090 | +0.008738 | +0.008592 | +0.035115 |
| TOHI-H2 | +0.000291 | +0.005052 | +0.008831 | +0.009347 | +0.014308 | +0.010348 | +0.015138 |
| TOHI-H3 | +0.000052 | +0.002610 | +0.006862 | +0.007128 | +0.012856 | +0.008631 | +0.013724 |
| TOHI-NoOrder | +0.000563 | -0.001106 | -0.000229 | +0.000303 | +0.016866 | +0.011624 | +0.023652 |

同4m、同target入度最多可替换22.81%（达到上限）；scene随机/最大替换65.67%/90.49%，保持入出度双边交换替换78.66%。后三者明确放宽4m，分别约59.98%/82.72%/69.09%的控制边在4m外，存在明显空间分布偏移。随机图仍包含真实其他人的运动，不是纯噪声。

删严格间接路径移除82.52%原边；匹配随机删边保持每帧每target完全相同剩余度数。前者保证严格版无二/三阶支持，NoOrder仍可有同帧路径。它同时删除直接信息并改变alpha/carry关系输入，因此不等于纯高阶值清零。H2/H3删关键路径比匹配随机删边更伤，但这仅支持条件依赖/共适应，不能覆盖独立训练负结果。NoOrder和Dynamic的随机重连有时更好，不预写“真实图必然最优”。所有图负对照独立重训练均未运行。

## 下一步与第四章定位

建议暂不把三阶保留为核心网络，不直接开展正式五场景。保留H1+R为清晰控制，二阶只有在明确独立收益时再加入。下一阶段优先：

1. 固定H1+R，单独分离carry重置与新可见target接收，显式核查未知速度/观测长度；不同时更改数据或损失。
2. 若继续二阶，比较三身份简单路径/不同中间人选择与当前无身份walk累计，并加入近容量/global scene-motion控制。当前共享alpha按source ego评分，却用来传递跨阶payload，内容/新鲜度选择可能不充分；这只是代码支持的设计假设。
3. 固定H、融合、数据和预算，再单独测试message/state尺度或谱约束，避免把归一化、时间规则与阶数混改。
4. 后续如支持时间机制，匹配有效路径预算/新鲜度并补同分布关系重训练控制；有证据后才推进五场景与论文贡献。

第四章目前可以写结构设计、依赖验证、融合与独立消融及失败边界；不能写现有高阶已优于基础/动态图。独立小论文尚缺稳定收益与充分机制归因。B、第五章和完整训练本次未开展；以上建议未启动，等待下一步指令。

## 证据索引、来源与范围

六份指定报告全部完成：01_ORDER_ABLATION.md、02_TEMPORAL_ORDER_COMPARISON.md、03_STATE_SCALE_ANALYSIS.md、04_COMPLEX_SCENE_ANALYSIS.md、05_MODEL_COMPARISON.md和本报告；00_PROTOCOL_AND_HYPOTHESES.md保留训练前假设。

真实表：all_runs.csv、three_seed_summary.csv、paired_differences.csv（57条）/paired_difference_summary.csv（19对）、group_comparison.csv（2832条）/three_seed_group_summary.csv（944条）、diagnostics_best_last.csv、relation_interventions.csv（96条）、state_error目录、path_identity_*和distinct_path_coverage_*；每个新run的config/30行epochs/best-last groups/summary均保存。新best/last权重在本地experiments/stage2p6/new，全部被gitignore；旧权重保留，checkpoint hash在summary。

新模型训练源ec8f665c1faab7ee6cd64172c8852c904c8f012b；冻结模型训练源90d987799d06bef4e213952342da231e93792a41。data manifest SHA=c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced；subset SHA=4b4a50ce47b8d8a2fc574a814f80982b602fed237939f7d7f019492be6dc4be1；新protocol SHA=f25f43f45bd4ef4fb9230c688666ee035b19eb3c8c3f0b8798d99be1e08770de。分析脚本在执行前分别提交，结果保留各自commit/源文件SHA。

所有实验判断来自本仓库真实运行；相关文献归属/范围沿用Stage1来源，不把残差/门控或同帧GNN当新理论。CPU/GPU归约与scatter不逐位保证，单一96验证反复用于开发，不是独立测试；三个seed不提供跨场景泛化或显著性证明。独立尺度训练、匹配有效walk的时间对照、图关系独立重训练、ETH测试、五折、方案B均未完成，未伪称完成。

本阶段只提交本地分支，用户手动运行git push -u origin feat/stage2p6-order-verification。完成后停止。
