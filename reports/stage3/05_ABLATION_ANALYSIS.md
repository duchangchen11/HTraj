# 关系选择贡献：独立训练与固定checkpoint分开

## 从头训练的主要证据

无社会信息：PedBase独立旧训练经完全兼容核查后冻结引用；其名义参数较少，不能单独排除通用修正容量。当前权重：Receiver-Fix（旧alpha）和Routing-Current（新特征MLP）从头训练。历史：Relation-Routing GRU从头训练。删除更新：NoUpdate同GRU每步前状态0，从头训练。Uniform同候选均匀权重从头训练。Random可靠同场景源、约入度匹配、放宽4m，从头训练。六种要求都具备训练/冻结训练控制，不用推理关闭代替重训。

History−NoUpdate best ADE/FDE=+0.001090/+0.000749m；名义参数73355相同，NoUpdate另有1728循环权重死参数。History没有赢得额外循环容量的收益。History−Current=-0.000991/-0.006536m，但Current编码/参数不同，非独立记忆证明。History−Uniform=+0.000191/-0.007953m，ADE/FDE方向分开报告。

History−Random=+0.000553/-0.000449m。Random实际43529边vs真实43708，差0.410%；替换65.579%、59.900%控制边超4m，CPU/CUDA随机图249条对称差边。它改变空间/连续性分布，不能称严格关系单因素或仅容量对照。即便真实图好也不足独立证明，若随机更好同样如实记录。

## Receiver-Fix单阶原结构对照

旧H1 seed17最佳/最后checkpoint按Stage3分组CPU重评，与旧主指标误差<1e-6m，源码与共同数据/损失/预算未变，记录supplementary_H1/audit.json。不是又训练一次旧H1。

| 组 | 旧H1 best ADE/FDE | RF best ADE/FDE | RF−H1 ADE/FDE |
|---|---|---|---|
| all | 0.442334/1.074096 | 0.443099/1.074098 | +0.000764/+0.000002 |
| history_full8 | 0.351502/0.874685 | 0.353522/0.877260 | +0.002020/+0.002576 |
| history_1 | 2.771184/5.288990 | 2.771639/5.275050 | +0.000455/-0.013940 |
| history_2to4 | 0.637589/1.303187 | 0.624641/1.272891 | -0.012948/-0.030296 |
| history_5to7 | 0.556312/1.258371 | 0.557952/1.276814 | +0.001639/+0.018443 |
| new_entry_with_neighbors | 2.597915/5.054080 | 2.597499/5.035686 | -0.000416/-0.018394 |

接收资格及必需未知速度语义是结构差异。各模型独立训练/按整体ADE分别选epoch，不能把某分组差值解释为所有已知目标的同权重必然改善。无邻居targets的预测也因共享训练参数改变，故有邻居组小改善不能全归于邻居信息。

额外receiver_same_checkpoint_probe.csv/provenance.json使用旧H1 seed17 best完全相同权重、仅切Receiver-Fix接收政策。整体ΔADE/FDE=−0.000171/−0.001763m；一帧有邻居组却为+0.009547/+0.025356m，全部一帧+0.007779/+0.020074m。完整8帧指标差0，坐标最大浮点差4.7684e-7m：原先“必须逐位相同”的辅助断言失败，检查后用明确1e-5m浮点容差通过，不隐藏该数值边界。新边行数改变GEMM批量形状可有微小舍入，远小于训练后2mm退化。这不是结构依赖变化或重新训练，该介入也对旧模型引入新输入分布。相对H1完整历史的退化主要反映共享训练/不同checkpoint选择，而非接收政策对完整历史的直接计算效应。

## 固定best checkpoint辅助干预（CPU、不是独立训练）

| 模型 | best固定推理case | ΔADE(m) | ΔFDE(m) | 预测改变均值L2(m) |
|---|---|---|---|---|
| Receiver-Fix | full | +0.000000 | +0.000000 | 0.000000 |
| Receiver-Fix | no_messages | +0.031658 | +0.055972 | 0.148007 |
| Receiver-Fix | uniform | +0.001367 | +0.001873 | 0.038562 |
| Receiver-Fix | reset_relation_each_frame | +0.000000 | +0.000000 | 0.000000 |
| Receiver-Fix | scene_random | +0.001273 | +0.005980 | 0.048343 |
| Relation-Routing | full | +0.000000 | +0.000000 | 0.000000 |
| Relation-Routing | no_messages | +0.027117 | +0.038302 | 0.135670 |
| Relation-Routing | uniform | +0.000529 | +0.002220 | 0.034000 |
| Relation-Routing | reset_relation_each_frame | +0.000298 | +0.002003 | 0.015870 |
| Relation-Routing | scene_random | +0.002723 | +0.013638 | 0.064898 |

no_messages精确绕过全部social历史；uniform改权重；reset_relation每帧置0边先前状态，Receiver-Fix该case是明确no-op；scene_random放宽距离规则。比较各自CPU full，预测变化仅用历史cutoff可见目标，ADE/FDE只用于事后评价，不用于选邻居或确认集调参。OOD/共适应效应不能替代上方从头训练结果。

## 状态与门控按历史分组

| 模型/分组 | 可见目标 | gate均值/P95 | state范数均值/P95 | 修正范数均值/P95 | 归一熵均值 |
|---|---|---|---|---|---|
| Receiver-Fix/all | 1226 | 0.0725/0.3014 | 18.846/25.723 | 1.071/4.496 | 0.5632 |
| Receiver-Fix/history_full8 | 1021 | 0.0634/0.3048 | 19.975/25.896 | 0.931/4.365 | 0.5783 |
| Receiver-Fix/new_entry_with_neighbors | 22 | 0.1221/0.1295 | 6.843/9.989 | 1.759/2.004 | 0.5067 |
| Receiver-Fix/new_entry_without_neighbors | 5 | 0.0000/0.0000 | 0.000/0.000 | 0.000/0.000 | 0.0000 |
| Relation-Routing/all | 1226 | 0.0524/0.2005 | 17.941/22.869 | 0.818/3.471 | 0.5396 |
| Relation-Routing/history_full8 | 1021 | 0.0434/0.1810 | 19.008/22.968 | 0.680/3.127 | 0.5509 |
| Relation-Routing/new_entry_with_neighbors | 22 | 0.1021/0.1304 | 6.397/8.775 | 1.491/1.936 | 0.6424 |
| Relation-Routing/new_entry_without_neighbors | 5 | 0.0000/0.0000 | 0.000/0.000 | 0.000/0.000 | 0.0000 |
| Routing-NoUpdate/all | 1226 | 0.0424/0.1568 | 17.867/23.119 | 0.597/2.411 | 0.6587 |
| Routing-NoUpdate/history_full8 | 1021 | 0.0343/0.1407 | 19.005/23.458 | 0.486/2.173 | 0.6732 |
| Routing-NoUpdate/new_entry_with_neighbors | 22 | 0.1001/0.1119 | 6.249/8.398 | 1.297/1.547 | 0.5896 |
| Routing-NoUpdate/new_entry_without_neighbors | 5 | 0.0000/0.0000 | 0.000/0.000 | 0.000/0.000 | 0.0000 |

范数为学习特征单位，不是速度或米。熵是cutoff alpha熵除log(degree)，degree≤1记0，所以跨组比较受degree/无邻居比例影响；Uniform有多邻居时等于1。全部21组/best-last/六模型保留252行诊断，未过滤高误差targets。无邻居新进入者support/gate/correction为0，精确回到训练模型自己的ego，不意味着等于另一个PedBase训练权重的预测。

人工测试在current几何、current速度和ego全部固定时，改变较早本对关系历史会改变History路由，而NoUpdate不变；证明计算依赖结构。已训练checkpoint敏感性与梯度说明模型计算使用消息/记忆的程度；从头消融却没有建立稳定精度优势。不能把非零梯度、非零gate、注意力模式或删消息退化直接写成新机制有效。

有效容量未严格排除、Random有分布偏移、仅一seed且22个一帧有邻居目标。当前结论是所测原型没有独立有效选择收益，不是证明所有关系信息理论上无用。
