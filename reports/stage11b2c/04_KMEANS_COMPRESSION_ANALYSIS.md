# 原始抽样→19中心+Softargmax覆盖变化

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

最终25px覆盖834/998=83.567134%，与原164个A25失败身份逐样本一致；10/25/50px失败数为489/164/99。原10000点命中但最终K20未命中86个。最终失败中零质量共现77个、正质量抽样未命中1个、抽样命中后最终失去覆盖86个。它们是观察到的链路事件，不能直接当作模块造成的ADE因果份额。

| type | targets | fraction | fixed_A25_count | fraction_of_fixed164 | minADE20 | minFDE20 |
| --- | --- | --- | --- | --- | --- | --- |
| zero_support_final_miss | 77 | 0.077154 | 77 | 0.469512 | 101.653635 | 185.439508 |
| positive_support_sampling_miss_final_miss | 1 | 0.001002 | 1 | 0.006098 | 29.919677 | 50.968139 |
| proposal_hit_final_miss | 86 | 0.086172 | 86 | 0.524390 | 24.907370 | 39.900488 |
| both_hit | 834 | 0.835671 | 0 | 0.000000 | 7.639144 | 9.378308 |
| proposal_miss_final_hit | 0 | 0.000000 | 0 | 0.000000 | nan | nan |
| other | 0 | 0.000000 | 0 | 0.000000 | nan | nan |

coverage_transition.csv保留3半径×998目标的A/B/C；all_event_combinations.csv保留所有8种组合（不存在的组合计数0、误差均值不伪填0）。T1—T6命名指标有重叠（例如正质量但抽样未命中与最终反而覆盖），不可直接相加；上述exclusive_category是互斥统计。半径10/50的fixed164比例仍以原A25固定164身份为分母，不能当作各阈值自身失败群体。

原始采样未命中而最终中心/Softargmax反而命中共0个（25px）。本次未观测反向恢复，因此这批数据中的最终覆盖是原始抽样覆盖的子集；但算法定义不保证该性质。KMeans中心不是必然从输入样点中选取；均值落入原抽样未命中邻域并非算法错误。原输入10000点的重复性/密度未改变，19个中心顺序不变。

Softargmax补充KMeans19在25px漏掉的目标0个。独立Softargmax/KMeans19/combined20的各阈值命中及距离见softargmax_kmeans_comparison.csv；这是同一冻结集合内来源分解，不是改变K后重训的公平模型实验。

静态源码发现：SoftArgmax2D对exp(logit-max)/(sum(exp)+1e-6)进行坐标加权，原epsilon未改；它不是阈值化Sigmoid采样分布的均值。附加float64公式与native位置差最大0.00038768986px（仅反映独立CPU/GPU数值归约差异，不要求该重算逐位一致；真正逐位门槛由原预测/新导出回归满足）。softargmax_formula_audit.csv记录两种分布的差异、Softargmax附近质量和坐标均值，但没有预注册模式数量/干预测试，故“多峰平均导致失败”仍UNVERIFIED，不能据GT距离或单张图证明。

Softargmax的作用依赖半径：10px下独立补救KMeans19遗漏14目标（合计509 vs495），25px/50px下独立补救均为0。不能用25px无新增覆盖推断它在所有精度要求下无作用；1/19/20候选来源比较不是公平重训练实验。
