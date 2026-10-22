# 固定86失败恢复与原成功退化

原86身份保持不变：恢复2，仍失败84；原834成功目标保留829，新增失覆盖5。总失覆盖164→167，覆盖净变化2−5=−3。不是只统计2个恢复忽略5个新失败。

| original_category25 | targets | original_covered25 | alternative_covered25 | recovered | newly_lost | original_minADE20 | alternative_minADE20 | original_minFDE20 | alternative_minFDE20 | median_local_draws25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_hit | 834 | 834 | 829 | 0 | 5 | 7.639144 | 7.633215 | 9.378308 | 9.367462 | 2299.000000 |
| positive_support_sampling_miss_final_miss | 1 | 0 | 0 | 0 | 0 | 29.919677 | 29.919677 | 50.968139 | 50.968139 | 0.000000 |
| proposal_hit_final_miss | 86 | 0 | 2 | 2 | 0 | 24.907370 | 24.860721 | 39.900488 | 39.955815 | 50.000000 |
| zero_support_final_miss | 77 | 0 | 0 | 0 | 0 | 101.653635 | 101.604366 | 185.439508 | 185.285528 | 0.000000 |


86组minADE20 24.907369696→24.860720804px；minFDE20 39.900487929→39.955815256px。局部原proposal25px次数与ADE变化的Spearman描述相关为-0.258647，包含相同窗口行人的相关性，不能作因果解释或独立样本显著性。

77个原阈值后零支持目标覆盖仍为0/77。没有新采样或改阈值，故不能宣称恢复原离散分布不存在的概率支持。1个原正支持有限抽样未命中目标也继续失败。本次两例恢复来自原86，未重定义失败组。

deathCircle覆盖310/443→310/443，组内恢复和损失抵消；其他四录像524/555→521/555，下降-0.540541pp。deathCircle占原86中的65/86，但不能由该录像微小路径改善推广到五录像。bookstore覆盖399/423→396/423，其余三录像覆盖数量保持，路径指标仍可能改变。

固定案例规则：四种配对结果分别按sample_id排序取floor((n−1)/2)，而非挑选最佳误差；case_manifest.csv与case_plot_data.json保留四个真实案例的完整20Goal/20Path/GT/SHA。图6同时展示恢复、退化、持续失败、持续覆盖。六图各PNG/SVG与源数据定位见figure_manifest.json。GT仅离线评分/绘图。

数据与结论范围：同一998个SDD开发验证目标窗口，5个既有录像，8帧历史/12帧未来、0.4秒，单位为SDD原参考像素px，resize=0.25。不是ETH/UCY，不是独立确认，不与米制结果排名。998个录像内行人ID；重叠窗口不能视为独立样本，本报告只给配对描述统计，不给窗口独立性p值。

原始模型SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`；原预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`；作者源码MIT版本 `a345ada1557f53dc2acb4c07f71d340d366980da`。所有计算沿用已冻结协议；无训练、无新采样、无KMeans重跑、无测试集访问。GT只用于全部新预测保存后的离线评分和绘图。

原始证据目录：[results/tables/stage11b2e](../../results/tables/stage11b2e/)。全仓pytest：652 PASS、1 SKIP、0 FAIL、0 ERROR；原始记录full_tests.txt/XML。SKIP不算PASS。 保护后92362项历史文件SHA全部未变（history_after.json）。
