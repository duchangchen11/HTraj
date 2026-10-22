# 缓存与最终簇分配审计

范围：EXPLORATORY_DEVELOPMENT_ANALYSIS；既有5录像998重叠目标窗口，非独立测试。审计全阶段CPU_ONLY，未重新执行冻结YNet模型、采样、路径解码、实验训练或测试集评分；完整pytest包含既有模型的CPU人工张量结构测试，不是新增预测实验。GT只用于既有数据的离线测量。

基础提交`7ddcd1cc88433e46a2e0c519d9ae78b1b4fe6e22`，协议冻结提交`ef66d85c47862f3cf5049224e152c5435011afd8`。运行前完成协议/数值源码提交；analysis_summary.json记录实际分析代码commit。完整来源见protocol_frozen.json、cache_audit.json。Checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`；原预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`。核对全部998个唯一sample_id、原始10000个int64索引顺序、float32采样坐标、19中心、Softargmax、20Goal和20路径，均通过；原网络和参数未调用或更新。

## 坐标与身份

网络坐标对应原像素网格(x/0.25,y/0.25)，不加0.5像素中心偏移。原顺序保留；重复抽样没有去重。中心原坐标=centers_network/0.25。所有几何用float64离线计算，原始float32张量不改。每项距离另除参考图像对角线，保留两种单位；SDD像素不与ETH/UCY米制比较。

## 原KMeans源码及不能复原的部分

只读`experiments/stage11a/cache/official/ynet/utils/kmeans.py`，文件SHA见冻结协议。每轮先以旧中心计算choice_cluster，再按这些标签更新中心，最后检验center_shift平方或iter_limit。因此返回的标签不是严格意义上“最终更新中心再次分配”的标签。原Stage2C没有保留原迭代标签/优化过程，不能从最终19中心复原迭代路径、初始化后中心移动或停止原因。

本阶段定义`FINAL_CENTER_VORONOI_ASSIGNMENT`：a_j=argmin_k ||P_j-C_k||²，使用最终冻结中心，分块512点，完全等距时选择原head编号较小者。不调用原KMeans或重新迭代，不称ORIGINAL_LAST_ITERATION_CLUSTER_LABELS。

## 计数和尺度

实际测量18,962个最终Voronoi单元，每目标19个，人口总和每目标10000。空单元0。保存每簇人口、唯一位置、局部25px人口、半径均值/中位/P90/最大值、非局部点GT距离以及最终中心到重新分配样本均值的距离。后者中位2.7568609e-05px，P99 0.00011340191px，最大0.652358px。数值不完全为0，不能把冻结中心精确等同于最终重分配的均值，也不能据此宣布KMeans程序错误。

局部抽样统计重复点次数，唯一位置另列；86失覆盖目标共有6050条25px内抽样分配行，见local_proposal_assignments.csv.gz，保留原proposal_index和SHA。834成功、77零支持、1正质量遗漏全部保留。

结果：OBSERVED。历史保护清单继承到Stage2C新增缓存；前后检查见history_before/after.json。大型原缓存、论文PDF和原始数据不上传Git。
