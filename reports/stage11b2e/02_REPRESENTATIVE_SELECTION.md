# 固定Voronoi平方medoid选择

普通平方medoid，无原创算法主张。固定原10000次采样P与原19中心C，float64 CPU分配 a_j=argmin_k||P_j-C_k||²，精确等距取较小k。分区Z_k含重复采样的全部次数。

平方距离恒等式：sum_q||q-p||² = sum_q||q-mean(Z_k)||² + |Z_k| ||p-mean(Z_k)||²。因此选择真实原采样点中距离经验均值最近者；精确相同float64目标值取原全局proposal下标最小者。复制原float32坐标，不额外四舍五入。复杂度O(10000×19)，分块512计算，不构造大簇N×N矩阵；小规模暴力平方medoid人工测试通过。

Head0原Softargmax逐位保持，Head1..19严格对应原分区；空分区回退原中心并标注index=-1，实际空分区0，人口范围8—1389。保留重复点，不补候选。实际新旧每个目标均有20个唯一Goal，无重复Goal；最终唯一Path平均均为19.956914（按整条12帧路径精确坐标去重）。

18962个非head0中18961个发生变化；998个目标中K20 Goal整组完全不变为0。包含head0的平均位移1.531095833px；19中心平均位移1.611679825px。每head索引/人口/均值/坐标/固定分区SSD见representative_selection.csv；每目标proposal/center/index/新Goal SHA见goal_change_summary.csv，新Goal缓存文件SHA `be0bfed0d70f1fea3749465d9d1a29fbdc0016c8e2c64d15b759acc45ecf00a0`。

新旧Goal不等于全部解码图发生变化。作者trajectory_maps调用get_patch，在乘0.25后按np.round生成Goal图；此量化是旧代码原有计算，不是新算法额外取整。实际只761个head改变图中心，对应761条Path发生逐位变化；518个目标的完整K20 Path逐位未变。全部20条均真实重算，没有复用旧Path伪装新推理。

数据与结论范围：同一998个SDD开发验证目标窗口，5个既有录像，8帧历史/12帧未来、0.4秒，单位为SDD原参考像素px，resize=0.25。不是ETH/UCY，不是独立确认，不与米制结果排名。998个录像内行人ID；重叠窗口不能视为独立样本，本报告只给配对描述统计，不给窗口独立性p值。

原始模型SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`；原预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`；作者源码MIT版本 `a345ada1557f53dc2acb4c07f71d340d366980da`。所有计算沿用已冻结协议；无训练、无新采样、无KMeans重跑、无测试集访问。GT只用于全部新预测保存后的离线评分和绘图。

原始证据目录：[results/tables/stage11b2e](../../results/tables/stage11b2e/)。全仓pytest：652 PASS、1 SKIP、0 FAIL、0 ERROR；原始记录full_tests.txt/XML。SKIP不算PASS。 保护后92362项历史文件SHA全部未变（history_after.json）。
