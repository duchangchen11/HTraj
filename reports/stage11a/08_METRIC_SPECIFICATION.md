# 后续评价规范草案

本阶段不计算任何新模型真实ADE/FDE或scene-compliance成绩。以下公式是规范，不是实验结果。统一同source/video/person/20帧支持、同GT及reference/semantic/transform SHA，输出单位原图像素。

对K条完整预测路径，e_kt=||Yhat_kt-Y_t||₂，ADE_k=(1/12)Σ_t e_kt，FDE_k=e_k,12。minADE_K=min_k ADE_k，minFDE_K=min_k FDE_k，独立取最小、不要求同一候选。K1须固定采样/确定性定义，不与K20直接排名。CV只有一条，不作为同候选预算战胜K20的证明；预测像素误差不和ETH米制混排，不同预测时长不排名。

GoalFDE_K=min_k||g_k-Y_12||；必须另存路径FDE。单一终点的“Endpoint ADE”在该时间点等于Endpoint FDE，没有另外12步平均含义；如果评waypoints则明示时间集合。Y-Net原test.py的val_FDE使用waypoint_samples末目标，ADE却使用future_samples完整路径，二者可不相等。PECNet原test_pretrained_model.py先用GT挑best endpoint、才解码一条path，不能把它叫独立minADE_K。将来应同时保留原口径和独立K完整路径重评，不偷偷更换官方数字。

TTST使用10000 proposal再聚类输出20goal；Ke×Ka=20只等于最终候选数量，不等于生成计算预算。所有对照同时记录proposal数、decoder次数、TTST/CWS开关、候选随机种子和耗时。离线GT oracle最小误差不是在线选择能力，不能用GT过滤/重排部署候选。

## Scene Compliance前置条件与定义

先取得与reference逐像素对齐的人工class map、palette、ignore类及质量说明。当前未满足，不填0或假违规率。不可通行类别必须按真实标注任务定义，不默认terrain/road/树冠不可行走。建议structure等经核实障碍单列，歧义类保持unknown，且报告有效语义覆盖。

对每个候选分别记录：终点in-bounds、终点障碍/unknown/boundary；以及全部12点的障碍率、任意点违规路径率、出图率。图像外单独计失败，不能clip到边界后记合规；细路径可能穿越两采样点间障碍，若做连线采样需冻结步长并分开报告。语义边界设置预定原像素不确定带，报告边界和已知内部各自分母；不得用丢弃unknown美化结果。

同K报告全候选合规比例、至少一条合规候选的目标覆盖、候选多样性及ADE/FDE权衡。GT选最准确合规候选只能标oracle；对所有candidate同样计算，不只显示成功者。静态标签一致性不代表真实避碰、安全性、动态路权或人的行为意图。

统计同时保留目标窗口加权、source/video/person先平均后等权及来源录像/物理site宏平均。重叠窗口不作独立样本，若以后给不确定区间按source/person聚类并注明有限source数量。相同背景跨split的泛化范围必须说明。
