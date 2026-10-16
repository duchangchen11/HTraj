# 历史研究结论与正式关闭范围

本阶段新正式训练=0、优化器更新=0、ETH评价=0。所有新神经输出均为固定随机权重的人工输入功能验证，不是HTraj真实ADE/FDE或论文性能复现。官方预训练权重没有下载、没有严格加载；严格恢复只指本轮随机state_dict往返。模型源码保持原始Git blob不变，依赖兼容别名、包装器与检查记录独立保存。

Stage1至Stage6.1代码、协议、指标、报告及checkpoint全部保护；不改旧学习率、seed、选择规则或采样比例。下列关闭对象不因换公开主干而悄悄恢复。

|关闭对象|原始证据|本轮决策与边界|
|---|---|---|
|TOHI高阶/图模块|Stage2.6：H3对H2三seed ADE都差，H2独立收益不稳定|停止增加高阶/时间传播结构；不推出所有社会预测必然无效|
|G0/G1社会图原型|Stage4.2最终同预算100轮：ZARA2 G1−G0 ADE +0.091252m，HOTEL −0.046452m，均弱于同目标CV|关闭原型迭代；单录像好不代表稳定独立社会收益|
|SocialVAE社会注意力改造|Stage5.5 Full−Placebo：HOTEL窗口加权2/3seed改善，人等权仅1/3；ZARA2两权重3/3|未通过跨录像/seed门槛，停止社会注意力/关系GRU/GNN追加|
|容量与Placebo路线|Stage5.3 NoSocial结构有效上界比Full少585792，5.4控制函数类不完全相同，5.5内容对照不稳定|已有对照冻结，不再用Full略优NoSocial直接宣称社会信息有效|
|Stage6.1历史速度分层采样|新30轮U/S：HOTEL ADE .395194→.379526，ZARA2 .280626→.287247；HOTEL正常速度偏差下降30.10%，静止FDE .207669→.295994|停止比例/阈值/模型/seed/训练长度搜索；ZARA2-U best=30，有限预算不能证明更长训练必然无效|

来源：reports/stage2p6/STAGE2P6_REPORT.md；reports/stage4p2_final/05_FINAL_RESEARCH_DECISION.md；reports/stage5p5/STAGE5P5_REPORT.md；reports/stage6p1/06_RESEARCH_DECISION.md，各原始表与权重SHA仍保留。不把partial、offline完整20、严格历史上下文三类协议串成同一数值排行榜。

公开模型历史审计也不重复失败：Social-STGCNN维护版节点卷积排列敏感；原Full/NoSocial存在BN/PTN混杂，公平G0/G1不能挽救旧结构。EqMotion官方历史源码/权重不严格匹配，不能用strict=False补齐随机参数。HighGraph未定位到可冻结完整实现。SGCN空间mask沿节点索引卷积，Stage6已有换序反例和train累计边界风险；其TXP节点实际在batch轴的纠正保留，不能继续传播旧静态误判。SocialVAE已证明可训练但社会独立价值未稳定。SingularTrajectory仅源码/组件审查，地图/vector field与协议成本尚未解决。本轮不重新调试这些路径。

旧失败支持“本项目这些协议/预算下旧假设未获稳定支持”，不支持“任何GNN或邻居信息没有价值”。新公开模型首先接受技术准入，再看真实同目标基线，不以论文排名抵消旧负结果。
