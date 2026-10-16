# 研究问题比较：证据与覆盖程度

## 全阶段证据链及适用边界

|阶段|可核查事实|当前影响|限制/原始来源|
|---|---|---|---|
|Stage1|HighGraph/UniEdge等已研究高阶与社会结构|高阶/时间记忆不是空白命题|docs/literature/；不把结构不同自动当创新|
|Stage2.6|H3对H2三seed ADE都差，H2独立稳定收益不足|旧高阶路线停|reports/stage2p6/STAGE2P6_REPORT.md；partial小开发协议|
|Stage3|Receiver-Fix/关系记忆未通过预定筛选|有效接收不等预测价值|reports/stage3/06_RESEARCH_DECISION.md；不外推所有seed|
|Stage3.5|完整/部分协议与录像隔离明确|先对齐任务而非跨协议排名|reports/stage3p5/STAGE3P5_REPORT.md|
|Stage3.6|冻结ego后社会支路仍未胜同容量个体修正|不能只归因联合训练破坏ego|reports/stage3p6/06_RESEARCH_DECISION.md；非完全同结构联合反事实|
|Stage3.7|原主干已平移等变；方向归一化只改善HOTEL|不能再用去绝对坐标包装新消融|reports/stage3p7/05_RESEARCH_DECISION.md|
|Stage3.8|OLS4/8/Median只在HOTEL改善，其他4录像退化|停止修CV估计器和可靠性gate|reports/stage3p8/05_RESEARCH_DIRECTION_DECISION.md|
|Stage4.1/4.2|节点索引伪关系、归一化混杂被定位，公平图对照跨录像失败|代码可靠性必要但不保证社会收益|原离线完整上下文不能与本轮因果协议混排|
|Stage5–5.4|公开VAE可真训练、因果loader、NoSocial与容量/Placebo对照|成熟基线比原型可靠；单seed迹象非定论|各冻结报告；不同控制函数类/分布不等价|
|Stage5.5|Full vs Placebo HOTEL跨seed/权重反转，ZARA2重复|结束由毫米收益驱动的复杂社会改造|3seed不是3个新数据集，两曝光录像|
|Stage6|冻结K1出现HOTEL运动速度收缩、方向/ID/时间误差可分解|B问题的直接可观测对象更明确|仅描述，不指定唯一原因；不训练新模型|

不能把不同时期的partial、A_offline_complete20、Causal8x12数值串成同一纵向排行榜；它们贡献的是不同层面的证据。当前结论也不是“所有社会信息无价值”：ZARA2有条件收益，但旧路线要求的跨录像稳定性未成立。

## A与B逐项判断

|维度|A：图神经网络社会交互|B：跨场景运动状态与预测泛化|
|---|---|---|
|本项目正向证据|有计算依赖和ZARA2条件收益；无稳定全局增益|多阶段神经HOTEL弱于CV、best很早、固定输出速度偏差明确|
|技术问题可观测性|需先修合法关系、排列、容量和负对照；可辨识性成本高|可量化逐步速度、沿/横误差、状态比例和泛化gap|
|现有文献覆盖|稀疏图/方向图/高阶/社会注意力已广泛研究|运动先验、域对齐、归一化、静动分支也已有研究，绝非空白|
|尚可证伪的不足|在排列正确、预算和容量控制后，真实关系何时有独立收益|只用source train历史定义的状态分布控制，能否减少稳定速度收缩且保留其它录像性能|
|硕士题目适配|符合复杂场景，但旧实验不足支撑主章贡献|跨录像速度/方向与运动状态变化属于复杂场景，符合总题目|
|有限时间的小论文机会|当前低：需重建可靠图基础，且旧收益未重复|相对更可行：已有可审计诊断，但仍缺新方法/独立验证，不能承诺可投稿|
|主要反证|动态图、关系记忆、高阶与公平G0/G1多次不稳定|归一化和长历史CV也已失败，不能把B改名为旧速度gate继续试错|

**优先B，A仅保留为备用科研问题。不是直接批准两个方法开发。** B的实验支持是“存在必须解释的泛化问题”，并不是“某个适应模块已有效”。不要把描述性诊断提升为方法创新或独立因果证据。

## 文献占位与新颖性风险

- [SGCN，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html)：稀疏空间/时间图已存在；本轮源码还暴露排列和训练边界问题。
- [Adaptive Trajectory Prediction via Transferable GNN，CVPR2022](https://arxiv.org/abs/2203.05046v2)：跨域图迁移与域对齐已有研究；应区分是否访问目标域资料。
- [Towards Single Source Domain Generalisation...，CoLLAs2023](https://proceedings.mlr.press/v232/huang23a.html)：CV运动先验加神经残差、source-domain泛化已存在；普通CV残差不能作为HTraj创新。
- [SingularTrajectory，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.html)：统一运动表示、静动分支、尺度方向处理、adaptive anchor已有实现。domain adaptation任务与无目标域资料的泛化须分开。

上述为聚焦检索，不足以量化哪一方向“论文数量更多”；没有穷尽检索就不编造覆盖率或新颖性分数。A/B均拥挤，当前只是B有更明确本地失效测量。后续若提出具体方法，必须进一步对该技术命题检索，并通过额外非测试录像/数据确认。当前没有一个已完成的、具备可信新贡献和独立验证的小论文方法。
