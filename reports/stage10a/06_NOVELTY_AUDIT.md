# 风险动态图创新性审计

结论：**风险量和风险选邻居不新；目前不支持把该框架作为新论文算法启动训练。** 文献部分采用nature-academic-search的multi-source-search；学术MCP未挂载，改用网页定位、CVF/作者PDF和官方源码。六篇原文方法均读取，PDF只在忽略cache保存，URL/下载SHA/text状态见paper_access.json。网页CVF fetch403后直接公开PDF下载200；未取得内容的搜索条目及不相干arXiv定位不作为方法证据。按标题/来源合并重复结果。技能附带OpenAlex检索脚本另返回HTTP429，失败日志保留，未反复请求；结论依靠已取得的原文与官方代码。本次是定向查重，不宣称穷尽2026全部论文。

## 最接近的六项工作

|论文/时间/来源|原结构及关系机制（概括）|与本轮的差别及重合|
|---|---|---|
|[Snapshot: Towards Application-centered Models for Pedestrian Trajectory Prediction in Urban Traffic Environments](https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Uhlemann_Snapshot_Towards_Application-centered_Models_for_Pedestrian_Trajectory_Prediction_in_Urban_WACVW_2025_paper.pdf)，WACV Workshops/HAVI 2025，预印2024|§3.3/Fig3/Tab3：社会Transformer+地图cross-attention+CNN。相对运动给tTCA/DCA及bearing变化，风险为三项平方尺度之和的负指数；比较top7距离/风险/无邻居|**直接覆盖本轮研究问题和核心选边思想**。AV2异构道路对象、1s→6s及地图不同于本轮ETH/UCY逐帧图/GRU；差异不足以自动成立新颖性。原文Tab3已报告风险选择改善，不能写“前人未比较风险与距离”。|
|[HighGraph: Higher-order Relational Reasoning for Pedestrian Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)，CVPR2024|§4.1 Eq2–3：运动射线交点C存在时Aij=1/[当前距离·两人到C的距离和]；多阶邻接幂及融合插件|运动/碰撞关系图已有；其射线交点不等于同步有限时域CPA。本轮没有高阶幂，但不能将“不用高阶”当创新。|
|[SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction](https://www.chengjianglong.com/publications/SGCN_CVPR.pdf)，CVPR2021|§3.1 Eq1–5：自注意力密集分数、非对称卷积、阈值稀疏mask、ZeroSoftmax、时空GCN及高斯输出；M=1[sigmoid(F)≥ξ]，A=(M+I)⊙R|稀疏/动态/自适应交互不是空白；本轮为解析选边，没有学习mask。旧SGCN代码结构问题不能证明其研究思想未存在。|
|[Social-STGCNN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf)，CVPR2020|§4/6.1 Eq2：图顶点差的逆L2核；ST-GCN+TXP、高斯NLL。维护版loader用相邻位移建图，不能一概叫纯世界位置距离|本轮D为明确位置距离，R为CPA；没有照搬其归一化/节点卷积，结构简化不构成创新。|
|[Trajectron++](https://arxiv.org/abs/2001.03093)，ECCV2020|§4.1：距离≤接收者类别感知半径建立有向图；按类型聚合邻居状态、edge LSTM、注意力、CVAE与运动学积分|动态节点、相对运动输入、距离筛选早已存在；本轮固定单跳MLP/CPA，不声称首个动态图或方向关系网络。|
|[SocialCircle](https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf)，CVPR2024|§3 Eq1–9：按相对角分区，汇总邻居净位移幅度、距离及方向，嵌入为虚拟时间序列与原骨干融合|位置、速度幅度和方向已有；本轮保留关系配对用于CPA，不重做Stage9交叉统计，也不从配对区别推导预测收益。|

独立CSV保存正式名称、结构、关系式、方法位置、代码及核查层级。HighGraph本轮不提供未经核实的官方repo URL；论文是当前方法证据。SGCN/Social-STGCNN版本沿用旧审计，不声称再次运行。

## 直接查重：Snapshot官方代码

[官方冻结版本](https://github.com/TUMFTM/Snapshot/tree/704e93339c58e15740c786881add06de8a397ba3)，Apache-2.0。`utils/social_features.py`真实实现了相对位置/速度、tTCA=-r·v/||v||²、DCA、collision risk及按risk排序截取邻居；config还有距离/风险选边开关。本轮通过git blob逐位核对文件SHA，未运行模型或移植其预处理。

论文Fig3风险公式为exp{-1/2[(tTCA/δt)²+(DCA/δd)²+(bearing_rate/δα)²]}。本轮为有限H裁剪、DCA平方加线性时间惩罚、未知速度距离回退。没有bearing-rate、使用不同常数或把kernel写成topK图，均不足以宣称新机制。源码部分常数/低相对速度回退与论文不同（如发布代码18/10和论文1.8/.3尺度），原因未核实，不以它们为本项目阈值，更不宣称本轮复现其论文性能。未审计完整AV2调用链，不能由局部代码猜测其未来泄漏。

## 三层结论

已发表/旧项目已有：TTC/TCA/DCA、碰撞几何、相对运动、风险选邻居、稀疏图、动态关系、共享GRU/MLP。准备复用：上述几何与简单消息框架、显式mask、稳定ID排序。尚需验证的研究问题：**在同一严格历史目标协议、相同候选/边数/容量下，解析风险排序究竟是否带来额外预测价值**，以及收益是否超越强CV/ego容量。

这可以成为复核/归因实验问题，但目前没有风险关系的真实预测收益证据，也没有区别充分的数学贡献。不是声称整个端到端网络与某文逐字等价，而是核心思想和比较问题已被明确覆盖。仅换核常数、GRU替换Transformer或更换数据集不支持SCI新颖性。若以后用户只授权机制复核，可另冻结小规模失败门槛；当前不推荐以“新模型创新”名义启动Stage10B。
