# 候选架构比较与开发选择

2026-10-14；判断基于原文结构与工程分析，不是实验排名。

| 维度 | A：阶数分离递推 | B：交互事件 DAG |
|---|---|---|
| 状态单元 | 行人×社会阶数 | 带时间的有向交互事件×路径阶数 |
| 时间传递 | 所有阶读取上一帧，carry 不增社会阶 | 前事件目标接后事件源，严格升时且有时间窗 |
| 相容链 | 隐式汇总，端点状态压缩 | 显式局部事件连接，汇总相容前驱 |
| 形状 | `[B,N,H,d]` 滚动状态 | `[M,H,d]` 和 `[2,E_D]` |
| 参数 | 按阶 $W_q,g_q$、共享时间 | event embedding、按阶组合 gate 与 attention |
| 社会传播成本 | $O(BT_oH(Nd^2+Nkd))$ | 默认联合 MLP $O(HE_Dd^2)$；可研究低秩版本 |
| 建图成本 | 朴素 $O(BT_oN^2)$ | 同左，另索引 $O(M\Delta k)$ |
| 内存 | 节点级；易做在线滚动 | 事件与连接级；更高 |
| 主要错误风险 | 同帧读写污染、混阶、空图 | DAG 索引、valid mask、复杂度低估 |
| HighGraph 差异 | 跨时间分阶递推，不取同帧图幂 | 事件链，不取行人邻接幂 |
| UniEdge 差异 | 无 patch 双图与 Hodge | 有向时间相容事件 DAG；无 Hodge |
| 最接近已有思想 | TGN、动态图 GNN、按阶组合 | line graph attention、DBGNN、时间图路径 |
| 创新潜力 | 可控增量，可能被视为工程组合 | 结构更可辨认，但通用路径原理已有先例 |
| 实现难度 | 低至中 | 中至高 |
| 当前定位 | 第四章优先实现 | 独立小论文条件候选 |

## 审稿人角度的技术评估

**问题贡献**：HighGraph 已提出间接社会影响，UniEdge 已提出高阶跨时间和 E2E。HTraj 不能重复这个动机后直接宣称新问题。可检验问题是严格时间路径及社会阶数隔离是否提高交互表示质量，特别是在不同密度、等待和链长下。[HighGraph §1/4](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)、[UniEdge §III](https://www.hubertshum.com/publications/tcsvt2025trajectory/files/tcsvt2025trajectory.pdf)。

**结构贡献**：A 有明确依赖约束但容易退化为动态 GNN 增加通道，需同参数阶数混合对照。B 的有向事件接力与组合 gate 较明显，但 [DBGNN](https://proceedings.mlr.press/v198/qarkaxhija22a.html) 已研究 time-respecting paths；不能声称首次时间有序高阶传播。[TGN](https://arxiv.org/abs/2006.10637) 已研究记忆式时间事件表示。

**实验可信度**：两者均未训练，不给分数预测或发表保证。先证明依赖结构实际执行，再看密度/交互分层与端到端成本。若收益来自更强 decoder、更多参数、不同数据划分或更大采样数，不能支持传播机制贡献。HoGAT 全文仍是未解除的创新风险。

## 开发决策和转向条件

1. 下一阶段先实现 A 的 ego encoder、稀疏关系、双缓冲分阶传播、同一确定性 head；用户下达阶段指令后开展。
2. 人工图若无法满足顺序/阶数支持，先修正公式或实现，不训练。
3. A 若与普通 GNN+GRU 等预算对照无稳定差异，收缩“创新”表述并分析严格时序是否过强；允许同帧短链只能作为另一个明确变体。
4. A 有可复现机制与预测收益、B 文献核对后差异仍存在，再开发 B 的事件索引与相容消息原型。
5. 第四章以 A 为起点并可加入 B 升级，不要求现在锁定最终发表模型；第五章方向保留。

推荐 **A 先做，B 后评估**。这是工程可控性与论文证据风险的平衡判断，不是文献已证明 A 优于 B。
