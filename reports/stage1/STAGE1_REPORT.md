# HTraj Stage 1 总报告

日期：2026-10-14。研究题目：《面向复杂场景的行人轨迹预测关键技术研究》。本阶段输出文献证据与设计草案，**没有训练、实测预测成绩或已证实的新方法贡献**。

## 1. HighGraph 究竟解决什么问题？

解决不同社会深度的间接行人影响：一对人的动作变化可能继续影响其它人。它强调多跳社会作用，区别于群体规模。依据正文 §1、Fig.1；并非所有“高阶”都是超图。[HighGraph 原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)。

## 2. HighGraph 采用什么网络结构？

逐帧位置/运动建立碰撞感知关系图，归一化邻接的多个幂作用于行人特征，用阶权重、拼接和 MLP 组合，再堆叠图层，接入原预测器。它是可插拔普通图模块，核心 $H_m=\widehat A_t^mX_tW_m$（§4，Eq.2–5）。宿主可以含时间模块；准确局限是其图幂没有显式约束跨帧严格有序社会路径。完整分析：[01_HIGHGRAPH_ANALYSIS.md](../../docs/literature/01_HIGHGRAPH_ANALYSIS.md)。

## 3. UniEdge 已经实现哪些高阶交互机制？

统一 patch 时空图将跨时间交互直接连接；E2E-N2N 双图更新交互边与行人节点；位置编码和未来占位 Transformer 处理预测时间。高阶跨时间和边间作用均已发表。[UniEdge 接受稿 §III](https://www.hubertshum.com/publications/tcsvt2025trajectory/files/tcsvt2025trajectory.pdf)。公开代码固定提交默认 Hodge 阶数为 1，静态检查发现该分支不执行边间聚合，需要区分论文结构和具体代码，不能据此否定论文。详见 [02_UNIEDGE_ANALYSIS.md](../../docs/literature/02_UNIEDGE_ANALYSIS.md)。

## 4. 时间有序传播是否有研究空间？

**有条件的空间，尚未证明创新充分。** 同帧多跳、任意跨时间边和严格时间相容链有计算差异，但 [TGN](https://arxiv.org/abs/2006.10637) 和 [DBGNN](https://proceedings.mlr.press/v198/qarkaxhija22a.html) 已研究时间事件/路径。研究问题应缩小为：行人预测中，阶数隔离与事件组合是否比同成本动态图以及受 mask 的跨时间边网络更有效？不能宣称首次时间有序高阶传播或真实因果发现。相关证据见 [工作比较](../../docs/literature/03_RELATED_WORK_COMPARISON.md)。

## 5. 候选网络 A 如何设计？

ego 前缀 GRU → 每帧稀疏有向图 → 每阶社会状态读取上一帧 → 阶融合 → 共享多步 head。核心：

$$s_{i,t}^q=\gamma_{i,t}^q\odot s_{i,t-1}^q+\sum_j\alpha_{ij,t}W_qs_{j,t-1}^{q-1},\quad s_{i,t}^0=h_{i,t}^0.$$

时间 carry 不增社会阶数，跨行人消息增一阶，避免同帧更新污染。隐式汇总合法链，不保留每条路径。借鉴记忆和图消息思想，重新设计阶隔离与读写规则。[方案 A 完整规格](../../docs/method_design/01_TOHI_NET_DESIGN_A.md)。

## 6. 候选网络 B 如何设计？

把每个 $(j\to i,t)$ 关系变成事件 token，仅让 $(k\to j,s)$ 在 $s<t$ 且间隔允许时接到它，形成 DAG。按阶注意力与相容 gate 组合前驱路径状态，再读出至终点行人。

$$e'\prec e\iff\operatorname{dst}(e')=\operatorname{src}(e),\quad0<t_e-t_{e'}\le\Delta.$$

无有效前链时高阶状态必须为零。使用事件身份及局部链组合，不使用 Hodge 谱传播；完整路径仍有汇总压缩。时间路径思想已存在，候选贡献限于具体人际事件结构及其验证。[方案 B 完整规格](../../docs/method_design/02_TOHI_NET_DESIGN_B.md)。

## 7. 两种方案有什么不同？

A 的状态是“行人×阶数”，通过时间 carry 压缩历史；B 的状态是“事件×阶数”，显式保留端点相容连接，成本更高。A 易在线递推；B 更易检查具体事件接力，但没有完整保留全部路径。二者共享数据、ego 与 head，结构差异不能由更强 decoder 代替。[架构比较](../../docs/method_design/03_ARCHITECTURE_COMPARISON.md)。

## 8. 哪一种更容易实现？

A。主要是双缓冲状态、稀疏聚合与阶融合。B 需要事件索引、DAG 批处理、阶有效 mask 和连接预算。此判断是工程分析，非实测工时。A 传播约 $O(BT_oH(Nd^2+Nkd))$；B 默认联合 MLP 约 $O(HE_Dd^2)$，$E_D\le M\Delta k$，二者均另含建图、ego 和 head。

## 9. 哪一种更适合硕士论文第四章？

优先 A：问题、公式、结构验证、消融和成本分析可形成闭环，范围可控。即使提升有限也能形成可信的机制研究。最终第四章应由真实证据决定，不能现在保证贡献成立。[六章框架](../../docs/thesis/THESIS_OUTLINE.md)。

## 10. 哪一种更适合独立小论文的方法？

B 更值得作为条件候选，其事件组合结构比 A 的递推变体更易展示明确差异。但应补全文献并证明超过普通时间有向 line graph，而不是仅换 mask。当前不判断一定可发表；A 若取得清晰增量证据也可能有论文价值。

## 11. 目前的创新性风险是什么？

- UniEdge 已覆盖跨时间与边间传播，不能把这些概念重新归为 HTraj 创新。
- TGN/DBGNN 已有记忆与时间路径思想，A/B 可能只是任务适配。
- [HoGAT 作者摘要](https://trid.trb.org/View/2643271) 已表述多阶 attention 与时空稀疏模块；全文受限，是否有相近机制未知。
- 严格时间约束可能损害合法历史窗双向编码；计算方向不是心理或统计因果。
- 结果可能来自容量、建图核或 head；必须同协议、同预算消融。
- 2026 年后续工作尚未完成系统全文核查，不声称文献穷尽。

## 12. 下一阶段建议先实现哪种网络？

先实现 A 的 **ego 前缀编码 + 稀疏关系 + 双缓冲阶递推**，加共享确定性 head。先验证人工图上的正逆序、阶数、空边、mask 和前缀不变性，再做小规模数据验证，之后才申请/执行下一阶段正式实验。详细消融、指标与统计见 [实验协议](../../docs/method_design/04_EXPERIMENT_PLAN.md)。当前按用户要求停止，不启动训练。

## 交付和研究边界

此前 Stage 1 旧计划没有在空目录/本次上下文中提供，本轮按用户列出的任务建立计划并完成要求的文献报告、双方案和论文框架。HoGAT 全文精读未完成，已明确缺口；其书目与作者摘要核查完成。所有新材料在 HTraj，未触碰或删除旧项目。

提交信息按用户建议使用 `docs: establish HTraj research framework and stage1 study`。仓库初查、提交检查、失败边界和远程核验流程见 [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md)；实际 SHA 与 push 结果在最终交付反馈中记录，避免在同一提交文件中递归写入自身 SHA。
