# 相关工作与传播机制比较

日期：2026-10-14。标记规则：**已核查**指已读取所指来源；**推断**是从已知结构得到的解释；**待验证**包括尚未获取全文、缺乏实验及未证明的创新。此表不是穷尽性系统综述。

## 1. 统一问题和概念

输入历史二维轨迹，预测未来轨迹或概率分布；HighGraph 是交互插件，UniEdge/GroupNet 将交互模块接入具体预测系统。必须区分三种“高阶”：普通图上的多跳社会深度、超边中的多人群体关系、不同时间的相容交互序列。只有路径的社会边次数和边时标都明确，才可讨论时间有序高阶传播。时间顺序约束不等于统计因果识别。

## 2. 原始结构、模块和传播对照

| 方法 | 研究问题与输入/输出 | 原始模块和传递单元 | 时间处理 | 已有创新/HTraj 关系 | 改进与核查范围 |
|---|---|---|---|---|---|
| HighGraph，CVPR 2024 | 历史轨迹支持间接社会影响；模块输出社会特征，宿主输出轨迹 | 碰撞核、多阶邻接、拼接 MLP；行人节点 | 逐帧建图，宿主处理时间 | 社会深度和可插拔高阶融合；HTraj 借鉴问题 | 正文/补充已读；跨帧链不是其邻接幂显式约束 |
| UniEdge，TCSVT 2025 | 跨时间交互与交互之间影响；轨迹→未来高斯参数 | patch 时空节点、GAT、边图/HLLConv、GCN、Transformer | patch 跨时间连接、位置编码 | 统一 ST 和 E2E-N2N 已提出；最关键对照 | 全文与官方代码已查；时间路径约束及代码一致性需专项比较 |
| HoGAT，Transportmetrica B 2025 | 高阶交互及确定性/多模态平衡；历史→两阶段预测 | 空间/时间图、多层高阶 GAT、可变形时空稀疏模块、两阶段 refinement | 摘要明确建空间和时间图 | 堆叠 attention、高阶及稀疏化已有先例 | 仅元数据和获授权转载摘要已核查；全文公式/复杂度未核查 |
| GroupNet，CVPR 2022 | 多种群体规模和关系解释；历史轨迹→关系/预测嵌入 | 轨迹编码、多尺度超边、节点→超边→节点、strength/category、CVAE | 轨迹级编码，不能仅据此断言无动态信息 | 显式群体交互，与社会链不是同一结构 | 正文已读；时间相容事件链不是所述核心建图依据 |
| MixHop，ICML 2019 | 通用图邻域混合；图/节点特征→节点表示 | 各阶邻接特征、拼接、稀疏正则 | 原方法不是专门时间模型 | 多阶混合不能视作 HTraj 原创 | 原始摘要/出版记录核查；不声称轨迹性能 |
| TGN，2020 | 时间事件动态图表示；事件→节点嵌入及下游预测 | message/aggregator、RNN memory、时间图 embedding | 时间事件与记忆更新 | 递推记忆来自已有思想；A 必须加阶数隔离对照 | 原文方法已查；直接移植 memory 不足以构成创新 |
| DBGNN，LoG 2022 | 时间图 causal walk/非 Markov 模式；时间边→高阶图→节点分类 | De Bruijn 图、路径状态、图层、统计阶数选择 | 明确 time-respecting walks | B 的时间相容和路径 lifting 有先例 | 正文方法已查；任务与事件实例化不同，不能称首次路径传播 |

## 3. 代表数学表达及来源归属

HighGraph：$H_m=\widehat A_t^mX_tW_m$（[原文 §4.2，Eq.4](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)）。UniEdge：$H_e=\sum_r\vartheta_r\Gamma_r(L_1)E$，再以边表示调制节点卷积（[接受稿 §III-C，Eq.5–8](https://www.hubertshum.com/publications/tcsvt2025trajectory/files/tcsvt2025trajectory.pdf)）。

GroupNet 的亲和度为 $A_{ij}=q_i^\top q_j/(\|q_i\|\|q_j\|)$（原文 Eq.1）；节点聚合为超边关系表示，再根据关系强度 $r_e$、类别概率 $c_{e,l}$ 和类别函数产生作用：

$$m_e=r_e\sum_l c_{e,l}f_l(u_e),\qquad h_i'=f_v\left(h_i,\sum_{e\ni i}m_e\right).$$

后两式是原文 §4.2 的**机制概括**，不是逐字公式转写。超图 incidence $H\in\{0,1\}^{N\times M}$ 区分节点与群体关系，不能与 HighGraph 的 $N\times N$ 邻接幂混称。[GroupNet 正文](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupNet_Multiscale_Hypergraph_Neural_Networks_for_Trajectory_Prediction_With_Relational_CVPR_2022_paper.pdf)。

TGN 的 $m_i(t)=\operatorname{msg}(s_i(t^-),s_j(t^-),\Delta t,e_{ij}(t))$、$s_i(t)=\operatorname{mem}(\bar m_i(t),s_i(t^-))$ 来自 [原文 §3.1，Eq.1/4](https://arxiv.org/pdf/2006.10637)。DBGNN 以路径作为高阶图状态，时间边 $(u,v;s)$ 与 $(v,w;t)$ 仅在时间相容时构成 causal walk（[原文 §2–3](https://proceedings.mlr.press/v198/qarkaxhija22a/qarkaxhija22a.pdf)）。不把它们的通用原理归给 HTraj。

HoGAT：全文不可用，**不填写猜测公式**。摘要只支持“通过多层图注意力捕获不同阶空间和时间交互”，不能确定 attention 是否用邻接幂、如何跨层融合、稀疏采样数学表达或是否具有严格时间约束。详见 [HoGAT 核查报告](04_HOGAT_VERIFICATION.md)。

## 4. 时间顺序为何可能有研究空间

设只在 $t=2$ 有 $a\to b$，只在 $t=4$ 有 $b\to c$；聚合静态图和合法时间链均允许 $a$ 间接影响 $c$。交换两个事件时间，聚合静态图不变，合法递增时间链却不存在。此例显示结构区分能力，不证明真实人群必遵循该链。

HighGraph 同帧图幂不专门编码这种跨帧链；UniEdge 显式跨时间和边间传播已覆盖很大一部分问题；TGN/DBGNN 更早提出时间事件和路径序列思想。因此，剩余研究空间只能有条件表述为：**能否在行人预测中，用阶数隔离或路径相容结构，得到超过同成本动态 GNN 与受时间 mask 的 UniEdge 变体的收益？**

## 5. 研究假设与风险

| 假设 | 待验证证据 | 风险 |
|---|---|---|
| 分离 1/2/3 阶传播有益 | A 对比同参数 GRU+GNN、无阶通道 | 可能只是增加容量 |
| 严格时间相容降低错误传播 | 同边集正序/逆序、mask/no-mask、前缀一致性 | 历史窗双向关联可能更有用 |
| 事件链保留比端点压缩更有效 | B 对比 A、普通边图 attention、方向 mask 对照 | B 仍聚合多条路径，不保留完整个体路径身份 |
| 稀疏链适合复杂场景 | 密度与交互强度分层、固定边预算测速 | 稀疏选择可能漏掉重要人 |

创新性检索非穷尽。后续应专项检索 TGAT、动态超图、SMEMO、时间路径图、2026 年 patch 时空预测及 PCHGCN。仅进入检索线索的工作不作为已核查结构依据。此前 Stage 1 计划未在本次会话或空目录中提供；本轮以用户给出的 12 项任务和交付要求建立完整执行计划，未假称读取旧计划。

## 6. 文献来源

除上列正文外：[MixHop 出版记录](https://proceedings.mlr.press/v97/abu-el-haija19a.html)、[DBGNN 出版记录](https://proceedings.mlr.press/v198/qarkaxhija22a.html)、[TGN arXiv](https://arxiv.org/abs/2006.10637)、[HoGAT DOI](https://doi.org/10.1080/21680566.2025.2532650)。来源等级、访问失败和固定代码提交见 [SOURCE_REGISTER.md](SOURCE_REGISTER.md)。
