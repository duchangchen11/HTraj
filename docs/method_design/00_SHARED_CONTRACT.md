# 候选结构共用协议与符号

日期：2026-10-14；所有候选都是 **HTraj 设计草案，未实现、未训练**。一句话研究论证：通过控制社会传播的阶数和时间相容性，检验复杂人群中的间接影响能否获得可解释、有效率的预测表示；目前仅有文献与计算结构支持，性能及创新性待验证。

## 术语与张量

| 符号/术语 | 固定含义 |
|---|---|
| $B,N,T_o,T_p$ | 批量场景数、场景内最大人数、观察长度、预测长度 |
| $d,k,H,\Delta$ | 隐维、每个目标最大入邻居数、最大社会阶数、事件连接最大帧间隔 |
| $p_i^t,v_i^t,a_i^t$ | 位置、速度、加速度；仅使用截至 $t$ 的差分 |
| $A_t[i,j]$ | 消息由 $j$ 传向 $i$，不等同物理因果方向 |
| 社会阶数 $q$ | 传播链中跨行人的交互边次数，不计个体时间记忆 |
| 时间有序 | 社会边时标严格递增；与统计因果关系不同 |
| $K_s$ | 评价预测样本数，默认计划为 20 |
| TOHI-Net | 候选研究名，不表示已发表、已验证模型 |

输入 `position[B,T_o,N,2]`、`valid[B,T_o,N]`、时间戳；不同场景不建边。变人数使用 padding 和 mask；不提供 $t>T_o$ 的坐标、目标、未来交互标签作为预测输入。输出确定性 `[B,T_p,N,2]`，可扩展概率分布/多样本 `[B,K_s,T_p,N,2]`。

## 共用个体编码与关系特征

以场景坐标系的最后位置为解码锚点，个体输入优先使用位移、速度、加速度和 validity：

$$u_i^t=\operatorname{MLP}_{ego}([v_i^t,a_i^t,\operatorname{valid}_i^t]),\qquad h_{i,t}^{0}=\operatorname{GRU}_{ego}(u_i^t,h_{i,t-1}^{0}).$$

不以 $p_i^t-p_i^{T_o}$ 作为前缀编码输入，否则会引入后续观察时刻。第一有效帧差分为零且标记缺失；断轨不把 gap 当正常一步；时间间隔使用真实秒数。ego GRU 仅更新自身运动，不混入社会消息。

共用关系输入 $f_{ji,t}=[r_{ji,t},w_{ji,t},\tau^*_{ji,t},D^*_{ji,t},\operatorname{valid}]$，其中 $r=p_i^t-p_j^t$、$w=v_i^t-v_j^t$，近似最近接近时间与距离为：

$$\tau^*=\operatorname{clip}\left(-\frac{r^\top w}{\|w\|^2+\epsilon},0,\tau_{max}\right),\quad D^*=\|r+\tau^*w\|_2.$$

这是恒速运动几何启发式，不是真实碰撞预测，也不是 HTraj 原创理论。以距离候选、运动特征和 learned score 选择至多 $k$ 个入邻居，删除自身边；关系 MLP 可以非对称，方向由计算选择定义。top-k 是硬选择、非连续，可在原型中固定邻居选择，并另消融 soft gate。

基于同一候选边定义 $\alpha_{ij,t}=\operatorname{softmax}_{j\in\mathcal N_i^t}\operatorname{MLP}_{rel}(f_{ji,t},h_{i,t}^0,h_{j,t-1}^0)$，空邻居返回全零。所有 graph/gate 仅依赖观察前缀。建图本身需计入成本：朴素两两距离 $O(BT_oN^2)$；采用空间索引的成本另测，不默认忽略。

## 共用解码器与公平性

先用简单确定性多步 MLP 预测位移 $[T_p,2]$ 并锚定 $p_i^{T_o}$，损失为平均 L2 距离或固定 Huber。该 head 在所有交互对照共享规格。后续概率版可用同一 CVAE，训练时 posterior 使用未来真值合法，但测试只能从 prior 采样；或使用二维高斯头，$\sigma=\operatorname{softplus}(s)+\epsilon$、$\rho=\tanh(r)$，优化 NLL。CVAE/GRU/attention 是已有思想，不能列入原创结构。

本阶段选定性诊断优先，概率 head 最终选择待下一阶段核定；不能将概率模型 best-of-20 与确定性 K=1 直接当同一成绩。结构实验固定数据、ego 编码器、解码器、采样数、参数/边预算；任何新增碰撞损失和数据增强必须给所有对照同样使用。

## 结构贡献边界

借鉴：HighGraph 的社会深度问题与按阶比较，UniEdge 的交互模式传播，TGN 的记忆递推，DBGNN 的时间相容路径，通用 GRU/attention/CVAE。重新设计候选：A 的固定社会阶通道递推和阶混合；B 的局部交互事件实例 DAG、阶数有效性及路径组合门控。是否已有同构研究及是否有效均待验证。

来源：[HighGraph](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)、[UniEdge](https://arxiv.org/abs/2502.02504)、[TGN](https://arxiv.org/abs/2006.10637)、[DBGNN](https://proceedings.mlr.press/v198/qarkaxhija22a.html)。详细原文定位见 literature 报告。
