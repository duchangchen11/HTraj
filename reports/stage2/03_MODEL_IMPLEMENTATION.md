# 模型实现与冻结设计一致性

日期：2026-10-14。所有源码为本阶段独立实现，仅复用已安装的 PyTorch 环境；没有使用任何旧预测模型代码。

## 四个可运行模型

- **PedBase**：自身速度/加速度/有效性→共享 motion MLP→每人独立 GRU→共享 deterministic displacement MLP。只访问自身轨迹。输入 `[B,8,N,2]`，输出 `[B,12,N,2]`。
- **FirstOrder-GNN**：相同 ego/decoder；一个社会通道只聚合上一帧 ego，不读取邻居的社会通道。保留该一阶状态的时间记忆，采用相同 gate/fusion，使它成为 A 的严格单社会阶对照，不是任意堆叠图层。
- **Dynamic-GNN**：相同 ego/decoder 和候选图；将 ego、邻居上一帧 social hidden 和自身上一帧 hidden 送入共享社会 GRU。阶数混在一个 hidden 内，无按阶隔离。这是本项目的简化动态 GNN 对照，不称作某篇论文的官方复现。
- **TOHI-Net-A**：`ego_encoder.py`→`temporal_graph.py`→`order_state.py`→`order_fusion.py`→共用 `trajectory_decoder.py`。默认 d=64、H=3、k=8；没有方案 B 或多模态实现。

四种模型的 encoder 和 decoder 架构完全一致；初始化顺序也一致，相同种子下共用组件权重相同。参数总数随交互模块不同，因此本次只算统一训练预算的 smoke，不是参数匹配性能研究。

## A1 不变：上一帧快照

实现严格遵循 Stage 1：

$$s_{i,t}^q=\gamma_{i,t}^q\odot s_{i,t-1}^q+\sum_j\alpha_{ij,t}W_qs_{j,t-1}^{q-1}.$$

`OrderState.step` 的 `previous` 固定，循环中的 q 通道只能读取该张量；新通道在列表中写完后 stack。q=1 从上一帧 ego 读取，q>1 从上一帧 q−1 读取。W 无 bias；gamma MLP 的 bias 只控制 carry 系数，不能生成社会状态。无效阶由 Boolean support 单独屏蔽；未接到合法链时始终为零。支持 mask 判断拓扑可达，与 learned 数值非零分开。

第一个观测点没有社会边；缺失/重现、长采样 gap 清零 ego 和社会 carry，序列之间没有外部状态。所有场景独立 flatten index；无邻居时 fusion 精确返回 ego，使共享 decoder 退回 PedBase。

## 图规则的工程定稿

Stage 1 允许原型固定邻居选择，但没有冻结具体距离阈值、score 或 ties 处理。这里补充实现规格：4 米内有效且两端均有连续历史步的候选；按 `D_closest + 0.05*distance` 从小到大保留至多 k=8；tau 采用冻结恒速公式，截断至4.8s。随后 relation MLP 以几何7维+当前目标 ego+上一帧源 ego 产生 alpha，按目标邻域 softmax。候选选择不使用 learned hidden 或未来 GT；它不是已验证最优建图。

边界完全同分时整组保留或丢弃，保证最大邻居数与行人置换等变；不会用数组下标任意选择其中几人。关系特征平均用于 gamma；稀疏 scatter 不构造模型状态的 N×N 高阶幂，但候选几何仍有 N×N 临时张量。

## 原设计中需要记录的边界

1. Stage 1 A2 写的是所有阶 softmax，但未写 invalid 阶的 mask。用户明确要求无链通道无效，本版在 softmax 前 mask；全无效返回 ego。A1 核心递推没有变动。
2. 运动 MLP 额外加入速度/加速度的连续有效性标记，以区别首次出现、缺失和静止；这些是输入缺失处理，不是研究创新。
3. q 指显式社会 hidden 消息经过的边次数。alpha/gamma 的几何控制也依赖邻居；不能把 q 说成全部输入依赖或真实社会因果的阶数。
4. A3 是 gamma=1、W=I 的线性示例；学习 gate 后只保留时间支持，不宣称数值仍等于该多项式。
5. 严格升时会阻止同帧二跳；carry 不增社会阶，跨帧回返 walk 允许。统计显示回返值得消融，但本轮未擅自删除它。

共享 head 直接预测以最后历史位置为锚点的12步位移；没有加入额外 CV shortcut、未来真值、碰撞损失或数据增强。loss 为有效 target track 的平均 L2；ADE=K1 每人有效未来步平均后再平均，FDE 只评最后未来步有标签的目标。部分监督导致 ADE/FDE 分母不同，在结果中保存。

## 复杂度与当前限制

TOHI 状态 `[B,T,N,H,d]`，推理可扩展滚动但当前 debug/训练保留时间张量；社会模块复杂度与 Stage 1 $O(BT H(Nd^2+Nkd))$ 一致，另有候选 $O(BT N^2)$、ego、关系 MLP、decoder。无未来多模态 head。本阶段验证可运行性，不声称高速、创新充分或优于对照。
