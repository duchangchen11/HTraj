# AgentFormer数据与训练信息路径审计（训练适配前冻结）

基础：HTraj `546c256755139edd59b1ffe09c184bc05260a1e8`；官方 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`。本审计先提交，再编写本阶段数据适配与探针。没有正式训练，也不新增模型。

## 源码定位与信息流

来源是Stage7已按Git blob核验的官方缓存；版本证据在 `results/tables/stage7/sources/AgentFormer.json` 和 `AgentFormer_single_retry.json`（两份取成功记录并集），上游路径：[官方版本](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。以下行号针对该版本。

| 路径 | 输入、计算与依赖 |
|---|---|
| `data/preprocessor.py:80–88` | `get_valid_id`同时要求min_past与min_future存在；8/12配置使上下文成为完整20帧离线集合。未来存在性会改变历史上下文。 |
| `data/preprocessor.py:135–153` | `FutureMotion`在后续缺失处复制前值；第一未来帧缺失直接报错。不是任意缺失标签可用的训练后验。 |
| `model/agentformer.py:504–583` | `set_data`同时接收历史与未来。`scene_orig`取历史所有位置均值；`agent_mask`只用当前历史位置，当前配置默认稠密零mask。未来张量虽在接口内被计算，推理路径不需要其值。 |
| `model/agentformer.py:128–168` | 历史场景中心坐标与相邻位移→共享线性映射→时间编码与两层Transformer→历史token及逐人context。不要求未来。 |
| `model/agentformer.py:209–256` | FutureEncoder从每个节点未来位置/速度构造12N个token；未来token间self-attention、对历史的cross-attention；生成每人q(z\|X,Y)。两类attention mask来自`agent_mask`，**不读取`fut_mask`**。 |
| `model/agentformer.py:395–429` | learned prior p(z\|X)仅来自历史context；训练decoder用q采样，推理用p采样；共享原始自回归decoder。 |
| `model/agentformer.py:589–608` | forward总是调用FutureEncoder；sample loss另走历史先验K20。inference(mode=infer)不调用FutureEncoder。 |
| `model/agentformer_loss.py:1–46` | MSE/sample支持逐时刻`fut_mask`，但均以全部N节点归一化；KL无目标mask，对所有节点求和/N后执行min_clip。ETH/UCY的`pred_mask=None`不提供训练监督子集。 |

信息流：历史X→历史Transformer→context→learned prior与预测decoder；训练标签Y→未来Transformer（跨所有节点）→q→训练decoder，同时q与p进入KL；先验K20轨迹另与Y计算best-of-20平方误差。MSE、KL及sample都是原始训练目标，不称为精确边缘NLL。

## 无完整未来邻居的关键限制

历史编码、关系mask、scene_orig、先验与自回归推理只要求完整历史。原始训练后验则为全部上下文节点建立未来token与潜变量；没有独立的“仅历史context节点”类型。给无标签邻居设置`fut_mask=0`只影响误差项，不能阻止其补值影响监督目标的q，也不能移除该邻居KL。

设上下文C、完整监督集合S⊂C，原MSE为 `sum_{i∈C,t} m_it||Y_it−Ŷ_it||²/|C|`，sample亦按|C|平均，KL为 `max(sum_{i∈C} KL(q_i||p_i)/|C|,2)`。仅给C\S的mask置零既未消除后验依赖，也没有保持原完整监督目标的平均权重。NaN乘零仍为NaN，不能把未知坐标填零当作真实标签。

若要让仅历史邻居参与历史attention而不参与无效未来监督，需要明确改造未来token/attention、后验查询与KL节点集合以及归一化（还要说明输出节点语义）。这是模型训练语义适配，不是纯loader修复。训练时删除这些邻居则丢失已声明的因果上下文，退回A也不等于B。

## 本阶段边界与待实测项

不改官方Transformer、attention、decoder、loss或KL。先实现仅历史上下文与分离标签；原始loss入口在任一上下文未来不完整时明确拒绝。推理接口使用NaN哨兵表示不可用未来，仅供infer，绝不作为正常标签。

后续仅验证：冻结目标/GT SHA；原始四列数据级未来反事实；真实单/多人推理；**所有因果上下文均有完整标签的真实窗口**原loss/反向/严格恢复；混合标签窗口的明确拒绝及补值影响负例；真实最大人数资源。负例补值明确标成无效训练诊断，零优化器更新。保留Stage7 N1/2/5/12、learned-prior、排列、GT隔离与N57原证据，不重写其结果。

**预审结论：仅数据适配预计不能通过协议B原始训练语义准入；最终以真实缺失标签覆盖与负例确认。** 若确需改后验/KL才可训练，最终No-Go并停止，不实施这些改动。不把工程准入或随机权重输出写成预测性能。
