# 两套协议的真实基础对照

训练源5771f31df0ae863daf70307a9f9b6c73f232ee68，students003原验证区70窗口，其它四recording原训练区；所有新神经模型seed17重新初始化，没有旧权重迁移。A2058、B2365有效训练窗口，不是再跑原1024/96。Adam0.001/batch8/clip1/d64/原per-track L2/30epoch统一；每协议自己的开发ADE选best，同epoch配对FDE与last30都保存。

| 协议 | 模型 | 参数 | best epoch | best ADE/FDE(m) | last30 ADE/FDE(m) | 训练秒 |
|---|---|---|---|---|---|---|
| A_complete8x12_history_context | CV | 0 | 解析 | 0.727884/1.627788 | 不训练 | 0 |
| B_partial_mask_v1 | CV | 0 | 解析 | 0.635479/1.692958 | 不训练 | 0 |
| A_complete8x12_history_context | PedBase | 35352 | 1 | 0.707297/1.524885 | 0.720126/1.563693 | 216.52 |
| A_complete8x12_history_context | Dynamic-GNN-RN | 102235 | 3 | 0.721141/1.560097 | 0.751859/1.623931 | 480.12 |
| B_partial_mask_v1 | PedBase | 35352 | 11 | 0.617815/1.614656 | 0.632383/1.653170 | 241.31 |
| B_partial_mask_v1 | Dynamic-GNN-RN | 102235 | 12 | 0.615008/1.611574 | 0.659313/1.716698 | 551.43 |

A评价706个完整12步目标，B1464个有效future目标/FDE986个，二者平均值**不能直接横比模型优劣**。参数PedBase35352、Dynamic102235（其中单channel scorer8321已知不活跃），容量未匹配，GNN优于Base也不自动证明独立社会价值。

CV用最后两个连续历史位置差分，缺失速度明确stationary回退。米坐标/0.4秒一致。所有神经run/config、30行epochs、best-last targets/groups/summary与本地ignored权重SHA保存。GPU index_add浮点不逐位稳定，无重新挑seed/epoch口径。

## 同一完整目标的人群控制

| 同706个完整目标 | A ADE/FDE | B ADE/FDE |
|---|---|---|
| CV | 0.727884/1.627788 | 0.727884/1.627788 |
| PedBase | 0.707297/1.524886 | 0.705813/1.543293 |
| Dynamic-GNN-RN | 0.721141/1.560097 | 0.703084/1.540871 |

用tuple(recording,sample_id,agent_id)对齐A与B，共同完整目标每人future_points=12。CV相同输入自身完整历史，结果一致；其它模型分别训练、上下文政策也不同，此比较排除评价人群差异但仍不隔离训练目标/上下文差异，不称A协议独立提升预测。原B全部难例继续作为主结果，未为提升指标删掉部分历史/未来。

B全体ADE较小，并不说明部分观测更容易：同706个完整目标上，B-PedBase ADE约0.705813而非全体0.617815。future标签长度/目标人群对平均值的影响明显；FDE分母又不同，不能只挑其中一项定义任务难度。

## 外部公开参考

Social-STGCNN正式名称、MIT许可、固定源码/数据loader、高斯输出/20样本评价已核查（01_PROTOCOL_AUDIT.md、external_reference_sources.json及supplement/limitations）。API限流后通过锁定Git文件树完成split文件名核查：官方ETH fold还含HTraj未用的crowds_zara03和uni_examples，processed内容/边界未逐项复现。本阶段**没有训练、评价、官方权重运行或公平比较**。它的SGD250epoch、完整20筛选/至少2人/全帧列表与K20输出均不同，不能以这里数字代替其官方结果。

## 旧结果严格分开

B_legacy_1024_96_frozen_analysis目录是旧PedBase/Dynamic三个seed最佳权重的兼容重评：源码逐字节、manifest/subset/checkpoint SHA及主指标<1e-6m核查通过。0新增训练。它只提供旧开发组的seed方向线索，不进入上表，也不冒充新录像隔离实验。
