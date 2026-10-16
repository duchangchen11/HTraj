# 最终准入：NO-GO

| 条件 | 证据/状态 |
| --- | --- |
| 原始来源/manifest/监督目标及GT SHA一致 | PASS |
| 原始数据未来反事实10项全部通过 | PASS |
| 真实单人/多人完整标签原loss反向及推理 | PASS_ON_FULLY_LABELED_CASES_ONLY |
| 历史完整、未来不完整邻居的原语义训练处理 | FAIL |
| 原始训练loss/梯度对全部因果场景有限且有定义 | FAIL_FOR_MIXED_LABEL_CONTEXT |
| 原始网络/loss/参数无未经声明修改 | PASS_UNMODIFIED_BUT_NO_MIXED_LABEL_TRAINING_SOLUTION |
| 真实batch资源可运行 | PARTIAL_K20_INFERENCE_AND_SMALL_FULL_LABEL_BACKWARD_ONLY |
| checkpoint严格恢复 | PASS_10_RANDOM_STATE_CHECKPOINTS |

严格历史数据适配与先验推理隔离成立；**冻结AgentFormer原始联合训练在本协议B下不成立**。主要阻断是未来不完整上下文的后验token、KL及目标归一化语义，非发现先验偷读未来坐标。HOTEL全部94个有目标验证窗口均出现该条件，不能用“少数边界样本”略过。

真实负例在监督mask为0时仍观测到监督目标q对未知邻居未来补值的依赖，两录像均成立。NaN失败、拒绝执行与完整标签成功分开记录。没有将缺失GT当零标签，没有以过滤邻居、丢弃困难目标、退回完整20帧或改loss强行GO。

如果未来考虑仅监督目标后验查询、缺失future token屏蔽、KL/归一化重定义或context-only节点潜变量等处理，均需新的模型语义审计和用户阶段授权；本轮**未实施或验证**这些方案，不推荐直接正式训练。也没有重选其他公开模型。

保留Stage7全部结构、learned-prior、排列一致性、GT数值隔离和N57证据。本轮只补足原始数据/真实batch检查，不把此前人工通过重写成真实因果训练成功。无正式训练、无optimizer更新、无ETH评价、无新创新模块；所有随机权重输出仅验证工程性质，未报告ADE/FDE或论文复现成绩。

完成报告、历史SHA核验、全量测试和本地提交后停止。后续科研决策交给用户。
