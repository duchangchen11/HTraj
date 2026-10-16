# Stage7.1 AgentFormer严格因果准入总报告

**结论：NO-GO，不进入正式训练。** 严格8帧历史上下文与原始数据级先验隔离检查通过，但未能在保留冻结原始后验/KL/loss语义时处理只有历史、未来不完整的邻居。

## 真实执行结果

- 官方源码 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`，5,972,098参数；审计先提交 `e0ae133`，执行配置及代码先提交 `6d07b7f4c697aa79142f383fe4040d133e1531b4`。官方文件未修改。
- 两fold全部Stage5.1训练/验证目标与历史/GT SHA一致；HOTEL验证318目标窗口、ZARA2验证1092目标窗口。上下文只取完整历史8帧，不按未来资格筛选。
- 原始四列数据反事实10/10通过；未来坐标、存在性、长度、未来新增行人和非监督未来变化均不改变历史context、scene_orig、关系mask、K1及固定噪声K20预测，最大预测差0。
- 10个真实工程场景均通过历史先验推理与随机状态strict checkpoint恢复；7个全标签场景通过原始真实loss及反向；3个混合标签场景明确拒绝原训练入口，未伪造监督。
- 两个混合标签补值负例确认mask-only方案不正确，监督q最大变化分别 0.559741、0.499242（各录像明细见03报告），NaN补值loss非有限。
- 最大历史完整上下文73人，超过旧N57；K20推理无OOM，整卡采样峰值 9315 MiB。73人真实原训练显存未确立，不能据推理成功宣称训练可行。
- 完整单元测试 395 passed，0 failed；首次ROS插件启动失败、预期拒绝和NaN负例均保留。

## 核心审计答案

历史context encoder、scene_orig、agent_mask、learned prior及自回归推理不需要邻居未来。FutureEncoder却将所有节点未来token联合attention，未使用fut_mask；KL对所有节点平均且min_clip，重建/sample按全部上下文N归一化。原实现没有仅历史context节点的训练语义。完整GT目标可合法进入训练后验，但未知邻居未来不能作为正常标签。

改loader可以构建严格因果推理输入；要正确训练混合标签场景则必须进一步声明和实现后验、监督集合、KL/输出等语义适配。用户本轮不允许通过核心改动强行准入，因此停止。没有把本阶段工程适配当成原论文复现或创新。

## 交付与边界

7份报告位于 `reports/stage7p1/`；原始JSON、逐窗口人数/ID、来源SHA、loss/梯度、checkpoint SHA、资源、测试和失败日志位于 `results/tables/stage7p1/`。原始反事实副本与随机权重仅在被忽略的 `experiments/stage7p1/cache/`；不上传数据/权重。历史保护清单 `configs/stage7p1/protected_artifacts.json`，最终核验见 `history_and_source_verification.json` 与 `delivery_verification.json`。

本阶段0完整epoch、0优化器更新、0ETH测试；不含正式预测性能，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p1-agentformer-causal-admission
```

执行完成后停止，等待下一步科研决策。
