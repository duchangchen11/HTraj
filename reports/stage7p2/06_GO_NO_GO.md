# 验收：GO，进入下一阶段分阶段官方基线复现

| 条件 | 结论 |
| --- | --- |
| 官方数据完整 | PASS: 74 files / 22 Git blobs / 8 recordings |
| 官方划分对齐 | PASS: 15 loaders, 148 packet checks, no held recording in train/val |
| 完整20帧资格明确 | PASS: min8/min12, N1 legal, future existence declared |
| minADE20/minFDE20评价复核 | PASS: independent winners, real serializer/GT/official eval CLI |
| 真实单人/多人训练入口 | PASS: original train function, N1/2/57, 12 updates and restoration |
| CVAE与DLow配置关系 | PASS: fivefold strict two-part loading, sampler semantics explicit |
| GPU可执行方案 | PASS_FOR_STAGED_START: H/Z full N57 measured; remaining fold-specific preflights required |
| 未说明数据/评价差异 | PASS_DISCLOSED: original selector unknown, metric/cohort and compatibility differences documented |

这是**公开维护版基线的工程准入**，不是五场景论文成绩复现成功，也不是新方法有效。下一阶段先HOTEL/ZARA2，其他三fold按各自原配置先资源预检；ETH额外MLP的最大DLow训练显存不能用本轮Hotel配置数值代替。正式计划见 `configs/stage7p2/proposed_full_reproduction_plan.json`，本轮未执行。

标准完整20帧资格已经由用户明确授权，不把它宣称严格历史邻居因果协议。历史GT数值不参与scene_orig的新版本归一化，测试只能评价samples；未来存在性用于官方离线上下文资格属于已披露的任务定义。旧Stage7.1 No-Go针对不同协议，不被抹掉，也不与本协议结果混合排名。

## 执行顺序偏差与可用证据

初始完整预算/协议于c28350f预先提交。随后数据/评价核验已完成并写入文件，但补充提交被新loader日志末尾空行拦下，smoke命令仍被启动；6场景12更新在93e8a5a补充提交前结束。试图SIGSTOP时进程已正常退出，**没有实际暂停/恢复**；早期处置记录描述的是尝试，`commit_order_actual_outcome.json`保存更正。没有重跑或增加预算。

不追认“全部smoke前完整核验材料已提交”。规则、数据、源码没有根据smoke误差更改；实际执行源码SHA与93e8a5a一致的对照见 `execution_source_closure.json`。本轮0完整epoch、没有测试指标调参；用户要求正式训练前提交完整协议/评测材料，在下一阶段开始前必须独立执行并核对提交结果。该工程顺序错误作为明确限制保留，不能在论文中称本轮严格预注册的预测实验。

未发生OOM/非有限训练loss；没有借删除行人、放宽GT资格、替换损失或strict=False通过。12项契约测试、原CLI import/help、所有官方数据/权重SHA与完整单测通过；但没有证明100+50轮收敛、跨seed稳定性或任何性能提升。正式复现如失败应停并分析基线，不增加创新模块。

完成本地提交后停止，不自动push，不在本阶段启动正式训练。
