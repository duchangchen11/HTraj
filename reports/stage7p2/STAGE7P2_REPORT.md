# Stage7.2 标准ETH/UCY基准对齐总报告

**结论：技术GO，可在下一阶段分阶段进入官方基线复现；本阶段不启动正式训练。** 全部旧实验保留，与新协议指标分开。当前没有可填写的AgentFormer五场景本机性能表。

- 分支 `feat/stage7p2-standard-benchmark-alignment`，基础 `81a2e239e8ee79c07d19e319401c0f96c0b5e7ec`。
- 模型/源码版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`，原Transformer、CVAE、FutureEncoder、learned prior、decoder、loss、DLow全部保留。43个官方源文件版本保护；运行时机械兼容无源码修改。
- 官方74数据文件、8录像/22唯一blob全部补齐，五折train/val/test与20帧完整资格明确；15个原loader入口、148个首/最大packet检查通过，完整窗口清单/原始SHA保存。每fold测试录像不入该fold训练/验证，train/val完整支持无共享frame，但同录像相邻时间相关性不叫独立确认。
- 用户授权协议切换已在c28350f记录：非ETH折允许ETH_train/val；ETH折自身测试不调参。本轮未计算ETH模型测试成绩。旧HTraj因果与部分观测结果不得与新基准混排。
- 官方metric公式、ADE/FDE不同候选赢家、真实四列输出序列化、GT frame/ID对齐及原eval CLI全部验证；主指标独立minADE20/minFDE20，K1只作明确命名辅助。DLow20头不是20次CVAE独立采样，不能用recon目录或旧K1代替。
- 作者517MB ZIP取得，五fold CVAE与DLow全部strict加载。ETH参数6,497,922，其他CVAE5,972,098；DLow各591,872。双checkpoint引用与SHA明确；尚未运行完整作者权重五场景指标，不声称已复现修正版数值。
- 六个真实N1/2/57场景、两阶段共12次原Adam工程更新、0完整epoch；原loss/梯度有限、严格模型/优化器/调度器恢复通过，DLow基础CVAE权重不变，无OOM。N57 CVAE整卡采样峰值9744MiB，后续串行；ETH特殊配置的最大DLow训练显存另需预检。
- 全量单测 407 passed / 0 failed；历史SHA保护见 `preservation_and_versions.json`。作者公开基线最佳论文值单独存 `literature_reported_metrics.csv`，不作为本机实测排名，PECNet候选选择与多人过滤等差异明确不可严格比较。

执行顺序错误完整保留：初稿预算协议已预提交，但补充核验材料commit检查失败后误先启动smoke，全部工程步先于93e8a5a补充提交结束；没有实际SIGSTOP成功、没有重跑或追加更新。不将其追认为全部smoke前完整预提交。详见06报告和两个顺序偏差JSON。下一正式训练需先独立冻结全部配置/选模/停止条件并检查commit成功。

交付七份报告 `reports/stage7p2/`，数据/源码/权重SHA、split/window清单、strict加载、原始loss与梯度、资源、CLI日志、所有失败与测试结果在 `results/tables/stage7p2/`。原数据、PDF、权重、环境、cache不入Git。下一阶段拟先固定作者权重五场景核对，再从头HOTEL/ZARA2、随后ETH/UNIV/ZARA1，完整CVAE100+DLow50轮；原val选模，绝不用test改超参。计划仅提案，实际训练未执行。

用户手动推送：

```bash
git push -u origin feat/stage7p2-standard-benchmark-alignment
```

本阶段完成后停止，等待下一步科研决策。
