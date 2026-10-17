# Stage7.6 研究总报告

**最终判定：NO-GO。停止当前AgentFormer因果上下文创新路线，等待用户科研决策。**

在`feat/stage7p6-causal-context-research-decision`上完成历史证据复核、2022—2026论文/源码核查、C/S训练目标推导、协议公平性及资源审计。没有模型修改、新预测推理、优化器更新、正式训练或测试指标重算。

真实问题已确认：历史可见邻居可能缺完整future；原AgentFormer联合FutureEncoder不能靠重建mask消除未知future影响。未来GT数值、未来存在性资格、训练后验缺标签是不同问题，Stage7.1原NO-GO保持不变。复用的10个原始数据反事实为严格历史适配器prior测试，不能充作官方loader严格历史准入证据。

已有Trajectron++、SocialVAE和TranSPORTmer分别提供目标/上下文分离或可用性mask的实质先例；一般masked/partial后验也非新思想。SocialTrans与Inter-GraphFormer补充了可获取的正文结构，但完整mask/KL/监督机制及作者源码仍未核实。没有可以宣告创新性通过的新机制。

可以写出仅用Y_S、对U潜变量prior补全的合法变分目标，不必补造未来；但变分族可能很松，原sample/clip工程目标不是精确NLL；后验、agent-ID映射、KL和分母须显式变化。还存在非随机缺失与未来资格选择偏差，不能声称已解决全体行人的真实未来预测。

标准ETH/UCY五折、8/12、0.4s、米、DLowK20主协议可保留；C8是独立辅助上下文协议，需匹配重训参考和公平消融。不同上下文资格不能混榜。旧test已观察，不能称为未来未接触确认集。

Stage7.4原训练预算HOTEL100+50约18.43h点估，五折原估60.0–129.5h；Stage7.1最大73人只运行prior推理，原57人训练整卡已到9677MiB。新混合标签训练资源**未建立**，不编造OOM，不删拥挤窗口。当前存储足够，不重复已解除的SSD阻碍。

## 交付索引

- [问题与反事实边界](01_PROBLEM_DEFINITION.md)
- [近年文献及源码矩阵](02_PRIOR_WORK_AUDIT.md)
- [数学自洽性、token与latent语义](03_MATHEMATICAL_FEASIBILITY.md)
- [主基准与历史上下文公平对照](04_PROTOCOL_COMPARABILITY.md)
- [实测预算、73人风险与消融成本](05_RESOURCE_FEASIBILITY.md)
- [门槛逐项判定与方向调整](06_FINAL_RESEARCH_DECISION.md)
- `results/tables/stage7p6/`：数据/代码SHA、全部/有监督窗口双口径、future长度直方图、反事实/失败探针复核、文献矩阵、解析检查、资源情景、测试及历史保护记录。

配置预先冻结提交`a27cf62`；分析基于Stage7.5 `a00c9a38793135b77635ee4f99862f1f017eee23`。实际脚本以execution_record与交付SHA索引为准；同一提交中无法自包含最终Git SHA，最终提交由交付时git log给出。所有缓存PDF/第三方源码/权重保持忽略，不进入Git。

研究执行没有优化器更新或新增数据推理；完整回归测试中的既有人工前向/后向和优化器检查不属于正式研究训练。本轮完整测试442通过（含9项新增解析/审计测试），详见unit_tests.xml/txt；历史SHA核验和最终交付检查见historical_preservation.json与delivery_checks.json。没有把解析高斯例或单元测试当作真实轨迹性能。

只完成本地提交，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p6-causal-context-research-decision
```

完成后停止，不自动开启Stage7.7。
