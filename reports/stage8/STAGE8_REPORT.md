# Stage8 选题与主干总报告

**最终NO-GO。当前未选定Resonance或MoFlow作为小论文创新主干。** AgentFormer保持已复现公开基线；不再连续改造它，不引入第三模型，不启动正式训练。

Resonance版本`7358b3d4fd3fcf160c18be9425445a67ee8211ee`，Qpid/Luna子模块分别冻结，MoFlow版本`f1b89b0e80ce95646214923e271e3c280d305cfa`。所有源码/权重/原数据与日志独立保留，旧Stage1–7.6结果不修改。Stage8协议先于工程执行提交`19c19d9`。

实际执行：Re原网络4个真实官方validation batch（N1/2/5/42）K20前向及损失反向、随机权重strict恢复；ZARA2作者Full权重strict加载并做42人validation前向；MoFlow A1/A2原teacher backbone CPU人工前向、teacher/student计数。没有正式epoch、参数更新、MoFlow完整采样/蒸馏或test评价；没有新的ADE/FDE基准成绩。

找到一个明确结构现象：Re在同角分区分别平均中心化运动谱与位置，可能抹除“谁在什么位置以什么运动前进”的绑定。作者权重人工交换见证汇总差约3.58e-7；真实历史5203目标窗口中3869满足预设的潜在歧义条件。该统计不是错误率/独立行人比例，更不是修复收益。仅用联合矩/联合MLP修复还不足以通过新颖性，近年动态交互/Rev等方法已覆盖相关思路。

协议差异不能回避：Re ETH6帧间隔、Luna val=test、univ13和邻居资格不同；MoFlow original变A1且原转换排除单人窗口，main trainer默认val=test；其LED是另一数据版本。原公开成绩不能与AgentFormer直接严格排名。原权重ZARA2与本轮validation也可能训练曝光，功能测试不能称独立泛化验证。

原Re约3.15M参数，小batch资源可行；MoFlowteacher约4.70M/student约4.09M、还有离线样本与两阶段训练。两个新模型的完整训练成本均未实测。由于创新性与公平实验链未通过，没有为填满GPU预算继续跑。

## 报告与证据

1. [近期研究查重](01_RECENT_RESEARCH_REVIEW.md)
2. [Resonance技术审查](02_RESONANCE_TECHNICAL_AUDIT.md)
3. [MoFlow技术审查](03_MOFLOW_TECHNICAL_AUDIT.md)
4. [ETH/UCY协议对比](04_ETH_UCY_PROTOCOL_COMPARISON.md)
5. [创新问题与否定条件](05_INNOVATION_PROBLEM_SELECTION.md)
6. [计算与训练成本](06_COMPUTE_FEASIBILITY.md)
7. [最终NO-GO](07_FINAL_GO_NO_GO.md)

结果均位于`results/tables/stage8/`：文献/协议矩阵、源码及权重SHA、作者args核查、真实工程前向/梯度/资源、validation目标ID与来源SHA、配对歧义见证、失败记录、完整测试与历史保护。

Inter-GraphFormer完整方法仍不可得，明确记录未核实；不从摘要推导它缺少某机制。完整单元测试与最终历史校验数量以unit_tests.xml、historical_preservation.json、delivery_checks.json为准。

完成本地Git提交，不自动push。用户命令：

```bash
git push -u origin feat/stage8-topic-and-backbone-selection
```

完成后停止，等待用户科研决策。
