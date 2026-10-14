# Stage3总报告：接收条件修复与历史关系路由原型

项目/home/lrj/HTraj；分支feat/stage3-relation-routing，基础2d786785813ee298af1cb148999e13a21ffab0c6。保护440项旧源码/配置/报告/结果/测试/文献和权重（其中78个checkpoint），未覆盖旧A0/A-R/H1/H2/H3。新模型与实验独立目录，仅本地提交、不自动push。

## 已执行与主结果

六个新社会变体seed17独立30epoch，180epoch/23040更新；1024/96原manifest、原loss/Adam/batch/clip、原ego encoder/残差/decoder；无额外阶数。冻结PedBase/Dynamic-RN的三seed best/last完全审计，主表仅配对seed17；CV历史速度不足两点时stationary回退，CV ADE/FDE=0.462669/1.146167m。

| 模型 | 名义/扣已知死参数 | best epoch | best ADE/配对FDE(m) | last30 ADE/FDE(m) | 训练秒（旧冻结单列） |
|---|---|---|---|---|---|
| PedBase | 35352/35352 | 24 | 0.441765/1.075548 | 0.447649/1.087694 | 107.43（旧） |
| Dynamic-GNN-RN | 102235/93914 | 24 | 0.437891/1.050755 | 0.442564/1.071946 | 252.50（旧） |
| Receiver-Fix | 73627/65306 | 19 | 0.443099/1.074098 | 0.447179/1.082299 | 276.57 |
| Relation-Routing | 73355/65034 | 20 | 0.442996/1.073757 | 0.451941/1.085896 | 297.60 |
| Routing-Current | 70931/62610 | 20 | 0.443987/1.080292 | 0.447394/1.082781 | 285.20 |
| Routing-NoUpdate | 73355/63306 | 24 | 0.441905/1.073008 | 0.448323/1.090936 | 290.53 |
| Routing-Uniform | 73627/56537 | 30 | 0.442805/1.081710 | 0.442805/1.081710 | 245.77 |
| Routing-Random | 73355/65034 | 23 | 0.442443/1.074205 | 0.444914/1.077513 | 322.16 |

整体1202 ADE/839 FDE目标，全部21历史/密度/链分组和差值保留。140项预训练测试通过后开始训练，最终142项完整测试通过；人工验证新target接收、carry reset、缺失源禁发、不伪造速度、ego精确回退、GT隔离、padding/置换、pair前缀依赖及NoUpdate失效循环权重。训练源c8d151c8c53367ef361d23d30569c9ea5ff8cf52；所有权重保持本地ignored，config/summary保存best/last SHA。

## 归因判断

开发27个一帧目标22个有可靠邻居、65条潜在接收边被旧条件阻断；其中36条邻居关系完整8帧（55.38%），22目标全都有至少一个已知运动的上一帧payload。RF解决结构阻断，但短历史有邻居ADE改善仅约0.35%；History约1.27%，均远低预设10%。两个主要候选完整8相对PedBase略好，2–4/5–7更差，整体ADE不及两个基础，FDE仅略好PedBase而弱于Dynamic。

结构匹配的H1对照更明确：RF一帧有邻居ADE仅好0.000416m，完整8帧ADE差0.002020m。固定H1权重只切接收政策时，一帧有邻居ADE反而差0.009547m；整体略好不能解释为新进入者获益，且是推理干预。完整8帧最大浮点差4.768e-7m，另记录初始逐位断言失败及1e-5m浮点容差，不混淆数值舍入和训练退化。

历史pair平均活跃4.778帧，旧alpha已有运动输入/ego历史；历史重复提供动机，不证明信息价值。History虽稍优于当前小MLP，却弱于同名义参数NoUpdate，Uniform主ADE也更好。人工依赖、非零梯度、固定checkpoint删消息或改alpha不能代替独立训练收益。NoUpdate额外1728循环权重失效；初始run死参数字段只数scorer，完整容量审计另表，不声称完全相同有效容量。

真实/随机关系从头训练对照保持同场景/可靠源/无自环重复，实际边预算少0.41%、约59.90%超4m，存在分布偏移；不能强称关系因果验证。另保存10条明确CPU固定checkpoint辅助干预及252条历史分组状态/门控/熵诊断。

## 研究决定

全部首轮结束后按预提交门槛筛选，0个候选合格，停止第二轮，不盲跑42/2026、不选幸运seed。新模型跨seed稳定性**未验证**。原548验证窗口余下448全与开发96支持重叠，隔离确认候选0，不能称独立验证，ETH继续封存。

当前History-Routing不建议作为第四章正式性能贡献；保留Receiver-Fix契约与失败原型。UniEdge结构与此不同，但Social Attention（ICRA2018）已使用边RNN历史与注意力，不能将新增GRU称创新。下一阶段如获授权，优先单独核查当前可靠源运动先验/消息新鲜度和观测不确定性，再用独立数据确认；不继续无限叠模块，未启动B/五场景/第五章。

## 交付与溯源

七份报告：01_PROBLEM_ANALYSIS.md、02_RECEIVER_FIX_DESIGN.md、03_RELATION_ROUTING_DESIGN.md、04_EXPERIMENT_RESULTS.md、05_ABLATION_ANALYSIS.md、06_RESEARCH_DECISION.md、本报告；00_PREREGISTRATION.md保存训练前规则。结果results/tables/stage3/包含逐run配置/30行曲线/分组/状态/summary、汇总比较、完整容量、负对照、manifest/源码/protocol SHA、pretraining_tests.xml/unit_tests.xml和确认划分。

data manifest SHA c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced；subset SHA 4b4a50ce47b8d8a2fc574a814f80982b602fed237939f7d7f019492be6dc4be1；protocol SHA 8d84a0c8ad96574f95e2a03db8ca5c315bb91d29218be4e4fbaec4d286fcee10。冻结源90d987799d06bef4e213952342da231e93792a41。各分析另存运行commit与脚本SHA。

用户自行git push -u origin feat/stage3-relation-routing。阶段工作结束，只完成本地提交并停止。
