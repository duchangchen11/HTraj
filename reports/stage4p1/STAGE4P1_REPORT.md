# Stage4.1：从头训练与社会结构基础归因

## 完成情况

分支feat/stage4p1-social-training，基于c0494e8aaba4bcd350d2efa75ba9a6f675d9146f。官方维护版333d3a57b4d2705e129b21aefefa09c79b2b9ae1源文件未改。Full与修订NoSocial均从头完成seed17、最多30轮（包含5轮功能验证），相同数据/损失/优化器/预算；ETH封存，没有push或新论文模块。原失败NoSocial及全部旧模型完整保留。

协议A：完整8/12、0.4秒、米制，完整20帧离线上下文至少2人。四个非ETH录像原train时段1488窗口/23349目标；留出ZARA2整录像921窗口/5833目标。B只生成独立历史上下文manifest，不训练，不与A混报。数据和SHA见01协议与configs/stage4p1。

## 真实结果

| 模型 | checkpoint/epoch | NLL | K1均值 ADE/FDE | K1首样本 ADE/FDE | K20独立最优 ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| Full | best/30 | -0.789741 | 0.788508 / 1.540245 | 1.229007 / 1.984716 | 0.525156 / 0.680384 |
| Full | last/30 | -0.789741 | 0.788508 / 1.540245 | 1.229007 / 1.984716 | 0.525156 / 0.680384 |
| NoSocial-ConsistentNorm | best/30 | -1.332365 | 0.759140 / 1.430127 | 0.910583 / 1.595987 | 0.467411 / 0.749415 |
| NoSocial-ConsistentNorm | last/30 | -1.332365 | 0.759140 / 1.430127 | 0.910583 / 1.595987 | 0.467411 / 0.749415 |

按预设NLL选定checkpoint的K1均值，NoSocial对照的均值ADE较低；NoSocial−Full差ADE -0.029368米、FDE -0.110119米。**按预设NLL选择时Full未超过NoSocial，当前协议未建立独立社会收益**；Full只在K20 FDE单项较低，不能据此宣布社会有效。N5–8组Full有条件优势，N≥9组明显退化，全部报告。仅一个已曝光录像、一个训练seed，无跨场景稳定性或独立确认结论。K20分别最优，不能与K1混排。所有目标严格配对，重复窗口不独立。

## 关键发现和失败

- Full从头训练流程可用，5轮checkpoint恢复预测完全一致，完整30轮指标/梯度有限；当前预算不证明充分收敛或论文性能。
- 官方checkpoint行人重排后，64窗口235目标平均预测差0.266736米、最大1.917565米；K1 ADE增加0.046619、FDE增加0.106658。ST-GCN保持约1e-7数值等变，首个TXP-CNN节点卷积打破等变。
- 单位图不是NoSocial：节点卷积和训练BN仍传递跨人信息。修订NoSocial在train/eval具备零跨人Jacobian、排列一致性、相同7563参数；原版未调用1309参数也保留。
- 原NoSocial第5轮评价非有限NLL，停止、保留4个checkpoint和1–4轮记录。诊断支持训练/评价统计不一致引发尺度放大；只做一次预先记录的修订，新目录从头训练，未修改Full/lr/数据/NLL/预算。不能隐藏失败或把修订称完全初始预注册。
- Full/NoSocial改变了多条社会路径及规范化/核几何，所以**不能独立证明社会图收益**。没有随机关系重训练、graph-only或多录像、多seed实验；不填假指标。

## 资源和验证

| 模型 | 训练秒 | 30轮验证合计秒 | 窗口/秒 | 目标窗口/秒 | 峰值allocated MiB | 最小均值ADE epoch（仅描述） |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 569.416 | 78.494 | 78.396 | 1230.156 | 16.707 | 29 |
| NoSocial-ConsistentNorm | 709.685 | 94.998 | 62.901 | 987.015 | 16.907 | 30 |

训练时间、评价时间单列；两主模型GPU串行；原失败试验只有已日志记录轮次的精确耗时，未捕获第5轮完整计时，不补造总时长。完整219项测试通过；GPU吞吐、checkpoint SHA、数据来源hash和执行commit保存到结果。历史1251项资产SHA逐项核验，包含全部Stage1–4跟踪文件、旧checkpoint、官方源文件/权重；数据、缓存、环境、checkpoint不进入Git。

## 研究结论

可以继续把该公开维护版作为工程基线；还不能直接设计/宣称新论文方法。下一步先审查checkpoint规则与规范化/优化波动，再进行控制运动表示/规范化/节点顺序路径的图独立消融，并核查固定预算收敛；不添加复杂模块。任何单录像收益均为条件性开发证据，ETH不用于选择。

## 交付索引

- 01_TRAINING_PROTOCOL.md：冻结协议、公式、张量形状、筛选与单次修订。
- 02_PUBLIC_BASELINE_TRAINING.md：全部实际指标、曲线、耗时、检查点和失败。
- 03_PERMUTATION_AUDIT.md：完整人数分组和逐模块定位。
- 04_SOCIAL_ABLATION.md：严格无跨人路径的对照、配对与解释混杂。
- 05_RESEARCH_DECISION.md：八项判断、停止规则和下一阶段条件。
- results/tables/stage4p1/：真实配置索引、数据hash、逐epoch指标、目标/行人配对、测试、日志SHA及delivery_checks.json。
- experiments/stage4p1/：本地不可覆盖epoch checkpoint与原日志；Git忽略。

完成本地commit后停止。用户自行运行：

```bash
git push -u origin feat/stage4p1-social-training
```


## 选择规则敏感性：事后描述，不改变正式checkpoint

从已记录的30轮曲线可见，Full最低K1均值ADE在第29轮：0.690975/1.317263米；预设NLL则选择第30轮：0.788508/1.540245米。NoSocial两种规则均指向第30轮：0.759140/1.430127米。因此**若事后按ADE挑轮次，K1相对排序会反转**。没有重新训练、没有重新评价第29轮、没有替换正式best或配对分析；下面只抄录原曲线，不能当作另一项预注册独立验证。

| 模型 | 规则 | epoch | NLL | K1均值ADE/FDE |
| --- | --- | --- | --- | --- |
| Full | preregistered_NLL | 30 | -0.789741 | 0.788508 / 1.540245 |
| Full | posthoc_min_K1_ADE_descriptive_only | 29 | -0.716118 | 0.690975 / 1.317263 |
| NoSocial-ConsistentNorm | preregistered_NLL | 30 | -1.332365 | 0.759140 / 1.430127 |
| NoSocial-ConsistentNorm | posthoc_min_K1_ADE_descriptive_only | 30 | -1.332365 | 0.759140 / 1.430127 |

主结果“Full未超过NoSocial”仅适用于本阶段**预设NLL选择的checkpoint**，不能推广为所有轮次或所有选择策略均无社会收益。结果敏感性、单录像单seed以及归一化/核几何混杂，共同使当前结论应为“独立社会图价值尚未建立”，而不是“社会信息无价值”。NLL与累积轨迹误差的目标差异、Full训练顺序对running统计的影响都是待验证解释，未运行反事实，不能写成确定机制。

下一阶段应先固定与研究主指标一致的checkpoint规则，并审查规范化/优化波动，再做真正单因素图消融；不在本轮事后改规则制造优势。上述描述性低值没有替代正式NLL选择和所有失败记录。
