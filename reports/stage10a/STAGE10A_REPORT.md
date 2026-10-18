# Stage10A：风险感知动态图可行性总报告

**工程机制通过；作为创新模型进入Stage10B正式训练，当前NO-GO。** 未训练新模型，未计算新模型验证ADE/FDE，没有测试集评价。没有设计高阶、关系GRU、注意力或新loss。风险选邻居已见Snapshot2025，旧TOHI也已用DCA排序，不能包装为首次方法。

## 交付与证据

|内容|报告/证据|
|---|---|
|历史失败审计|[01_HISTORY_AUDIT.md](01_HISTORY_AUDIT.md)：Dynamic-RN/高阶、G0/G1/PTN、SocialVAE、Stage9具体负结论及边界|
|因果数据冻结|[02_DATA_PROTOCOL.md](02_DATA_PROTOCOL.md)：四目标集合逐ID/history/GT SHA匹配Stage5.1；raw反事实/Guard；不与Stage9 Complete20混排|
|共同模型及公式|[03_MODEL_DESIGN.md](03_MODEL_DESIGN.md)：E/S/D/R，GRU64、单跳MLP、12步确定性回归，有限H的CPA/DCA|
|结构与失败记录|[04_STRUCTURE_TESTS.md](04_STRUCTURE_TESTS.md)：全仓库501通过，初次API/人工期望失败保留；R/D选边确实不同|
|真实资源/预算|[05_TRAINING_BUDGET.md](05_TRAINING_BUDGET.md)：RTX3080合成CPU/GPU前后向，优化器更新0，最多79人未裁剪|
|查重|[06_NOVELTY_AUDIT.md](06_NOVELTY_AUDIT.md)：六篇原文方法及Snapshot冻结官方源码，未声称穷尽检索或已有方法等于新创新|

数据协议：8/12、.4s、米制，HOTEL/ZARA2旧曝光开发区域；训练排除对应验证录像，其他旧validation区域不混入训练，ETH继续封存。训练目标27,145/23,545，验证318/1,092；28/35个验证source×person ID，重叠窗口不能当独立行人。邻居资格只看历史出现性、允许部分历史和位置-only消息；真实未来只定义监督目标及GT，未进入选边/风险/预测输入。

冻结图radius3m、K8、sigma.5m来自来源训练历史统计，H4.8s。S首次共同可见距离定义、永久槽/visibility；D当前距离；R同D候选池及边数、按历史相对运动外推CPA风险排序。S的首次到达偏置和后续可能在3m外的旧关系已披露；D/R才是严格同候选排序对照。风险不是碰撞概率/真实意图，常速度近似及硬topK均有限制。

四模型名义31,896参数，S/D/R结构有效上界相同；E上界18,968，不能将E与社会分支差异直接认定邻居信息价值。64个预定来源训练窗口中D/R候选及每接收者边数/特征完全一致，23窗口选边不同，2,909/59,924条边替换；仅为机制证据。synthetic K1 fixture选ID20/30的可复核反例另存，不把它当实验K8配置。

GPU B8/N79前后向E/S/D/R约16.47/38.30/30.97/29.97ms，allocated峰值最高约99MiB、整卡前后观测最高744MiB，无OOM。仅估算两录像×四条件×30轮约.65–1.86GPU小时（建议资源预留约3h），未实际执行完整epoch，不保证此为上界。真实GPU smoke总约8.09s、权重逐位未变。完整每步/分位数/参数/源与数据SHA见`results/tables/stage10a/`。

## Stage10B判断

1. 历史协议、mask/前缀/未来反事实、排列和平移等变、有效梯度及资源准入通过。当前检查未发现新的输入未来泄漏；资格用未来标签不等于输入泄漏，结论不超出测试范围。
2. 核心新颖性未通过：风险对距离选邻居已被Snapshot明确提出并比较；旧TOHI已有几何风险排名。现有随机权重测试不能证明风险图改善预测，更不能替代因果社会价值证据。
3. **不建议自动启动Stage10B创新训练**。若以后仅以严格归因/重复检验为目标，仍需单独授权并冻结强CV、ego容量控制、同目标逐人加权、边预算及跨录像失败门槛；不能为论文需要继续堆模块。本轮无新ADE/FDE可报告。

所有Stage1–9研究文件、权重及官方预测按SHA保护；最终结果以history_after.json为准。旧test预测仅不解释字节的SHA保护，没有重算/读取其误差作为选型依据。新阶段代码/报告/轻量证据独立，数据/PDF/cache/checkpoint不入Git。先冻结配置commit61ee113，最终提交与远程SHA以Git交付记录和最终反馈为准。本阶段用户明确授权push新分支，不修改main、不force。

完成后停止，不进入Stage10B。
