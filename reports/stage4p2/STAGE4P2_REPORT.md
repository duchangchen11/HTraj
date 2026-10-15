# Stage4.2社会图归因总报告

G1仅在biwi_hotel整体ADE改善，在crowds_zara02未改善；属于单seed、录像相关的条件性现象，不支持跨录像稳定社会收益。

|验证录像|模型|ADE选中轮|best ADE|配对FDE|last ADE|last FDE|
|---|---|---|---|---|---|---|
|crowds_zara02|G0|30|0.661059|1.274254|0.661059|1.274254|
|crowds_zara02|G1|30|0.713563|1.344462|0.713563|1.344462|
|biwi_hotel|G0|30|1.227714|2.162568|1.227714|2.162568|
|biwi_hotel|G1|30|1.180636|2.136477|1.180636|2.136477|


四个独立seed17 run均完成30轮；先5轮smoke再从相同checkpoint继续，G0/G1各1553参数，同逐人归一化、相同初始化、相同解码/损失/训练预算，唯一主要差异为0.25强度显式邻居聚合。ETH封存，旧Stage1–4.1全部保留。G2只完成结构/图统计与G1固定权重辅助换图；未独立训练。seed42/2026未启动。

旧Full29/30端点已重算；固定权重交换BN统计使ADE显著变化，说明不能把NLL下降等同均值预测改善。新decoder不沿节点索引卷积；训练前排列/梯度/Jacobian测试通过，训练后256窗口位置恢复最大误差1.06624032e-06m。模型确实能够使用图关系，但非零消息本身不能证明收益。

本轮按最低验证高斯均值ADE选checkpoint，Stage4.1按NLL选，两阶段不能宣称相同选择协议。辅助首采样与K20各自报告，K20两个最小值独立选择。

|录像|ADE降低(m)|最低要求|ADE达标|FDE增加(m)|FDE允许|FDE达标|
|---|---|---|---|---|---|---|
|crowds_zara02|-0.052504|0.006611|False|0.070208|0.025485|False|
|biwi_hotel|0.047078|0.012277|True|-0.026091|0.043251|True|


本阶段停止追加seed42/2026，不增加注意力、高阶传播或关系记忆。 不把常规图、等变性或规范化当作论文创新。不据曝光录像或重叠窗口宣称独立确认；本阶段只证明/否证当前小型受控家族在固定设置下的收益。

## 交付索引

- `01_PROTOCOL_AND_DIAGNOSIS.md`：预注册与旧结果审计。
- `02_ORDER_INVARIANT_DECODER.md`：公式、张量、参数、复杂度与等变边界。
- `03_GRAPH_ONLY_ABLATION.md`：匹配图对照及G2合法性/混杂。
- `04_TRAINING_RESULTS.md`：best/last、三类指标、预算与耗时。
- `05_INTERACTION_ANALYSIS.md`：全部组、同目标配对、消息强度与推理干预。
- `06_RESEARCH_DECISION.md`：11个科研问题、预注册门槛与停止条件。
- `00_RUNTIME_COMPATIBILITY_NOTE.md`：最初确定性CUDA积分兼容修复及保留的中止记录。

真实结果`results/tables/stage4p2/`包含4×30轮曲线、逐epoch JSON、最佳/最后逐目标压缩CSV、配对/分组/逐人表、120个checkpoint SHA、数据hash、配置、环境、原始运行日志压缩包和测试XML。权重仅本地`experiments/stage4p2/`，不进入Git。

训练前冻结提交`daf1e55975f94c04392a1a68c912131766e72d38`；有效训练代码提交`0a1706d6c6253abeb12fa7d7dccf6748d98e6ed3`。最终交付commit以`git log -1`为准，避免自引用SHA。完整测试、训练代码hash和历史SHA保护结果见`results/tables/stage4p2/final_delivery_checks.json`，前次检查`delivery_checks.json`同时保留。

仅本地commit，不自动push。用户手动执行`git push -u origin feat/stage4p2-graph-attribution`。完成后停止，等待下一阶段决策。
