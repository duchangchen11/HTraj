# Stage 4.2：训练前冻结协议与Stage4.1审计

本文件和`configs/stage4p2/protocol.json`必须先提交，再启动新增模型训练。基础提交为`ce77c49c68367b169bc6d8f7a29bb4cba4400575`，官方维护版冻结提交为`333d3a57b4d2705e129b21aefefa09c79b2b9ae1`。本阶段为受控归因，不宣称提出新网络创新。

## 1. 旧结果复核

原始来源：Stage4.1四份指定报告、`results/tables/stage4p1/Full/curve_30.csv`与原checkpoint。新的实际重算见`results/tables/stage4p2/previous_full_weight_BN_factorial.csv`；全部921个ZARA2窗口、5833个目标窗口，完全复现原29/30轮端点。

|可训练权重epoch|BN运行统计epoch|验证NLL|K1均值ADE(m)|配对FDE(m)|
|---|---|---:|---:|---:|
|29|29|-0.716118|0.690975|1.317263|
|29|30|-0.804847|0.762845|1.515401|
|30|29|-0.665508|0.732897|1.368133|
|30|30|-0.789741|0.788508|1.540245|

原29→30轮NLL下降0.073623，均值ADE却上升0.097533m，FDE上升0.222983m。固定29轮权重，换BN统计使ADE上升0.071869m；固定29轮BN，换权重使ADE上升0.041922m。二者非线性耦合，不能把单独效应直接相加作为总效应。此为固定checkpoint推理交换，未独立重训，不证明训练因果来源的唯一性。

NLL衡量逐未来步真实**位移**在高斯密度中的似然；均值ADE衡量累积位移得到的位置误差。方差、相关系数、步间累积误差以及BN推理统计都可造成不同方向。NLL下降不蕴含均值ADE下降；上述交换提供BN参与退化的直接证据，但不证明NLL目标本身错误。

原Full使用BN和TXP节点3×3卷积；原NoSocial以9×1卷积并修改归一化。失败NoSocial在第5轮出现非有限评价，前4轮结果保留；NoSocial-ConsistentNorm取消训练/推理统计差异后完成30轮。其31个共享张量与原失败实验前4轮相同，不能把修复前失败解释为“社会信息必需”。原Full与修复NoSocial同时改变卷积和归一化，不能作为图单因素归因。

两者名义7563参数，未进入forward的`tpcnns.4.*`、`prelus.4.weight`共1309，实际进入计算图的分配参数6254。新模型每个1553参数，所有参数进入前向与梯度计算，G0/G1/G2初始权重完全相同。旧模型容量与新模型不同，不做跨家族等容量比较。

旧排列审计表明ST-GCN阶段误差约1e-7，第一层TXP节点卷积即出现明显差异；64窗口235目标的平均位置差0.266736m、最大1.917565m。普通节点索引卷积与BN跨节点统计均是额外社会路径，因此仅把旧图设为I不是严格无社会模型。

## 2. 数据与目标冻结

仅`A_offline_complete20`：完整8历史/12未来、0.4秒、米制、确定性主指标。上下文按完整20帧离线筛选，至少2人；此**存在性筛选依赖未来可见性**，不冒充history-only协议B。图的距离、权重和运动输入仅来自8帧历史，未来坐标仅进入监督与评价。ETH不读、不评价。

|留出录像|训练窗口/目标窗口|验证窗口/目标窗口|
|---|---|---|
|crowds_zara02|1488 / 23349|921 / 5833|
|biwi_hotel|1970 / 26994|301 / 1053|

每折完全排除验证录像；训练仅其他四录像的Stage3.5原训练时间区间，原验证和隔离带不混入。两个录像均已历史曝光，不能称独立确认。完整20帧支持跨训练/验证不重叠，因为录像完全隔离；同一验证录像内部的重叠窗口不独立。

manifest路径和SHA256、五个原始文件SHA256、窗口帧/行人ID见`configs/stage4p2/manifests/*.json`及`protocol.json`；处理缓存SHA256为`7cfe17f5b10a92a474a9f484bb061342fa6c91cff27a064f9a7e64d9dcf46b02`。不复制旧自定义项目数据/结果。

## 3. 选择与训练规则冻结

- 主指标：全验证目标窗口加权的K1高斯均值ADE；选择最低ADE，完全相同选更早epoch。FDE始终取同checkpoint。
- 辅助：验证NLL、K1首采样ADE/FDE、K20独立最小ADE与最小FDE（可来自不同候选）。每窗口CPU标准正态库由SHA256(`314159:window_id`)固定，与模型、batch和checkpoint无关；首样本取该库第一个。
- Stage4.1按NLL选择，Stage4.2按ADE选择；旧/新结果不属于相同选择协议。保留每轮所有指标，最小NLL轮仅描述。
- 固定seed17、宽16、SGD lr0.01、momentum=0、weight_decay=0、无scheduler、梯度裁剪1。mini-batch16窗口、每64窗口累积一次更新，余块按实际窗口数平均。损失保持官方PDF-clamped二维高斯NLL，每窗口平均后等窗口训练权重；验证NLL同时保存目标加权和窗口平均。
- 每折G0/G1独立从同初始权重训练，先5轮smoke；有限loss/指标/梯度、非零梯度、checkpoint逐位恢复通过后，恢复同一epoch5继续6–30轮。5轮属于30轮预算，无额外调参/重启；非有限值停止并记录，不搜索配置。
- 所有模型同逐人时间归一化；无running statistics，train/eval相同。无碰撞loss、额外注意力、高阶传播、GRU或新decoder变体。
- 预设值得验证条件：每录像G1 ADE降低至少`max(0.005m,1%×G0 ADE)`，且FDE增加不超过`max(0.01m,2%×G0 FDE)`。两录像均通过才建议下一阶段seed42/2026，本阶段不自动运行。
- G2独立重训需同时满足两录像上述收益及每折验证换边≥25%、改变非空图≥50%。否则仅结构与图统计，预测指标记未运行。当前ZARA2换边24.54%，已未达到门槛，不能看完预测误差后放宽。

## 4. 保护与运行

`configs/stage4p2/protected_artifacts.json`逐文件SHA256覆盖旧tracked源码/配置/结果/报告、历阶段登记checkpoint、官方资产、Stage4.1缓存和五个非ETH原始文件。旧Full、NoSocial失败前4轮与修正版均保留。

独立代码：`src/graph_attribution/`；入口`python scripts/stage4p2/train.py --fold <recording> --model G0 --phase smoke`，同模型随后`--phase finish`。使用既有公开复现独立环境`experiments/stage4/venv/bin/python`，新checkpoint只写`experiments/stage4p2/<fold>/<model>/epochXXX.pt`，排除Git。训练前结构和完整单元测试必须通过。

来源为本仓库冻结源码与真实实验产物；本阶段未增加外部论文性能主张。
