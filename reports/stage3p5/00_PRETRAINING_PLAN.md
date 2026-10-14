# Stage3.5训练前数据与预算冻结

基础46ecb91169eeadb5ba7ec28b7c648dd0ace837c6，新分支feat/stage3p5-benchmark-audit；606项Stage1–3源码/脚本/配置/报告/结果/权重保护，其中90个checkpoint。只新增Stage3.5代码和目录，不开发新网络，不使用旧项目模型或缓存。

协议A_complete8x12_history_context：目标历史8点/未来12点均完整，真实0.4秒间隔，米制K1 ADE/FDE；上下文仅凭完整8帧历史选择，不用未来存活选邻居。协议B_partial_mask_v1：原HTraj部分历史/部分未来mask不变。同一原manifest的五种录像隔离候选，新训练始终仅取原train区域的其它四个recording，原val不进入训练；held recording原train全部舍弃。验证仍是held recording原val区域，**不是整段录像全部窗口**。ETH测试不读取。

首轮固定students003为验证recording，因另一UNIV recording students001可用于训练，预先选择不根据指标。A训练2058有效窗口/19034目标，验证70窗口/706目标；B训练2365有效窗口/31458 ADE目标、23423 FDE目标，验证70窗口/1464 ADE、986 FDE目标。除去无合法标签的窗口，保留所有满足各协议定义的难例。完整五fold统计已保存split_counts.csv/split_group_counts.csv，当前只运行一个fold。

新训练数据与验证录像完全隔离，旧val没有偷偷混训；但这些录像曾被过去研究使用，70新val窗口与旧96开发有11个精确相同窗口，更多还有时间支持交叉，不能称独立确认集或全新数据。本次所有神经模型重新初始化，不复用旧权重于新fold。另可对旧B_legacy1024/96三seed做兼容冻结分析，必须单独命名，不能混入新fold指标。

预算先提交：以真实旧RTX3080时间PedBase107.43秒、Dynamic-RN252.50秒/1024窗口30epoch线性估计，A二模型约723秒训练，B约831秒，共约26分钟，另加读取/验证与群体大小影响；这是估算，不是已执行结果。全原训练pool在30/60/100epoch下的估计见pretraining_budget_estimate.csv。旧训练尾部小loss下降并不证明收敛，prior_convergence_audit.csv保留各run真实曲线诊断。

计划仅两协议×两个现有神经基线，各seed17、30epoch，CV解析评价；相同Adam0.001/batch8/clip1/d64/loss，own-protocol最小ADE选best并配对FDE，last30同时报告。公开参考Social-STGCNN只核查正式论文、固定代码与MIT许可/loader/输出/20样本评估，本阶段不运行其训练或发布公平性能排名。

分组仅依历史：密度平均连续帧入度0/(0,1]/(1,3]/>3；完整8/部分与长度细分；相向/同向/交叉按cutoff速度>0.2m/s与cos阈值±0.7，同一目标可属于多个运动类别；关系变化按相邻持续候选对的相对速度变化平均≥0.5m/s，稳定为低于阈值，缺少持续关系单列。这里的“稳定”是相对速度稳定，**不等于邻居集合或距离不变**。GT可用性只选择各协议评价目标，绝不参与上述分组。

每个模型保存同一target×window误差，进行配对差值/改善比例和固定不重叠窗口敏感性分析。单recording单seed不能回答普遍稳定改善，旧三seed开发只是历史假设证据。若效果不足如实记录，不据ETH调参、追加复杂结构或临时延长预算。
