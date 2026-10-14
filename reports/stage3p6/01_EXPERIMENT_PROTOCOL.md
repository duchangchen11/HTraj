# Stage3.6预注册：强个体运动与冻结社会残差

基础2abca51dd42489a06668e6430531b984dff14641；分支feat/stage3p6-frozen-ego。configs/stage3p6/protected_artifacts.json保护全部642个既有tracked文件及98个既有checkpoint，共740项。旧源码/模型/配置/报告/结果不修改；新增输出独立目录、拒绝覆盖已有run。

本报告与protocol.json、实现及165项训练前通过测试一起提交后才启动训练。所有阈值/终点在观察本阶段验证结果之前固定；不自动push。

## 数据与任务

仅A_complete8x12_history_context：8完整历史、12完整未来、0.4s、米制ADE/FDE、确定性K1。历史上下文只按完整8点选择，不以未来存活选邻居；GT只定义完整监督目标与计算误差。保持Stage3.5原始数据、采样、mask及manifest，不宣称Social-STGCNN官方loader复现；不运行B，不读取ETH。

使用Stage3.5 LORO的原train其它四录像、held录像原val，held录像原train舍弃，原val绝不混train。完整20点支持190frame-ID ticks/7.6s，原同录像时间切割200ticks隔离仍保留，新的train/val录像无交集。ZARA2/HOTEL选择来自旧开发收益/退化，且旧研究使用过这些记录，所以不是独立确认数据，也不是完整录像未用标签测试。

| 留出录像 | manifest SHA256 | 有效训练窗口/目标 | 有效验证窗口/目标 |
|---|---|---|---|
| crowds_zara02 | 6c63448ecd2e921a8f99c1af4d7cbc2f109e9a6d40dc92e8db74f52525e7f6fa | 1684/23545 | 172/1092 |
| biwi_hotel | 3f8e2cf1e6462656fc0a15f0483ebe3ea63db948b946d8cce70ebd9331e7a7ea | 2121/27145 | 94/318 |

HOTEL原149个验证窗口中55个无完整8/12监督，按预先定义的协议无合法目标；部分观察任务仍完整保留在旧B结果，不以表现删难例。每目标ADE=12步平均距离，FDE=第12点距离；同target tuple录像/窗口/行人配对，重叠窗口不是独立样本。全部源文件SHA保留protocol.json及各run配置。

## 固定训练列表与预算

第一轮每录像seed17：CV解析评价；PedBase与CV-Residual分别constant和step10_20各从头30epoch，共8个神经run。Adam0.001/batch8/d64/原masked逐人L2/clip1；step在10与20轮结束乘0.3，即11–20轮0.0003、21–30轮0.00009。两模型编码器与decoder相同、初始参数逐字节相同、总参数均35352；CV-Residual只把输出主路径由当前位置改为历史CV。CV不足连续两点时明确stationary fallback，encoder保留速度缺失flag，不伪造真实速度。没有新loss/碰撞约束。

所有run固定30轮，无延长训练、额外LR搜索或反复切换录像。模型选择均本录像验证最小ADE，精确平局选最早轮，配对FDE与last30另报；不把FDE另选最小。

anchor按每录像四个神经候选最低best ADE选择，平局按列表顺序。若解析CV不差于最佳神经模型，输出固定CV，但保留该最佳神经checkpoint的encoder/head作为冻结观察编码器；否则输出该神经预测。输出方式在所有社会变体中完全相同，不为了社会收益使用明显更弱的ego。需四个神经run均30轮finite loss/grad/metrics且结构测试通过；非有限失败即停止社会训练。

第二轮同anchor：Ego-Only解析；Ego-Residual、Social-Previous、Social-Current、Social-Random各从头训练新增分支30轮。constant Adam预算同上；原ego所有参数requires_grad=False、eval，优化器只含14425个新增参数，每轮参数SHA验证、梯度为空，最终checkpoint前后验证。新缓存仅本阶段从历史生成的冻结编码/预测，在内存中，不用旧缓存。

第三轮仅在某个社会变体**同时在两录像**优于Ego-Only和Ego-Residual的ADE至少max(0.005m,参考ADE×0.5%)，且两项参照的配对FDE均不退化时执行42/2026。这是探索筛选，不是显著性检验。候选优先Current/Previous/Random；如果通过，固定17选择的每录像ego家族/schedule/CV输出方式，新seed分别重训anchor，四种分支均用对应同一anchor/预算；不替换seed。若未通过停止，不盲目扩展模型。判定文件记录实际已运行与未运行项，不填假结果。

## 信息来源与融合

统一4m/k8/CPA4.8秒、双端连续两点资格。固定均匀邻居平均和有消息历史帧平均，不引入时间注意力、高阶状态、关系GRU或新decoder。Previous为h_j,t−1，Current为严格前缀编码h_j,t；几何均来自t，两者只改变payload年龄。随机图在同batch窗口、同历史帧、两端运动已知、无自环、4m内合法集合中按坐标/时间hash选择，与真实入度预算一致（整个hash平局弃去并报告差值）。原集合≤k时无法换边；记录真实换边率/距离分布，不能因弱负对照无变化声称无关系价值。

所有新增分支相同71→64 payload MLP、128→64→24残差头及一个scalar gate，共14425参数；Ego-Residual输入自身上一帧h和自身7维运动特征，不读邻居，作为同容量对照。三社会分支用邻居h及7维当前关系几何。残差最后层统一零初始化，初始预测精确anchor；gate logit统一−2。来源不同，模块/初始化/优化预算相同。Ego-Residual使用全部连续自身帧，社会仅有合法消息帧，这是信息支持差异，应明确报告。

## 分组、终点与解释边界

沿用Stage3.5全部17组，并单列no_legal_social_messages。历史匀速：有效相邻速度变化均值<0.2m/s；显著变化≥0.5，中间与不足观测另列；相向/同向cos±0.7及双方速度>0.2，交叉类别可重叠；邻居相对运动变化阈值0.5m/s。不根据空组调整阈值。CV大误差组只在训练结束后从保存误差取CV ADE≥1m，不进入输入/gate/选择。

保存每epoch实际LR/loss/ADE/FDE/梯度/状态/门控/修正及消息数、best/last权重SHA、同target逐人误差、全部分组/人数，低人数与空组如实显示。两个录像单seed结果不能证明跨seed稳定，旧recording曝光与多候选anchor选择亦限制泛化。若社会不超过anchor或同容量自修正，不支持独立社会收益；即使冻结成功，也没有同结构联合训练反事实，**不能单独证明过去GNN失败必然由联合训练破坏主路径导致**。
