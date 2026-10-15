# Stage4.1训练前冻结协议

本阶段使用官方维护版 `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`，不是精确CVPR2020架构。Full直接实例化Stage4冻结 `model.py`，不改源文件、ST-GCN、TXP-CNN、高斯输出或原clamped NLL。新代码位于 `src/public_training/`、`scripts/stage4p1/`；复用Stage4独立Python环境，不用旧项目模型或缓存。原始来源与版本证据见 `reports/stage4/01_SOCIAL_STGCNN_AUDIT.md`。

## 两套数据协议

**A_offline_complete20**：8帧完整历史、12帧完整未来，所有参与行人均覆盖20步，每窗口至少2人，ID排序、四位小数位置。上下文和目标由未来存在性共同筛选，与官方维护版离线筛选规则一致；不是在线历史上下文协议。建图函数只有历史参数，没有未来坐标参数。

**B_history_context_complete_targets**：上下文只要求完整8帧历史，未来存在性仅标记完整12步的监督目标；至少2个历史上下文且至少1个目标。未监督的邻居未来用占位张量加mask，不进入图或输入。本阶段只生成其独立manifest和计数，没有B训练/评价，不将A/B混合对比。

两者固定0.4秒、米制坐标、frame-ID间隔10、完整20帧支持；拒绝跨缺帧区间和跨录像滑窗。与官方数据相比，浮点位移/图的计算顺序可能有微小舍入差异，人工图与官方实现已作2e-6容差比较，不声称字节级loader复刻。

| A划分 | 录像 | 窗口 | 目标窗口 |
| --- | --- | ---: | ---: |
| train | biwi_hotel | 231 | 758 |
| train | students001 | 336 | 11691 |
| train | students003 | 413 | 8988 |
| train | crowds_zara01 | 508 | 1912 |
| train合计 | 四个非ETH录像 | 1488 | 23349 |
| validation | crowds_zara02整录像 | 921 | 5833 |

训练仅使用此前Stage3.5标记的原train时段，继续保留其他录像的旧val和隔离带不用；没有把原验证窗口悄悄加入训练，也没有只取1024子集。验证使用留出ZARA2整个录像，其原train/val时段均不进入本轮训练。仅录像隔离，不等于新独立确认：ZARA2早已曝光。ETH未加载、未评价；官方额外zara03/uni_examples也不在本轮训练池。

ZARA2在官方benchmark中称test，本次按用户指定重新定义为**开发验证录像**；它不再具有官方未调参测试的含义。本轮封存测试是ETH，没有额外读取测试指标选择epoch，也不把ZARA2开发成绩冒充论文官方测试复现。

manifest存于 `configs/stage4p1/{A_offline_complete20,B_history_context_complete_targets}_manifest.json`，数据文件及manifest SHA256固定于 `protocol.json`、`preparation.json`。完整cache仅在忽略的 `data/processed/stage4p1/A_windows.pt`，不可上传。

## 网络、图和张量

实际Full/NoSocial均7563参数。Full输入 `[1,2,8,N]` 与 `[8,N,N]`；ST-GCN产生 `[1,5,8,N]`，沿原 `.view` 变为 `[1,8,5,N]`，TXP-CNN时间通道8→12、保持feature轴5，最终原 `.view` 恢复 `[1,5,12,N]`；包装输出 `[12,N,5]`。原网络n_txpcnn=5的第4号中间卷积/对应PReLU未被其forward循环调用，保留原行为并记录未用参数，未自行修复。真实逐模块形状在各模型 `smoke.json`。

历史位移 $d_{i,t}=x_{i,t}-x_{i,t-1}$，首步0。官方图为 $W_{ii}=1$，非对角 $W_{ij}=1/\|d_i-d_j\|$（差为0则边权0），$L=I-D^{-1/2}WD^{-1/2}$。这是位移差逆距离，不是物理位置距离图；没有用未来坐标或未来图作为模型输入。

输出 $(\mu_x,\mu_y,\log\sigma_x,\log\sigma_y,\rho_{raw})$，$\sigma=\exp(\log\sigma)$、$\rho=\tanh(\rho_{raw})$。直接调用冻结 `metrics.bivariate_loss`，包括PDF最小1e-20的原始clamp；没有改loss或添加碰撞项。采样位移累加到最后历史位置。

## 预设训练与终点

- 模型列表仅Full、NoSocial；均seed17，从随机初始化训练，禁止用官方权重初始化。Full与NoSocial共享相同原始参数抽样，NoSocial节点卷积权重只重排形状。
- SGD，lr0.01、无momentum/weight decay/scheduler；每64窗口累积一次更新，梯度范数裁剪1；每窗口NLL等权，最后不足64窗口按真实数量平均。与原官方train入口相比，显式修复其累积边界漏样本/尾批处理，未照搬异常计数；两模型相同。
- 顺序由固定epoch seed确定，两模型相同；每轮仅train更新。validation用eval/no_grad，既不反传也不更新BN统计。
- 先5epoch；若loss/三种指标/梯度有限、梯度非零、epoch5 checkpoint含optimizer/RNG可恢复且预测逐元素相同，则恢复同一run继续6–30epoch。5轮包含在30轮总预算内，不额外重训。没有验证提升门槛、不按结果重新挑配置。
- best按**验证目标加权NLL最小**，精确平局选最早epoch；只在验证集选。K20与ETH均不参与选择。描述性最低ADE epoch另列但不用于更换交付模型。
- 每epoch保存不可覆盖的 `experiments/stage4p1/{Full,NoSocial}/epochXXX.pt`、训练NLL/梯度/吞吐/显存、验证所有指标。best与last记录实际路径和SHA；不覆盖旧checkpoint。
- 若NaN/Inf、非正定高斯或恢复失败，停止对应run并报告，不临时调lr或补造指标；最多30epoch，无多seed、无换录像、无新网络设计。

## 完整NoSocial定义及解释边界

NoSocial将图设为I；所有TXP-CNN节点维3×3卷积改为9×1，原九个核系数重排到单人feature轴，保持相同参数数量；ST-GCN本身的3×1时间卷积和其他原结构保留。训练态BN改为逐(B,节点)的时间维均值/方差，共享affine，running统计只从训练数据更新。评价时使用训练running统计，不读取当前邻居。由此不存在图、卷积或当前BN跨行人路径。

代价是卷积核几何和规范化一起改变，函数空间/优化行为不完全相同；相同参数量不等于同一有效容量。该对照回答“整套跨行人计算是否胜过明确隔离的近容量个体结构”，不能独立证明图边有贡献。单位图不足以移除TXP-CNN消息，已有输入Jacobian测试；NoSocial在train/eval下其他行人的输入Jacobian均为0，且保持排列等变。它是基础归因控制，不是新论文方法。

## 评价和诊断预注册

同一目标集合同时报告K1高斯均值、K1首采样、K20独立minADE/minFDE及NLL。固定采样seed314159，使用跨模型/epoch一致的标准正态银行；ADE和FDE分别取K20最优，不能认为对应同一轨迹。逐目标best/last配对误差完整保存，重复窗口不作为独立统计样本。

排列诊断只用Stage4冻结的官方ZARA2权重、验证集时间顺序前64窗口、排列seed271828、K1均值，不采样。历史/图同时置换、预测恢复原ID，分模块定位最先出现差异处；不修改官方网络掩盖问题。它是固定权重推理诊断，不是独立训练性能证据。

以上配置和代码先本地提交再训练；后续真实执行commit与SHA记录于结果，所有历史资产由本阶段SHA清单保护。只本地提交、不自动push，阶段完成即停止。

## 首次烟雾失败后的单次明确修订

原NoSocial在第5轮评价遇到非有限NLL，按门槛停止，未继续到30轮。第1–4轮真实日志和checkpoint保留，未用于最终公平对比。第4轮固定权重、同一批历史的诊断中，全局running统计使最大相关logit达到8.399、绝对log-sigma达到19.260；改回训练时的逐人统计分别为1.441/2.397。它支持归一化模式不一致造成输出尺度放大的解释；由于第5轮失败时未保存参数，不能精确追溯该轮每个NaN运算。

因此只允许一次修订 `NoSocial-ConsistentNorm`：训练和评价均使用同一逐人时间统计、共享affine，不保存running统计。其训练态变换与原NoSocial逐元素一致，NoSocial参数仍7563，邻居Jacobian仍0。数据、种子、优化器、lr、NLL、Full网络和最多30轮预算全部不变。新定义在修订模型训练之前提交，修订模型从头训练；旧失败run独立保存，不将其checkpoint当作新模型初始化。此为明确记录的流程修订，不冒充最初就完整预注册。

若修订后5轮门槛仍失败，停止，不继续改结构或搜索配置。NoSocial归一化/卷积与Full仍存在混杂，不能据此宣称纯图独立收益。修订清单 `configs/stage4p1/nosocial_amendment.json`，独立实现 `model_consistent.py` 与 `run_consistent.py`；原模型和原训练脚本保留。
