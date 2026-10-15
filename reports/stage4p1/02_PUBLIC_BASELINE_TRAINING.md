# Social-STGCNN从头训练：真实执行与数值边界

Full保持官方维护版结构和原PDF-clamped NLL，完成seed17、30epoch；未用预训练权重。首5轮在提交04254e4之后运行，检查点预测逐元素恢复一致，随后同一run继续6–30轮，不重复训练前5轮。验证NLL选best，所有epoch完整保存，不按ADE改选。

## 真实主结果

所有验证项使用同一921窗口/5833目标窗口，米制。NoSocial指单次修订后从头训练的NoSocial-ConsistentNorm；原失败版本独立保存，不混入结果。

| 模型 | checkpoint/epoch | NLL | K1均值 ADE/FDE | K1首样本 ADE/FDE | K20独立最优 ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| Full | best/30 | -0.789741 | 0.788508 / 1.540245 | 1.229007 / 1.984716 | 0.525156 / 0.680384 |
| Full | last/30 | -0.789741 | 0.788508 / 1.540245 | 1.229007 / 1.984716 | 0.525156 / 0.680384 |
| NoSocial-ConsistentNorm | best/30 | -1.332365 | 0.759140 / 1.430127 | 0.910583 / 1.595987 | 0.467411 / 0.749415 |
| NoSocial-ConsistentNorm | last/30 | -1.332365 | 0.759140 / 1.430127 | 0.910583 / 1.595987 | 0.467411 / 0.749415 |

K20对每个目标分别取minADE、minFDE，可能来自不同候选。固定采样seed314159在两模型使用相同标准正态银行；不是三训练种子。NLL连续概率密度可大于1，因此负NLL不是负距离或程序错误。

## 训练/验证与耗时

| 模型 | best train NLL（目标加权） | best train K1均值 ADE/FDE | best validation K1均值 ADE/FDE |
| --- | --- | --- | --- |
| Full | -0.839265 | 0.835855 / 1.599930 | 0.788508 / 1.540245 |
| NoSocial-ConsistentNorm | -1.093723 | 0.887970 / 1.680544 | 0.759140 / 1.430127 |

| 模型 | 训练秒 | 30轮验证合计秒 | 窗口/秒 | 目标窗口/秒 | 峰值allocated MiB | 最小均值ADE epoch（仅描述） |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 569.416 | 78.494 | 78.396 | 1230.156 | 16.707 | 29 |
| NoSocial-ConsistentNorm | 709.685 | 94.998 | 62.901 | 987.015 | 16.907 | 30 |

吞吐为CUDA同步后的训练循环实测，包括搬运和反传；两模型串行运行，没有彼此争用GPU。train耗时合计不含每轮验证、checkpoint序列化、准备和最终best/last重评价。显存为PyTorch allocated峰值，不代表整个CUDA进程显存。环境Python/torch/CUDA/GPU及数据/代码commit见environment.json和各模型execution.json。

每模型30×24=720次SGD更新；每轮1488窗口/23349目标窗口，最后16窗口累积按实际数量平均。原官方train入口未执行，避免其累积边界漏窗口；优化器仍SGD/lr0.01，不做scheduler或多配置搜索。训练梯度均有限且非零，梯度均值/P95/max在每轮曲线。两者均7563可训练参数，其中原forward永不调用的1309参数保留，有梯度路径的分配参数计6254；不隐藏这部分容量。

训练NLL初末值：

| 模型 | epoch1 train-mode窗口均值NLL | epoch30同口径NLL |
| --- | --- | --- |
| Full | 1.566442 | -0.519615 |
| NoSocial-ConsistentNorm | 1.673893 | -1.172101 |

这证明有限预算下可以数值稳定学习，不证明已经充分收敛或达到论文水平。30轮内验证误差并非单调，NLL和均值ADE最优轮次也可能不同；描述性最低ADE轮次不用于改变预设选择规则。训练态窗口等权NLL不能与评价态目标加权NLL直接解释为泛化gap，上表提供同为eval/目标加权的train/validation结果。

## 原NoSocial失败与一次修正

原NoSocial完整记录第1–4轮，执行第5轮训练后评价出现非有限NLL，未保存第5轮完整结果或checkpoint；停止而非填0。失败日志、4个权重及nosocial_failed_pilot.json保留。第4轮固定权重诊断显示逐人训练统计与全局running评价统计不一致，能明显放大Gaussian输出。修订前冻结nosocial_amendment.json，只改为训练/评价均逐人时间归一化，数据、lr、损失、预算、Full实现均不变。修订从头训练，5轮恢复门槛通过后继续到30，没有第三次尝试。

此修订不是最初就完整预注册，须连同失败过程报告。另核查两run前4轮checkpoint，每轮31个共有状态张量逐元素一致，支持修订未改变训练态变换/更新，确实只修正评价模式而非寻找更有利参数。新run仍从头训练，未复用失败权重。初始diagnosis里的零参数偏移字段源于诊断累积变量错误，不能使用；nosocial_failure_diagnosis_v2.json和保存的诊断脚本给出正确最大参数偏移17.1893，策略的log-sigma/correlation最大值结论未改变。

## checkpoint与来源

| 模型 | 选择 | epoch | 本地checkpoint | SHA256 |
| --- | --- | --- | --- | --- |
| Full | best | 30 | experiments/stage4p1/Full/epoch030.pt | 54c21a3af3333094ef36201d6bc53c05fc73a74179b5bc1d099a37967c509c74 |
| Full | last | 30 | experiments/stage4p1/Full/epoch030.pt | 54c21a3af3333094ef36201d6bc53c05fc73a74179b5bc1d099a37967c509c74 |
| NoSocial-ConsistentNorm | best | 30 | experiments/stage4p1/NoSocial-ConsistentNorm/epoch030.pt | 58e8469b5b68f72bb6153176bf75dcf5b1408d9ef3a06160c9def5bfdb8ac498 |
| NoSocial-ConsistentNorm | last | 30 | experiments/stage4p1/NoSocial-ConsistentNorm/epoch030.pt | 58e8469b5b68f72bb6153176bf75dcf5b1408d9ef3a06160c9def5bfdb8ac498 |

| 协议 | manifest SHA256 |
| --- | --- |
| A_offline_complete20 | 191cab0e257a56af4238557da95bd3d5a2bd9d593346c29f7cbe2708399abfaa |
| B_history_context_complete_targets | 711295f121aad002dc675a24ec61db6c0f310d5ef750049c1cd197db1501e390 |

data SHA逐录像存于preparation.json；原始数据、29MB缓存和全部checkpoint均被Git忽略。curve_30.csv保留每轮所有三种ADE/FDE、NLL、梯度、显存、训练/评价秒和吞吐。逐目标配对数据仅保存误差，没有上传原始轨迹。

与Stage4官方checkpoint不能直接判优：本轮排除ETH/zara03/uni_examples、保留其他录像旧val不用，只有1488窗口/30epoch；官方权重的训练预算、训练池、上游选择历史不同且可能使用ETH训练材料。本轮ZARA2整录像评价任务与官方离线筛选相近，但存在细微浮点处理顺序差异，不能称完全训练复现。没有运行官方250轮训练或任何ETH评价。

完整单元测试219通过，耗时9.425秒；test XML、真实epoch日志和数据hash均已保存。


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
