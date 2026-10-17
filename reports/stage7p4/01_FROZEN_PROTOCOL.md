# Stage7.4执行前冻结协议

基础提交`d87551873338183bbeb178bec3e321f54c732256`，新分支`feat/stage7p4-agentformer-training-feasibility`。官方维护版固定为`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；保留Stage1—7.3所有成果，包括Stage7.3全部预测文件与测试指标。阶段代码、配置及停止规则真实提交成功，且工作区干净后才运行。

仅使用官方8/12、0.4秒、米制、完整20帧离线上下文协议。未来存在性可用于官方上下文资格，但本阶段历史分组只读取8帧历史。不得重新计算任何五折test误差，不将已观察test表现用于后续选择。

validation诊断固定HOTEL与ZARA2 fold的官方`val`，不是这两条held-out测试录像。两折validation各包含其余录像的官方`*_val`部分，可能彼此重叠，不称为两个独立验证集。按原作者CVAE+DLow严格权重加载、DLow mean=True、20个learned latent heads，只调用infer，不使用recon。指标在原float32输出上进行float64计算，不作Stage7.3文本三位小数序列化；不混排test结果。

CV-Last2/OLS4/OLS8均为一条确定性轨迹，形式为X_last+v_est×k×0.4；OLS取相应历史点的含截距最小二乘斜率，预测仍锚定X_last。分别展示CV的K1与DLow的K20；不能声称同候选预算优势。DLow第一头K1为辅助，不是数学期望。

历史平均速度为七段相邻位移模长的平均值/0.4；固定stationary<0.05、low<0.4、normal<1.8、fast≥1.8m/s；同时保留静止/运动两组。历史邻居组按最后观测位置2米内、完整历史8帧可见的其他行人数划分0、1—3、4—9、10+，不使用未来存在性；这仅是诊断，不改变模型图。完整20帧上下文人数另存描述字段，不混为历史密度。每个未来步分别保存首头、整轨迹ADE最优头、FDE最优头的误差；后两者为事后oracle诊断，不是预测时选择器。终点离散度为20头终点RMS半径及190对平均距离，不使用GT构造输入。

训练只测HOTEL：原CVAE配置字节不变，随机初始化，官方seed=1、Adam lr=1e-4、原三项loss、K20 sample loss、StepLR(10,0.5)，原num_epochs=100不改，实际调用原train(0)一次后停止。原shuffle遍历所有候选frame slots，None无效窗口不优化，3118个有效窗口各更新一次，不删任何高人数样本。最大上下文57，低于原HOTEL max_train_agent=100，不会触发原源码的截取规则。

DLow随机初始化，只引用本轮CVAE的1-epoch checkpoint。配置唯一覆盖为pred_epoch 50→1，pred_cfg名称不变，在新隔离工作区解析；其余原K20、KL、diversity、reconstruction、Adam lr=1e-4与StepLR(5,0.5)不改，原num_epochs=50不改，执行一次完整epoch。CVAE保留官方requires_grad/eval行为，可有梯度但不在DLow优化器中；检查优化器参数ID不相交并逐参数状态hash确认其未更新。训练期间torch.load仅允许本轮工作区checkpoint，禁止作者或旧smoke权重初始化。

保持原train.py函数体（AST提取），只增加不修改梯度的loss/gradient/parameter有限性检查、同步计时和日志。同步与诊断会产生测量开销，单独记录，不声称测得完全无仪器开销的原CLI吞吐。稳态时间只排除最初10次更新用于统计，真实epoch仍覆盖所有窗口。loader初始化时间、所有candidate读取时间与epoch计算时间分开保存。

每阶段保存模型、Adam、scheduler、Python/NumPy/torch CPU/CUDA RNG与generator顺序/index。严格恢复后，在影子副本按恢复RNG执行下一epoch的首次有效窗口一次更新，验证优化器状态未丢失、参数可继续变化；每阶段最多1次工程恢复更新，共最多2次，不计入完整epoch成本，不覆盖原1-epoch checkpoint，不保存为正式结果，不与作者性能比较。

OOM、NaN/Inf、目标/数据错配或恢复失败即停止后续训练，保存现场，不重试、不缩K、不删行人、不改学习率/损失/网络、不延长。CVAE失败则DLow未执行；未运行项必须明确标记。

完整预算仅估算100 CVAE+50 DLow：使用HOTEL实测step与上下文人数拟合非负[1,N,N²]时间项，按各折冻结人数分布估算，并预留0.7—1.5倍敏感区间。ETH原CVAE cap32、额外结构、未测fold资源等另列风险。规划额外validation每10个CVAE/每5个DLow epoch一次，按实测DLow K20 val耗时0.5—2倍估算，不冒充原train.py内置验证。原checkpoint频率CVAE每10轮、DLow每5轮，全部保留。

预设工程时间门槛：HOTEL上界72小时，五折上界336小时；这是本阶段规划筛选门槛，不是用户已批准正式训练。若完整预算超门槛、当前存储不足保留官方五折checkpoint和128MiB余量，或核心执行检查失败，总判定NO-GO，明确阻碍原因；否则GO，仅允许下一阶段制定正式计划。当前磁盘条件可能导致存储NO-GO，不能删除历史权重来强行通过。本轮执行最低空闲400MiB，测量预算约280MiB，低于保护线则不启动。

所有数值均来自实际运行，失败不补零。原始数据/模型/预测只在ignored cache中保存；轻量配置、日志、逐目标与逐步统计进Git。本阶段不自动push。

来源：[官方固定版本](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)的train.py、model/agentformer.py、model/dlow.py、cfg及data入口；本地完整SHA清单见configs/stage7p4/protocol.json。
