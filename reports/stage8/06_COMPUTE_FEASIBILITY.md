# 计算、资源与论文实验成本

RTX3080，nvidia-smi总显存10240MiB，PyTorch当前可见10366025728字节。新环境目录与运行工作区独立；公共依赖只读复用，不修改Stage4环境。执行版本与作者pinned版本不同，完整环境分别记录。

原Re随机初始化4个真实validation batch与一次作者权重forward的记录wall总计4.851s（含CPU扫描/设置，不含进程导入；不是纯GPU kernel计时），远低于7200秒预算。MoFlow GPU使用0秒，只做CPU组件检查。不是GPU还有时间就继续训练。

Re最重case为42上下文/42目标，peak allocated约89.39MiB；reserved及整卡快照详见JSON。整卡读数是时点快照，不是高频峰值监测。单人/多人无OOM、无NaN/Inf、原损失反向有限，checkpoint严格恢复。没有优化器更新、没有完整epoch、没有测作者batch1000，因此不能宣称200epochs完整预算已验证。

## 训练成本的可报告与不可报告部分

Re原论文为一阶段200epochs、batch1000、RTX3090；发布ZARA2 args实际lr1e-4，论文lr3e-4，HOTEL发布的是300epoch消融。未来如重训需要先冻结一套可核验配置，不能根据val成绩在几套作者配置之间切换。

以已有官方HOTEL训练目标29676为**适配后规划参考**，batch128、200epochs约46400次更新。若未来实测每step为0.1/0.5/1s，对应1.29/6.44/12.89h每run；这些step时间是假设而非本轮测量，实际还需validation、IO和checkpoint开销。原batch1000为约6000次更新，但其显存和step时间均未测，不能用42人的小batch直接线性保证。原Luna训练目标数不同，不替换为这一参考数量冒充原论文预算。

一个最低可解释算法比较至少3组×2来源/fold，再做容量/机制消融与多seed，成本成倍放大。单个小batch便宜，不代表一篇论文全部实验成本已经得到准确估计。当前不建议在新颖性尚不明确时支付这笔训练预算。

MoFlow默认teacher150epochs+student150epochs，另需来源训练池teacher离线样本（README100步示例）、存储、teacher权重及EMA。其teacher4.70M/student4.09M的CPU参数量不是训练显存；从未运行其完整loss/backward、采样/IMLE训练，所以不能借用AgentFormer Stage7.4完整epoch时间来估算MoFlow。

结论：Re有限工程运行在本机可行；两者完整论文训练预算未实测。MoFlow完整依赖和原协议匹配尚不足。没有删高密度窗口、缩K20、改网络来制造资源可行性。研究GO未成立，后续大batch或完整epoch工程预检也不自动启动。
