# 原样完整训练预算

|情景|未来完整运行秒|未来小时|与当前剩余额差距秒|假设|
|---|---:|---:|---:|---|
|A_HISTORICAL|16352.19|4.542|0.00|Semantic timings only; no guarantee for new NoSemantic distributions|
|B_POST_EPOCH1|17613.24|4.893|369.86|All7later validations use postdiagnosticEpoch1 mean; later checkpoint behavior unmeasured|
|C_SLOW_STRESS|22352.89|6.209|5109.51|All7validations retain slower initial/epoch1 per-sceneP90; conditional stress, not empirical runtime bound|

情景A来自旧语义完整30轮和7次全量验证的真实记录，迁移到无语义模型仍是假设。
情景B以本轮完整epoch耗时外推30轮，用epoch1后每录像3个固定目标的普通完整预测耗时均值，
按真实录像目标数投影998个目标和7次验证，另计重复检查、旧评分/保存开销、31个checkpoint、启动、异常和退出预留，增加10%时间裕度。
情景C对每录像取NoSemantic初始与epoch1的较慢P90，假设7次验证均保持该慢速，增加20%裕度。
这是条件性压力情景，不是总体统计上界，也不能假定实际epoch30仍为初始速度。
没有对未测epoch做插值；所有30轮和998目标的新增成本均是外推，未执行。
当前剩余额已扣除Stage11C-1和本轮全部GPU相关进程wall；诊断checkpoint未获准复用，因此成本从头计算。


建议提交用户审查的累计预算为23400秒（6.50小时），比原18000秒增加5400秒；它覆盖本轮条件压力情景及已用预算，不是实际耗时保证，也不构成正式训练授权。

已用原C1=66.674948s，本轮=689.949887s，合计=756.624836s，剩余=17243.375164s。
