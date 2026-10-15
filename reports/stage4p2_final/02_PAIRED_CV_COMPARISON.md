# 完全配对CV与G0/G1验收

直接读取Stage4.2冻结缓存的ZARA2 921窗口/5833目标窗口、HOTEL301/1053，全体6886目标窗口均保留。每一行共同window_id、agent_id、原始float32真实未来，另存逐目标future/history SHA256；同一12步ADE/FDE定义。没有使用Stage3.8任何旧CV指标。

CV公式：Last2速度=(X7−X6)/0.4；OLS4/OLS8对最近4/8帧真实0.4秒时间坐标求线性斜率，均以X7为起点预测X7+v*k*0.4。完整8帧任务没有缺失速度回退或未来补齐。解析模型0参数，无拟合超参数。

|录像|解析模型|ADE|FDE|纯解析计算秒|评价秒|
|---|---|---|---|---|---|
|crowds_zara02|CV-Last2|0.325740|0.728451|0.001512|0.001088|
|crowds_zara02|CV-OLS4|0.365459|0.780967|0.001040|0.001239|
|crowds_zara02|CV-OLS8|0.428842|0.873811|0.001023|0.001249|
|biwi_hotel|CV-Last2|0.322666|0.616897|0.000324|0.000317|
|biwi_hotel|CV-OLS4|0.249866|0.478500|0.000384|0.000234|
|biwi_hotel|CV-OLS8|0.241865|0.456838|0.000284|0.000172|


以上时间为CPU单线程向量化纯预测/距离评价实测，未含数据加载、写表，不能当端到端速度。神经预测位置保持原float32输出，与同一GT提升float64计算距离；聚合误差与冻结旧评价器核对差<=1e-6，不覆盖原指标。

## 全预算与末轮

|预算/选择|录像|模型|目标窗口|不同ID|ADE(m)|FDE(m)|
|---|---|---|---|---|---|---|
|budget30_best|crowds_zara02|G0|5833|187|0.661059|1.274254|
|budget30_best|crowds_zara02|G1|5833|187|0.713563|1.344462|
|budget30_best|crowds_zara02|CV-Last2|5833|187|0.325740|0.728451|
|budget30_best|crowds_zara02|CV-OLS4|5833|187|0.365459|0.780967|
|budget30_best|crowds_zara02|CV-OLS8|5833|187|0.428842|0.873811|
|budget30_best|biwi_hotel|G0|1053|112|1.227714|2.162568|
|budget30_best|biwi_hotel|G1|1053|112|1.180636|2.136477|
|budget30_best|biwi_hotel|CV-Last2|1053|112|0.322666|0.616897|
|budget30_best|biwi_hotel|CV-OLS4|1053|112|0.249866|0.478500|
|budget30_best|biwi_hotel|CV-OLS8|1053|112|0.241865|0.456838|
|budget100_best|crowds_zara02|G0|5833|187|0.514221|1.003209|
|budget100_best|crowds_zara02|G1|5833|187|0.605474|1.174621|
|budget100_best|crowds_zara02|CV-Last2|5833|187|0.325740|0.728451|
|budget100_best|crowds_zara02|CV-OLS4|5833|187|0.365459|0.780967|
|budget100_best|crowds_zara02|CV-OLS8|5833|187|0.428842|0.873811|
|budget100_best|biwi_hotel|G0|1053|112|0.980086|1.820474|
|budget100_best|biwi_hotel|G1|1053|112|0.933634|1.737428|
|budget100_best|biwi_hotel|CV-Last2|1053|112|0.322666|0.616897|
|budget100_best|biwi_hotel|CV-OLS4|1053|112|0.249866|0.478500|
|budget100_best|biwi_hotel|CV-OLS8|1053|112|0.241865|0.456838|
|budget100_last|crowds_zara02|G0|5833|187|0.560316|1.099074|
|budget100_last|crowds_zara02|G1|5833|187|0.622948|1.198346|
|budget100_last|crowds_zara02|CV-Last2|5833|187|0.325740|0.728451|
|budget100_last|crowds_zara02|CV-OLS4|5833|187|0.365459|0.780967|
|budget100_last|crowds_zara02|CV-OLS8|5833|187|0.428842|0.873811|
|budget100_last|biwi_hotel|G0|1053|112|0.985292|1.830687|
|budget100_last|biwi_hotel|G1|1053|112|0.933634|1.737428|
|budget100_last|biwi_hotel|CV-Last2|1053|112|0.322666|0.616897|
|budget100_last|biwi_hotel|CV-OLS4|1053|112|0.249866|0.478500|
|budget100_last|biwi_hotel|CV-OLS8|1053|112|0.241865|0.456838|


30与100分别用各自预算范围最低验证均值ADE、同一checkpoint FDE，完全同值更早轮；last100独立展示。100使用同开发集增加了选模次数，不是新的独立确认。CV在所有预算列完全相同，不能按录像事后混选CV作为一个新方法。

## 同目标配对差值

|录像|模型|对照|ΔADE|ΔFDE|逐目标ADE改善比例|
|---|---|---|---|---|---|
|crowds_zara02|G0|G0|0.000000|0.000000|0.000000|
|crowds_zara02|G0|CV-Last2|0.188482|0.274757|0.218241|
|crowds_zara02|G0|CV-OLS4|0.148762|0.222241|0.259215|
|crowds_zara02|G0|CV-OLS8|0.085379|0.129398|0.322647|
|crowds_zara02|G1|G0|0.091252|0.171412|0.453969|
|crowds_zara02|G1|CV-Last2|0.279734|0.446170|0.199211|
|crowds_zara02|G1|CV-OLS4|0.240014|0.393653|0.240528|
|crowds_zara02|G1|CV-OLS8|0.176631|0.300810|0.303446|
|biwi_hotel|G0|G0|0.000000|0.000000|0.000000|
|biwi_hotel|G0|CV-Last2|0.657421|1.203576|0.078822|
|biwi_hotel|G0|CV-OLS4|0.730220|1.341974|0.059829|
|biwi_hotel|G0|CV-OLS8|0.738222|1.363636|0.050332|
|biwi_hotel|G1|G0|-0.046452|-0.083046|0.603989|
|biwi_hotel|G1|CV-Last2|0.610968|1.120530|0.102564|
|biwi_hotel|G1|CV-OLS4|0.683768|1.258928|0.062678|
|biwi_hotel|G1|CV-OLS8|0.691769|1.280590|0.047483|


Δ为模型−对照，负值改善。完整目标配对明细`budget*_paired_targets.csv.gz`，不只保留成功目标。`all_group_results.csv`包含所有运动/人数/合法邻居组及空组；`per_person_errors.csv`按录像内行人ID聚合，`aggregate_comparison.csv`分开录像宏平均与目标窗口加权，重叠窗口不当作独立样本，不做iid p值。

完整20帧上下文人数包括目标且由离线完整筛选决定，不能混称历史可见邻居数；历史运动/密度分组沿用Stage4.2，并增加预定义静止细分。分组不读取未来误差；未来仅作为统一评价GT。ETH未评价。
