# Stage7.4训练可行性与validation诊断

最终决策：**NO-GO**。

- 当前磁盘不足以保留官方五折全部周期checkpoint及128MiB余量，正式训练启动NO-GO；不得删除历史成果解决。

完整epoch及恢复工程检查：True。官方模型、loss、Adam和调度器保留原样；只测HOTEL两阶段各1轮与各1次独立影子恢复更新。无五场景test重复评价，无新模型。

|validation fold|窗口|目标窗口|minADE20|minFDE20|
|---|---:|---:|---:|---:|
|hotel|688|5203|0.190930|0.329762|
|zara2|590|4262|0.203742|0.346566|

K1预算单独报告：

|validation fold|确定性模型|ADE|FDE|
|---|---|---:|---:|
|hotel|CV-Last2|0.464598|1.033970|
|hotel|CV-OLS4|0.514762|1.097719|
|hotel|CV-OLS8|0.595732|1.211699|
|hotel|DLow第一头（非数学期望）|1.179269|2.681585|
|zara2|CV-Last2|0.502014|1.106373|
|zara2|CV-OLS4|0.545533|1.157252|
|zara2|CV-OLS8|0.627149|1.272799|
|zara2|DLow第一头（非数学期望）|0.701207|1.541769|

完整测试：{'tests': 426, 'failures': 0, 'errors': 0, 'skipped': 0}。保护历史文件11459份与Stage7.3预测/GT/recon文件73656份，SHA变化0。

预算与假设见05报告，完整阶段日志、逐步耗时、loss、checkpoint SHA、状态恢复和分组结果见results/tables/stage7p4/execution_v2。首次validation因进度日志缺少sys导入在100窗口后停止，零训练更新；原现场和首次冻结提交保留，补导入并在独立execution_v2目录重新冻结后执行。首次与本次已生成重叠窗口需核查预测不变。真实失败见根目录execution_failure.json或本次对应_epoch.json，不存在的运行不填指标。

只本地提交，不自动push：`git push -u origin feat/stage7p4-agentformer-training-feasibility`。完成后停止。


## 完整epoch实测摘要

|阶段|有效窗口/更新|epoch秒|P50/P90/P99 step秒|稳态均值秒|整卡采样峰值MiB|
|---|---:|---:|---:|---:|---:|
|CVAE|3118|509.883|0.1536/0.2102/0.2993|0.1622|9677|
|DLow|3118|307.355|0.0844/0.1510/0.2371|0.0971|8982|

每阶段遍历4349个官方候选frame slots，其中3118个有效窗口、29676个目标窗口、最大上下文57人。共6236次完整epoch更新，另有2次独立影子恢复探针更新；probe不计入epoch耗时，主checkpoint没有被probe覆盖。
PyTorch allocated/reserved是精确峰值；整卡显存为名义0.1秒轮询加命令耗时的采样峰值，可能漏掉短暂峰值，不能与前两者当作同一时刻的读数。
成本仪器说明：原train函数体执行，TensorBoard add_scalar调用由计数receipt接收，完整loss另存逐步CSV/JSONL；未测原SummaryWriter的全部磁盘开销。epoch计时包含同步与有限性检查，不含loader初始化、checkpoint保存及恢复探针。
