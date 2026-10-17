# Stage7.3执行入口

所有操作只读历史成果，只在本阶段目录创建新文件。既有结果使用exclusive create保护，原目录不能直接重复执行；重复实验需要独立的干净运行副本与相同冻结输入。

主线执行顺序：prepare.py → preflight.py → 成功的独立Git冻结提交和clean gate → run_evaluation.py infer → audit_outputs_before_metrics.py → run_evaluation.py score → 报告归档 → verify_results_batched.py → 本地交付commit。完整pytest在推理结束后的输出完整性核验期间完成；历史SHA亦进行了逐文件复核。所有正式模型评价保持GPU串行。

- `run_evaluation.py infer`：原官方函数体串行预测，严格加载两份权重，零训练。
- `audit_outputs_before_metrics.py`：所有场景文件/ID/frame/K检查，不计算误差。
- `run_evaluation.py score`：同一samples文件的官方CLI与独立评分。
- `verify_results_batched.py`：最终有效验收版本，全部文件SHA和一次批量Git忽略检查。
- `verify_results.py`：保留被中断的逐文件Git检查旧版本，作为本阶段工具修订证据，不用于后续验收。

配置、执行冻结commit、数据与checkpoint SHA见configs/stage7p3/evaluation_protocol.json和results/tables/stage7p3/execution_start.json。官方源代码未修改。
