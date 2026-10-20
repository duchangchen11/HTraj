# Stage11B-1

本轮停止于真实K20可复现性门槛，正式训练0epoch。禁止把预算通过、553项单元测试通过写成真实GPU评价器通过。

配置冻结提交58c5a7211f2fe7a12785fe1ee26ab844af43ca6c；官方模型/函数只读复用Stage11B-0，MIT来源与固定版本见原adapter。本目录没有另写Y-Net结构或损失。训练配置没有因验证成绩改动。

执行顺序：保护历史→身份审核/分离artifact→人工指标测试→冻结配置→训练目标完整TTST K20预检→重复性诊断→失败关闭训练门禁→同目标CV/OLS→完整单元测试和历史SHA→报告。

已执行命令使用隔离环境：

```bash
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1.protect --freeze
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1.data
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1.precheck
experiments/stage11ap/cache/venv/bin/python -m scripts.stage11b1.baselines
```

结果为独占创建，不能直接覆盖既有输出重跑；首次/兼容预检失败分别保留initial_*和compat_*，第二次诊断错误日志也保留。重复性诊断程序sampling_audit.py记录5次固定输入；原始执行脚本快照与源码SHA留档。无正式训练入口被执行。

完整单元测试覆盖以下两组（合计553）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage11ap/cache/venv/bin/python -m pytest tests \
scripts/stage11b0/tests scripts/stage11b1/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage11ap/cache/venv/bin/python -m pytest \
scripts/stage11a/tests scripts/stage11ap/tests -q
```

禁用自动发现仅排除系统ROS插件的无关lark导入，不改变测试选择；初次插件失败原日志仍保存。

`baseline_predictions.csv.gz`全部先生成，之后才读取GT评分。Y-Net不存在验证预测/成绩/真实案例图。所有NOT_RUN结果留空，不填0。预算中上一阶段训练耗时与本轮推理耗时明确区分。最终checkpoint恢复未执行，不能复用旧实验的恢复结果冒充本轮。
