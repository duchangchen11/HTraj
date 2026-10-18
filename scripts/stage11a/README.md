# Stage11A：执行与复核入口

此阶段不训练模型、不做真实预测性能评价。Python使用现有只读`experiments/stage8/cache/venv/bin/python`，对应Torch2.5.1；没有安装/降级共享依赖。完整结果含实际失败，不能把合成core forward当完整Y-Net通过。

新增代码/配置/报告/轻量结果只在四个`stage11a/`目录；原始镜像样例、官方代码、论文与Git对象留在被忽略的`experiments/stage11a/cache/`。不加载其他旧项目的模型。SDD样例23.19MiB，六场景六录像，无视频。作者约949M补充包未下载，完整视频包69GB未下载。

主要顺序：protect.py --freeze → fetch_source.py / fetch_sdd_samples.py → alignment_audit.py → baseline_probe.py / synthetic_gpu_probe.py → draft_protocol.py / literature_matrix.py / inventory.py → write_reports.py →完整pytest→protect.py→seal.py。

工具写JSON/CSV默认exclusive；已有结果不会由脚本静默覆盖。复核旧结果优先核对SHA；重跑应在独立输出副本进行，不删除旧文件。初版图形可读性不足，原图与原统计保留在alignment_figures_initial及*_initial，终版用于报告。

完整单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage8/cache/venv/bin/python -m pytest tests scripts/stage11a/tests -q
```

人工可视化仅为真实8点历史：首次可获得完整8点、净位移≥20px的轨迹，非预测误差/模型分组。未来训练协议仍是DRAFT，不能拿六例拼成训练集或视作官方split。

许可证：作者模型MIT（见license_audit）；SDD原始及本轮派生图遵守官网CC BY-NC-SA3.0，派生图声明在结果图目录。论文全文不入Git。来源和完整访问失败均保留，不把MCP未挂载、OpenAlex429或出版方403隐去。
