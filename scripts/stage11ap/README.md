# Stage11A+复核入口

本阶段只做作者资源补齐/协议与工程准入，不训练、不计算test或validation排名。用户明确授权作者Drive约949M包及≤1GPU小时工程测试，没有授权69GB视频。

目录：源码/报告/配置/轻量证据限四个stage11ap目录；ZIP、权重、原图、mask、pkl、venv、论文及第三方代码全在忽略的`experiments/stage11ap/cache/`。同名JSON/CSV证据默认exclusive创建，失败初始stdout保留。不要直接删除重跑。

执行顺序：

1. `protect.py --freeze`：继承Stage11A保护表，并保护旧cache，88,326文件。
2. 新venv装gdown；`download_author.py`仅作者指定ID。首次中断后`--resume`正常HTTP206续传。
3. `archive_audit.py`先校验路径/空间/压缩风险，再按资源类型提取，不执行任意脚本。
4. `resource_audit.py`精确白名单读DataFrame，分别核验split、20帧支持、图像与mask；test只做身份/时间/SHA，不做预测。
5. `prepare_runtime.py`复制并核对冻结源，使用新venv小包+只读pth复用stage4 Torch2.5.1。没有向旧环境安装包。
6. `pipeline_admission.py`：六个train样例，正式原权重strict加载，完整分割/TTST/decoder；`counterfactual.py`检查未来不进推理。
7. `alignment.py`：六组train真实对齐图，GT只供该离线脚本。
8. `protocol_candidate.py`：NOT_READY候选；`literature_update.py`：三篇有限查重；`legacy_source_audit.py`：旧库源码变更记录。
9. 完整pytest、报告、`protect.py`、交付SHA、diff检查、commit。

完整单元测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage11ap/cache/venv/bin/python -m pytest \
tests scripts/stage11a/tests scripts/stage11ap/tests -q
```

`adapters.py`仅替代数据容器/concat/collate，不改官方模型和损失。`checkpoint_io.py`先weights_only；旧分割完整对象必须经已校验作者SHA和固定类白名单，不允许任意GLOBAL。YNetTorch先构建原核心，再装入安全恢复的相同分割对象，最终整个state dict strict加载。旧类源码差异12/17真实保留，不能称原Torch1.5逐位复现。

GPU两进程不足10秒墙钟，B1前向不代表训练预算；Y-Net工程入口没有optimizer.step；完整旧pytest包含临时人工模型的优化器单测，非真实训练实验。当前整体NOT_READY/NO-GO，不自动启动下一阶段。具体事实以results中的JSON与报告为准。
