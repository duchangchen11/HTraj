# 真实上下文资源与参数审计

官方版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；[源码与配置](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。本阶段模型执行代码版本 `6d07b7f4c697aa79142f383fe4040d133e1531b4`，逐文件SHA见 `results/tables/stage7p1/execution_setup.json`。

GPU：NVIDIA GeForce RTX 3080，9.65 GiB。执行环境 `experiments/stage4/venv/bin/python` 加 `experiments/stage7/cache/deps`；torch 2.5.1+cu124、numpy 1.26.4。完整pip依赖快照在 `execution_setup.json`。没有新增可学习层，原参数量 5,972,098；保留官方Hotel pre-CVAE配置，两录像共用，不含DLow阶段。

| 录像/帧(N) | 执行范围 | allocated峰值/MiB | reserved峰值/MiB | 整卡采样峰值/MiB | 时间/s | OOM |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel/14950(2) | K1+严格恢复 | 56.26 | 70.00 | 698 | 0.2248 | False |
| biwi_hotel/16170(15) | K1+严格恢复 | 62.18 | 76.00 | 704 | 0.2181 | False |
| crowds_zara02/8760(6) | K1+严格恢复+原K20 loss/反向 | 233.54 | 324.00 | 963 | 0.4164 | False |
| crowds_zara02/8620(9) | K1+严格恢复 | 89.40 | 116.00 | 755 | 0.2304 | False |
| crowds_zara02/9300(13) | K1+严格恢复+原K20 loss/反向 | 602.15 | 720.00 | 1359 | 0.4121 | False |
| biwi_hotel/1060(1) | K1+严格恢复+原K20 loss/反向 | 88.19 | 118.00 | 758 | 0.3975 | False |
| biwi_hotel/2770(2) | K1+严格恢复+原K20 loss/反向 | 115.58 | 146.00 | 786 | 0.4160 | False |
| biwi_hotel/590(3) | K1+严格恢复+原K20 loss/反向 | 146.73 | 166.00 | 807 | 0.4046 | False |
| crowds_zara02/990(1) | K1+严格恢复+原K20 loss/反向 | 88.19 | 122.00 | 763 | 0.4520 | False |
| crowds_zara02/10(2) | K1+严格恢复+原K20 loss/反向 | 115.58 | 146.00 | 787 | 0.3456 | False |

计时包括对应场景推理、可定义时的原loss/反向、随机checkpoint保存/读取及恢复推理；**不是纯训练吞吐**。allocated/reserved是PyTorch峰值；整卡显存为nvidia-smi每约0.1秒采样最大值，可能漏过短峰，包含驱动/其他进程；不冒充精确整卡峰值。

## 最大上下文

冻结来源支持最大N=73，`students001:30`，53个完整12帧GT目标，20个未来不完整节点；比Stage7人工N57多16人。未设人数上限、未裁剪。K20 learned-prior推理实际执行 0.3998s，无OOM；allocated峰值 2559.79 MiB，reserved峰值 9072.00 MiB，整卡采样峰值 9315 MiB。

该73人窗口来自已冻结训练来源，只做资源检查，不读取预测误差。因混合标签原训练语义不成立，**73人真实loss反向/optimizer训练显存没有验证**；不以K20推理无OOM外推训练可行。显存保留量已接近10GB卡容量，后续训练可行性仍有风险，但本阶段不据此删邻居或改结构。

## 历史完整上下文的信息损失

| 划分 | 仅有监督窗口的full8节点窗口 | 部分历史排除节点窗口 | full8覆盖率 |
| --- | --- | --- | --- |
| biwi_hotel/train | 37502 | 12235 | 75.40% |
| biwi_hotel/validation | 713 | 506 | 58.49% |
| crowds_zara02/train | 32920 | 11555 | 74.02% |
| crowds_zara02/validation | 1486 | 443 | 77.03% |

节点窗口分母为历史内至少一帧可见节点（含目标），仅计有冻结监督目标的窗口；不是独立行人覆盖率。完整支持区域的人数直方图与无监督窗口另保留在 `data_audit.json`。部分历史行人按预先历史规则排除，不伪造历史，不按未来缺失排除上下文；本轮不承诺对这类邻居充分利用。
