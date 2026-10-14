# Stage 2 最小训练：真实运行记录

日期：2026-10-14，Asia/Shanghai。四个模型实际运行完成；没有未运行模型被填入数值，也没有正式测试折结果。

## 统一规格

ETH held-out，只从其余场景的 train/validation manifest 取固定子集：256个训练窗口、96个验证窗口。种子17，每模型5 epoch×32 batch=160次优化，batch=8，Adam lr=0.001，grad clip=1。hidden=64，TOHI H=3，图 radius=4m/k=8，模型 encoder/decoder 规格、loss和指标相同。没有增强、碰撞loss或采样选择，所有数值为deterministic K=1。

训练前源码、配置和结构测试已提交，实际来源 commit：`c2a04752ac2a5d3e713c060ccdcde49dab9f69c2`，启动时 tracked source 干净。GPU：NVIDIA RTX3080；Python3.10、PyTorch2.5.1+cu124。环境细节、运行开始时刻、种子、config/hash、manifest/hash和参数量完整保存在每模型JSON。

## 真实结果（最终固定epoch，未选最佳验证epoch）

| 模型 | 参数量 | 固定训练子集loss：初始→最终 | 验证ADE/m | 验证FDE/m | GPU allocated峰值/MiB | 训练循环/s | 包含诊断评价的总计时/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PedBase | 35352 | 1.538955 → 0.485743 | 0.518005 | 1.226333 | 36.82 | 4.64 | 6.00 |
| FirstOrder-GNN | 77722 | 1.538349 → 0.763259 | 0.929199 | 1.766975 | 64.94 | 11.01 | 14.03 |
| TOHI-Net-A | 120092 | 1.539515 → 0.745075 | 0.910711 | 1.763913 | 90.58 | 14.23 | 17.99 |
| Dynamic-GNN | 97881 | 1.536100 → 0.688737 | 0.822809 | 1.696738 | 76.10 | 9.93 | 12.50 |

loss是对每个有标签目标先平均有效未来步L2，再平均目标；初/终loss在同一固定256窗口完整评价，不把随机batch的loss变化伪称完整训练集loss。逐epoch的在线train loss及验证值在 epochs.csv。

验证ADE分母为1202个有至少一帧未来标签的人—窗口目标，FDE分母为839个最后预测帧有标签的目标。每次评价沿用相同mask和分母；不同于“完整20帧过滤+best-of-20”文献协议，不能直接比较文献指标。数据loader读取的是开发 train/validation，没有计算 ETH test 指标。

GPU allocated/reserved峰值为该进程在每模型运行前reset后的PyTorch分配器统计，不是nvidia-smi整卡显存，也不是单纯参数大小。训练循环计时含batch传输/forward/backward/优化；总计时另含初始训练集诊断、验证及最终训练集诊断，不含最初原始数据解析和模型初始化。

## 稳定性与限制

四模型训练、预测、反向梯度均有限，训练loss下降，CPU单元测试与真实GPU训练均通过。总参数不相等、只有一个seed、开发子集、5epoch；本轮仅验证可训练性，未证明性能优劣。CUDA scatter加法包含浮点原子操作，固定种子不保证逐位完全一致。

短训练中PedBase验证值明显低于三个社会模型；不能因此断言高阶机制无用，但当前没有“TOHI优于baseline”的证据，也不建议直接投入完整五折三种子。保持A1不变，先查融合是否保留运动状态、状态尺度、gate和梯度，再决定下阶段改动。

## 训练后状态诊断（同一开发验证，未再训练）

读取本地checkpoint，在历史输入上统计 cutoff 1226个人—窗口，既不读取future位置也不用GT筛人。TOHI 1/2/3阶支持比例约96.57%/96.25%/95.35%；平均L2范数4.02/7.88/18.25，而ego范数约2.57。平均融合权重0.156/0.752/0.058。

这些数值说明三阶不是仅仅因为路径不存在而失效；其尺度增长和较小融合权重值得研究。但状态范数与权重并不直接证明误差原因，更不能当成真实社会影响强度。后续可考虑受约束W、gamma、按阶尺度诊断和保留ego的融合；本轮未改公式、未额外训练，也未开发B。

## 结果与复现入口

- [summary.csv](../../results/tables/stage2_smoke_v1/summary.csv)：运行元数据与最终指标。
- [epochs.csv](../../results/tables/stage2_smoke_v1/epochs.csv)：20条实际epoch记录。
- [state_diagnostics.csv](../../results/tables/stage2_smoke_v1/state_diagnostics.csv)：阶支持、状态与融合权重。
- 每模型JSON和environment.json：配置与环境快照。
- `experiments/stage2_smoke_v1/checkpoints/`：本地权重，被忽略，不上传；轻量config.json允许提交。

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m src.training.smoke --run-id NEW_UNIQUE_RUN_ID
```

训练脚本拒绝覆盖已有run目录；复现前需先提交tracked源码/配置变更。运行顺序PedBase→FirstOrder-GNN→TOHI-Net-A→Dynamic-GNN；代码没有自动五折循环。
