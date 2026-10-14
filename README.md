# HTraj

硕士大论文：**《面向复杂场景的行人轨迹预测关键技术研究》**。

## 研究背景与目标

复杂人群中的运动不仅受相邻行人影响，也可能受间接、跨时间的社会交互影响。本项目研究运动状态编码、高阶社会影响传播及后续多模态协同预测，目标是建立可复现、可消融、具有清晰计算结构的预测方法。所有贡献均须经实验验证。

## 当前研究方法

Stage 1 已完成 HighGraph、UniEdge 原文分析和候选结构设计。HighGraph 的研究问题是不同社会距离上的间接影响；UniEdge 已覆盖跨时间交互和边间传播，不能把这些概念单独当作新贡献。当前暂名 TOHI-Net（Time-Ordered Higher-order Interaction Network）：

- A：按社会阶数分离的时间递推传播，Stage 2 已独立实现并通过结构测试及短训练。
- B：时间相容交互事件 DAG 与路径状态，仍为未实现的设计候选。

Stage 2 同时实现 PedBase、FirstOrder-GNN 和 Dynamic-GNN。59 项测试通过；四模型完成同一开发划分的 5 epoch smoke。当前 TOHI-A 尚未显示优于基础网络，不能以这次短训练宣称性能优势。见 [Stage 1](reports/stage1/STAGE1_REPORT.md) 和 [Stage 2 报告](reports/stage2/STAGE2_REPORT.md)。

Stage 2.5 已保留A0并实现R/N/RN、普通社会RN与ego扩宽诊断。19次固定预算训练完成，全部87项测试通过。1024窗口三种子best开发ADE均值：PedBase 0.439398m、Dynamic-GNN-RN 0.440075m、TOHI-A-R 0.442214m；尚无稳定整体ADE优势。98条固定checkpoint控制记录包含full和明确的no-op；二阶有条件增益，三阶较小，独立高阶/时间约束贡献未证明。详见 [Stage 2.5 总报告](reports/stage2p5/STAGE2P5_REPORT.md)。ETH测试集没有评价。

## 数据集

ETH/UCY 的六个公开轨迹 TXT 已核查 SHA256 并在本地复制，独立重建五场景留一 manifest、观察 8 帧预测 12 帧；历史行人集合与未来监督 mask 分开。没有复用旧项目预处理缓存。原始数据置于 `data/raw/`，缓存置于 `data/processed/`，均不上传。详情见 [数据协议](reports/stage2/01_DATA_PROTOCOL.md)。SDD 尚未开展，米制与像素不能混算。数据分发与许可不因上游代码 MIT 许可自动覆盖。

## 项目目录

```text
configs/                 设计配置与后续实验配置
data/{raw,processed}/    数据与缓存（忽略）
src/data/                数据读取与预处理
src/models/              基础网络与候选交互模块
src/training/            后续训练入口
src/evaluation/          后续评价与诊断
scripts/                 仓库检查和辅助脚本
docs/literature/         文献、来源与比较
docs/method_design/      数学设计、架构比较与实验协议
docs/thesis/             大论文框架与研究进度
experiments/             轻量实验配置；运行缓存不上传
results/{tables,figures}/ 轻量汇总表和完成图片
reports/{stage1,stage2,stage2p5}/ 阶段报告与诊断
tests/                   人工时间图、数据、mask与评价正确性检查
```

## 研究阶段

Stage 1：证据核查与结构设计；Stage 2：数据、对照与A原型；Stage 2.5：融合诊断与固定开发集三种子验证（2026-10-14完成）。目前先补独立高阶归因，暂不进入五场景正式评估。B与多模态方向暂不开发。后续阶段需用户指令启动。

## 当前运行方式

```bash
cd /home/lrj/HTraj
htraj_python=/home/lrj/anaconda3/envs/ped_intent/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$htraj_python" -m pytest -q
python3 scripts/check_stage1.py
python3 scripts/check_stage2.py
python3 scripts/check_stage2p5.py
git diff --check
```

本阶段复用了已有 Python3.10/PyTorch2.5.1+cu124/pytest8.3.3 环境，未安装或改动旧环境。`configs/stage1_design.yaml` 保持为原始设计；实际 smoke 配置为 `configs/stage2_smoke.json`。

数据重建：`"$htraj_python" scripts/prepare_stage2_data.py --manifests NEW_MANIFEST_DIR`；覆盖统计：`"$htraj_python" scripts/interaction_statistics.py --output NEW_OUTPUT.csv`。两者拒绝覆盖既有产物。

最小训练入口：`"$htraj_python" -m src.training.smoke --run-id NEW_UNIQUE_RUN_ID`，只运行固定开发子集，不会自动五折训练；开始前需提交 tracked 源码/配置。已运行结果在 [summary.csv](results/tables/stage2_smoke_v1/summary.csv)，checkpoint 保留本地且忽略。重跑不是当前阶段必须执行的动作。

Stage2.5训练入口：`"$htraj_python" -m src.training.social_diagnostics --phase round1`，第二轮为`--phase round2`，使用预提交的`configs/stage2p5/protocol.json`和selection.json；既有结果目录存在时拒绝覆盖。本阶段已经运行完毕，不再次启动。表在`results/tables/stage2p5/`，best/last权重在本地`experiments/stage2p5/`，检查脚本需这些本地数据/权重以核实hash；GitHub不包含权重或数据。

## 后续计划

时间前缀与正逆序已验证；下一阶段优先同融合/近容量的H=1/2/3独立训练、时间约束单因素、历史长度×密度和更有力合法负对照，避免将条件敏感性当成独立贡献。补齐HoGAT全文与近期时间图文献。第五章只保留方向，不预设最终模型。

每个阶段完成本地提交；用户已在Stage2.5改为自行从终端上传，后续不自动push。冲突停止。旧项目保留。本仓库不包含凭据、数据集、权重或作者论文全文副本。
