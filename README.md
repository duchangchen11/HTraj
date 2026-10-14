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

Stage2.6完成H1/H2/H3/非严格同帧NoOrder及两个容量控制的18次独立训练，另6次完全核实的冻结参照；118项测试通过。三种子best ADE均值：H1 0.441364、H2 0.441229、H3 0.444220、NoOrder 0.445767m。H2收益不稳定；H3主ADE三个seed都弱于H2，对近参数H2-Cap两指标三个seed均更差。严格时间规则均值略好但seed方向混合。暂不以高阶优势作为第四章已验证贡献。见 [Stage2.6总报告](reports/stage2p6/STAGE2P6_REPORT.md)。

Stage3完成Receiver-Fix与Relation-Routing及四个独立训练消融，共六次seed17/30epoch，142项测试通过，旧440项SHA保护不变。主best ADE/FDE：Receiver-Fix 0.443099/1.074098、History-Routing 0.442996/1.073757m，均未超过Dynamic-RN 0.437891/1.050755m；NoUpdate优于History主指标。没有候选达到预设门槛，第二轮未启动，不宣称新模型跨seed稳定或独立确认。见[Stage3总报告](reports/stage3/STAGE3_REPORT.md)。

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
reports/{stage1,stage2,stage2p5,stage2p6,stage3}/ 阶段报告与诊断
tests/                   人工时间图、数据、mask与评价正确性检查
```

## 研究阶段

Stage1文献、Stage2原型、Stage2.5融合、Stage2.6独立阶数/时间与关系归因均已完成（开发范围）。当前高阶独立收益不足，后续先解决可靠消息接收和有效路径/尺度，再决定正式评估。B与多模态暂不开发，后续需用户指令启动。

## 当前运行方式

```bash
cd /home/lrj/HTraj
htraj_python=/home/lrj/anaconda3/envs/ped_intent/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$htraj_python" -m pytest -q
python3 scripts/check_stage1.py
python3 scripts/check_stage2.py
python3 scripts/check_stage2p5.py
python3 scripts/check_stage2p6.py
python3 scripts/check_stage3.py
python3 scripts/check_stage3p5.py
git diff --check
```

本阶段复用了已有 Python3.10/PyTorch2.5.1+cu124/pytest8.3.3 环境，未安装或改动旧环境。`configs/stage1_design.yaml` 保持为原始设计；实际 smoke 配置为 `configs/stage2_smoke.json`。

数据重建：`"$htraj_python" scripts/prepare_stage2_data.py --manifests NEW_MANIFEST_DIR`；覆盖统计：`"$htraj_python" scripts/interaction_statistics.py --output NEW_OUTPUT.csv`。两者拒绝覆盖既有产物。

最小训练入口：`"$htraj_python" -m src.training.smoke --run-id NEW_UNIQUE_RUN_ID`，只运行固定开发子集，不会自动五折训练；开始前需提交 tracked 源码/配置。已运行结果在 [summary.csv](results/tables/stage2_smoke_v1/summary.csv)，checkpoint 保留本地且忽略。重跑不是当前阶段必须执行的动作。

Stage2.5训练入口：`"$htraj_python" -m src.training.social_diagnostics --phase round1`，第二轮为`--phase round2`，使用预提交的`configs/stage2p5/protocol.json`和selection.json；既有结果目录存在时拒绝覆盖。本阶段已经运行完毕，不再次启动。表在`results/tables/stage2p5/`，best/last权重在本地`experiments/stage2p5/`，检查脚本需这些本地数据/权重以核实hash；GitHub不包含权重或数据。

Stage2.6入口：`"$htraj_python" -m src.training.order_verification`，协议`configs/stage2p6/protocol.json`；18次新训练已完成，既有run目录存在则拒绝覆盖。新表在`results/tables/stage2p6/`，best/last权重在本地`experiments/stage2p6/new/`。冻结参照的旧曲线路径和checkpoint/hash保留，不能把引用的六次旧run说成本阶段新训。

## 后续计划

独立阶数/时间规则、历史长度×密度和强关系干预已完成开发核查，缺少稳定高阶收益。下一阶段优先固定H1+R单独验证新进入目标接收、不同身份二阶路径/新鲜度选择、独立尺度变体，再决定是否保留二阶。三阶暂停作为核心性能贡献；补齐HoGAT/创新近邻文献。第五章只保留方向。

Stage3已核查接收阻断并修复，但与匹配H1比较没有明确新进入者ADE收益，完整历史训练后稍差；新增边记忆也未胜过无更新对照。目前停止增加模块，不启动第二轮或完整训练。未来如获授权，单独研究可靠源当前运动先验/消息新鲜度与不确定性，并先准备时间隔离的验证数据；原剩余非ETH验证没有不重叠支持。源入口为`"$htraj_python" -m src.training.relation_routing`，既有run目录拒绝覆盖，已运行六次不再次启动；结果在results/tables/stage3，权重仅本地experiments/stage3/new。

每个阶段完成本地提交；用户已在Stage2.5改为自行从终端上传，后续不自动push。冲突停止。旧项目保留。本仓库不包含凭据、数据集、权重或作者论文全文副本。

Stage3.5完成完整8/12协议A与原partial-mask协议B核查，五个非ETH录像隔离候选已统计，只students003原val70区域做四次seed17/30epoch基线。A PedBase/GNN ADE0.707297/0.721141m；B0.617815/0.615008m，分母不同不能横比。148测试通过，旧606项SHA保护；公开Social-STGCNN未复现、ETH封存，录像隔离不是独立确认。详见[Stage3.5总报告](reports/stage3p5/STAGE3P5_REPORT.md)。

当前建议先加强运动先验、在其它预定recording复核，再考虑冻结主路径的可撤回社会修正。入口`python -m src.training.benchmark_audit`已完成四个run，既有输出拒绝覆盖，不再次启动。真实表在results/tables/stage3p5，权重仅本地experiments/stage3p5/pilot；本阶段结束，等待新指令。
