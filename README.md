# HTraj

硕士大论文：**《面向复杂场景的行人轨迹预测关键技术研究》**。

## 研究背景与目标

复杂人群中的运动不仅受相邻行人影响，也可能受间接、跨时间的社会交互影响。本项目研究运动状态编码、高阶社会影响传播及后续多模态协同预测，目标是建立可复现、可消融、具有清晰计算结构的预测方法。所有贡献均须经实验验证。

## 当前研究方法

Stage 1 已完成 HighGraph、UniEdge 原文分析和候选结构设计。HighGraph 的研究问题是不同社会距离上的间接影响；UniEdge 已覆盖跨时间交互和边间传播，不能把这些概念单独当作新贡献。当前暂名 TOHI-Net（Time-Ordered Higher-order Interaction Network），两个候选仍是未实现的研究设计：

- A：按社会阶数分离的时间递推传播，作为优先实现方案。
- B：时间相容交互事件 DAG 与路径状态，作为有条件的独立论文候选。

详见 [Stage 1 报告](reports/stage1/STAGE1_REPORT.md)。目前没有模型实现、训练结果或可运行训练入口。

## 数据集

计划先使用 ETH/UCY，遵循五场景留一评估、观察 8 帧预测 12 帧；随后使用 SDD 及固定划分。ETH/UCY 的米制坐标与 SDD 的像素坐标不得混算。数据未下载、许可与划分尚待登记。原始数据置于 `data/raw/`，缓存置于 `data/processed/`，均不上传。

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
reports/stage1/          本阶段总报告与仓库审计
tests/                   后续结构与评价正确性检查
```

## 研究阶段

Stage 1：证据核查与候选结构设计（2026-10-14）；Stage 2：基础网络与 A 核心模块原型；Stage 3：规范评估与消融；Stage 4：B 可行性或结构修订；Stage 5：多模态协同方向及论文写作。后续阶段需用户指令启动。

## 当前运行方式

```bash
cd /home/lrj/HTraj
python3 scripts/check_stage1.py
git diff --check
```

检查脚本仅验证交付文件、相对链接、Git 忽略规则和文件风险，不下载数据、不训练模型。候选 YAML 为设计规格，不是训练程序支持的配置；本阶段不安装深度学习环境。

## 后续计划

先验证时间前缀不变性与“正序可达、逆序不可达”例子，再实现同一解码器下的 A、静态多跳和动态 GNN 对照。补齐 HoGAT 全文、检查近期时间图与行人预测工作，完成公平实验后再决定贡献表述。第五章保留研究方向，不预设最终模型。

每个阶段提交并推送；冲突停止，认证失败保留本地提交。旧项目保留。本仓库不包含凭据、数据集、权重或作者论文全文副本。
