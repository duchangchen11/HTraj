# Stage7.5 AgentFormer缺陷与创新可行性总报告

**结论：本轮创新开发NO-GO。当前未找到值得启动训练的创新方案。** 保留可靠AgentFormer基线；本轮没有训练新模型、没有修改官方模型/权重、没有重算test成绩。

本报告仅分析Stage7.4保存的官方validation预测，不重新评价test。官方源码e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5，作者CVAE+DLow、mean=True、K20、8/12帧、0.4s、米制、完整20帧离线上下文资格均保持不变。协议在034727969d0892632889bf5e038c0fac5f892d53提交；原始文件数值列修复在350aceb提交后执行。未训练、未更新模型参数、未增加模块。

HOTEL/ZARA2是权重fold名称；validation分别来自其余7个来源录像的_val区域，并非对应held-out录像的test。两折共享3944个目标窗口，history/GT SHA完全相同，不能作为两个独立确认集。窗口加权、来源录像内行人ID等权和逐来源结果分别保存。相同行人可跨历史运动组出现，组内人数不可相加。旧test已经被观察过，未来不能称为完全未接触的独立确认集。

完整数值及有效方向样本数见[complete_group_statistics.json](../../results/tables/stage7p5/complete_group_statistics.json)，原始逐目标/逐head表见同目录。早期group_statistics.json可能漏掉首行未定义的方向列，已明确由complete版本替代；旧文件和失败记录保留。空值代表方向不可靠或无适用样本，不填0。

## 真实执行范围

重读并核验Stage7.4全部1278个validation预测文件，HOTEL 688窗口/5203目标、249个来源行人ID；ZARA2 590窗口/4262目标、237个来源行人ID。共有3944目标窗口；使用7个来源录像分解，不能以9465个窗口目标宣称9465个独立行人。CPU统计实测111.57秒，本阶段未使用GPU执行正式推理或训练。模型人工前向仅限完整单元测试。

## 三个研究问题

- 运动：窗口加权下，整体高速路程归一化ADE未劣于正常组；行人等权下ZARA2略高，HOTEL略低，没有跨口径一致劣势。同一biwi_eth_val中高速仍有局部相对误差和速度偏低。全部高速只有51个共享窗口/21人，不能泛化。没有发现跨运动状态统一速度偏差，也未验证一致转向不足。
- 多候选：近重复对约1.21%/1.15%；K10到K20仍带来约26.5%/26.9%的固定排列平均ADE下降。最高head胜出占比约18%/11%，不是相同少数head垄断。oracle覆盖不是在线排序/选择能力。
- 社会：原网络已有全时空agent-aware attention与历史条件化DLow；没有手工相对边特征不构成新模块必要性。邻居统计只有描述性相关，独立社会贡献无法判断。

## 文献与候选

核查8篇，6篇取得全文并阅读方法/实验；SocialTrans2025、Inter-GraphFormer2026已核实身份和公开方法概要，但完整公式/协议未取得，不能冒充精读完成。全文/接口受限均已留档。TUTR、MoFlow、SocialCircle、Joint Metrics Matter已覆盖候选概率、多样性、显式社会特征、联合度量等相近方向。

最多列出1项未获批准的q_mlp历史速度摘要草案；它工程改动小，但更像普通特征增强，来源证据和新颖性不足，推荐启动训练的方案数为0。

## 交付索引

1. 01_AGENTFORMER_ARCHITECTURE_AUDIT.md：原始信息流、配置、源码定位。
2. 02_MOTION_ERROR_ANALYSIS.md：速度/转向/尺度及来源敏感性。
3. 03_MULTIMODAL_CANDIDATE_ANALYSIS.md：固定K子集、head贡献、相似性。
4. 04_SOCIAL_INTERACTION_DIAGNOSIS.md：历史关系描述与归因边界。
5. 05_RECENT_LITERATURE_REVIEW.md：论文方法、输入输出、loss、协议和重合风险。
6. 06_INNOVATION_CANDIDATES.md：1项暂缓草案、消融及否定条件。
7. 07_FINAL_RESEARCH_DECISION.md：NO-GO和替代建议。

结果位于[results/tables/stage7p5/](../../results/tables/stage7p5/)：逐目标/逐head/逐时间步误差，历史组、来源、行人等权，固定K曲线，文献矩阵，来源/权重SHA及全部失败与测试记录。canonical统计为complete_group_statistics.json。图表数据均可从这些真实表复核，本轮未制作虚构或示意性能图。

全部9465个目标窗口的ID、历史与GT均与Stage7.4一致，重算指标最大差异7.11e-15m。Stage7.4数组SHA包含dtype/shape，本轮逐目标表SHA只哈希float32字节；target_provenance.csv.gz保留两种明确命名的SHA，逐一重验，不把不同哈希编码误报成GT变化。检查见stage7p4_target_metric_equivalence.json。

保护检查87992次SHA通过，其中12924项旧文件保护、73656个旧test输出仅按不透明字节验SHA，不解析旧test误差。单元测试433通过。初次数值列解析及可空字段导出失败均修复并保留现场，不改既定组、阈值、模型或历史指标。

分支feat/stage7p5-agentformer-innovation-study。本地提交后停止，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p5-agentformer-innovation-study
```
