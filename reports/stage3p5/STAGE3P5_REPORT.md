# Stage3.5：数据协议核查与社会交互价值总报告

项目/home/lrj/HTraj；分支feat/stage3p5-benchmark-audit；基础46ecb91169eeadb5ba7ec28b7c648dd0ace837c6。606项Stage1–3源码/脚本/配置/报告/结果/权重SHA保护，90个旧checkpoint未改变；未使用旧项目自定义模型或缓存，只本地提交。

## 实际完成

明确A_complete8x12_history_context完整8/12任务与B_partial_mask_v1原部分观测任务；world米/0.4s/K1，不混分母。标准公开Social-STGCNN/SGAN采样/筛选/划分及MIT代码已核查；固定源码与作者稿SHA保留，外部模型**未复现**、没有虚填其结果或公平排名。官方split文件清单已核实，但processed内容/边界未复现；其ETH fold额外使用crowds_zara03与uni_examples，训练人口也不同。只读取文件树，未读取ETH测试轨迹。

五个非ETH录像隔离候选都有真实窗口/目标/历史/密度/运动统计；首轮只students003原val70窗口，train仅其它四recording原train，held原train413舍弃，原val未加入训练。A2058训练窗口/19034目标→706验证目标；B2365/31458 ADE、23423 FDE训练目标→1464 ADE/986 FDE验证目标。与旧96有11个相同窗口且旧研究看过这些录像，所以不是独立确认。

四个神经baseline seed17从头30epoch/120epoch，加两套CV解析评价；训练源5771f31df0ae863daf70307a9f9b6c73f232ee68，旧B三seed只完全兼容冻结重评，不算新训练。148项测试训练前通过，最终测试记录在unit_tests.xml。

| 协议 | 模型 | 参数 | best epoch | best ADE/FDE(m) | last30 ADE/FDE(m) | 训练秒 |
|---|---|---|---|---|---|---|
| A_complete8x12_history_context | CV | 0 | 解析 | 0.727884/1.627788 | 不训练 | 0 |
| B_partial_mask_v1 | CV | 0 | 解析 | 0.635479/1.692958 | 不训练 | 0 |
| A_complete8x12_history_context | PedBase | 35352 | 1 | 0.707297/1.524885 | 0.720126/1.563693 | 216.52 |
| A_complete8x12_history_context | Dynamic-GNN-RN | 102235 | 3 | 0.721141/1.560097 | 0.751859/1.623931 | 480.12 |
| B_partial_mask_v1 | PedBase | 35352 | 11 | 0.617815/1.614656 | 0.632383/1.653170 | 241.31 |
| B_partial_mask_v1 | Dynamic-GNN-RN | 102235 | 12 | 0.615008/1.611574 | 0.659313/1.716698 | 551.43 |

源manifest SHA a7698e867ac7b7163dee035382eab0ebccb94a733e51955c4e3461d3ea5a3cf5；旧manifest SHA c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced；protocol SHA 10fefd17291f655b8a004f0560cce1feb9b7f0a4803315f9a3f26aa046b3558b。各config含源码/data/index SHA、参数、checkpoint SHA与环境。

## 归因与第四章判断

新pilot全部历史分组和同target误差配对保存，不只报改善。A GNN−Ped ADE/FDE=+0.013845/+0.035211m，B=-0.002807/-0.003082m；单recording/seed，且容量不匹配，不能证明普遍独立社会价值。旧B ZARA2两指标3/3方向受益，交叉FDE3/3但ADE2/3；无邻居、2–4帧、变化关系组3/3退化。此为待新开发复核假设，非独立验证。

关键口径补充：B合法空图51目标中47有原始4m物理邻居、46有可靠源，不能把接收被阻断称为没有社会信息。新B部分历史有条件受益，与旧B退化方向不同；A高密度/同向组CV略优神经模型。关系变化A0/B5且只有1个行人，无稳定结论。随机图实际只换A3.059%/B5.042%，完整支持再留8秒隔离仅2窗口，不足独立确认。共同706完整目标对比排除了人群差异，A/B训练与上下文仍不同。

训练后物理密度探索补充保留四组全量配对：真实历史无4m可见邻居A0/B2，无法判断孤立行人一般规律。A物理低/中/高密度均退化，B低/高小改善、中密度退化，不能按密度直接宣称社会信息有效。

模型无边时回到自己的ego，却不等独立PedBase权重，主路径共同优化是关键混杂；固定checkpoint删消息/合法随机图只是依赖证据。标准A应保留作主研究任务，B保留作独立鲁棒设置；暂不开发复杂社会结构，建议下一步先验证强运动先验与冻结主路径的小型、可撤回社会修正，已有思想不改名宣称创新。

旧30epoch与新更多训练步数都不证明收敛，早期best/后续val退化真实保留。首轮实际训练24.82分钟；全pool/60/100/多个fold预算仅估算，RTX3080实测基线内存可行。没有ETH调参、其它fold训练、独立确认或公开模型训练。

## 文件

reports/stage3p5的01_PROTOCOL_AUDIT.md、02_BASELINE_COMPARISON.md、03_INTERACTION_VALUE_ANALYSIS.md、04_TRAINING_FEASIBILITY.md、05_CHAPTER4_RESEARCH_DECISION.md及本报告；00_PRETRAINING_PLAN.md保留预提交设计。results/tables/stage3p5保存所有真实配置、split/group counts、30epoch曲线、best-last逐目标误差与分组、配对/共目标对比、历史三seed审计、图介入、hash、测试、预算及保护清单。未运行实验明确为估算/未复现，不填假结果。

用户自行git push -u origin feat/stage3p5-benchmark-audit；完成本地commit后停止，等待下一指令。
