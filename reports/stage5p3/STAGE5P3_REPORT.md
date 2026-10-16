# Stage 5.3 社会信息归因总报告

两个录像在两种权重均达到预设K1门槛，值得进一步验证，但容量混杂未排除，不能宣称社会信息独立有效。

两个NoSocial模型均以seed17全新初始化，分别完整训练100epoch；Full严格复用Stage5.2冻结结果，官方源码未改，未增加新图模块/损失。8/12帧、0.4s、米制因果协议、训练/目标/GT、选择与采样完全配对；ETH未评价。

| 录像 | 模型 | best epoch | K1 ADE/FDE | minADE20/minFDE20 | 实际分钟 |
| --- | --- | --- | --- | --- | --- |
| HOTEL | Full | 3 | 0.3872/0.7515 | 0.2025/0.3253 | 38.98 |
| HOTEL | NoSocial | 2 | 0.3952/0.7778 | 0.2461/0.4166 | 33.66 |
| ZARA2 | Full | 30 | 0.2726/0.6075 | 0.1575/0.3048 | 34.78 |
| ZARA2 | NoSocial | 55 | 0.2789/0.6134 | 0.1551/0.2942 | 30.24 |


K1与K20不能混用；同模型各项来自同一K1 ADE选择的best。last和首采样均保留在04及真实结果文件。全部CV-Last2/OLS4/OLS8见04，不选弱基线突出神经网络。

## 交付索引

- `01_SOCIAL_INFORMATION_PATH_AUDIT.md`：源码路径、分层信息流、未来邻居实际mask和Jacobian detachment限制。
- `02_NOSOCIAL_IMPLEMENTATION.md`：最小功能修改、训练前冻结协议及停止门槛。
- `03_CAPACITY_AUDIT.md`：名义/有效/梯度容量，独立容量控制设计及未运行边界。
- `04_TRAINING_RESULTS.md`：两录像Full/NoSocial/CV的best/last、各预测口径、实际耗时、曲线/泛化。
- `05_INTERACTION_GROUP_ANALYSIS.md`：全部历史分组、窗口与行人两种权重，含空组和退化组。
- `06_RESEARCH_DECISION.md`：预设规则结论、容量混杂、下一步和停止。
- `results/tables/stage5p3/`：代码/数据SHA、逐轮逐batch/逐参数梯度、逐目标配对误差、全部组/宏平均、checkpoint SHA、测试和日志。

## 验收和限制

完整单元测试与所有原始失败记录按实际执行保存。`delivery_verification.json`给出旧文件SHA保护、完整训练预算、200份新checkpoint校验和测试结果；不得将结构测试等同预测性能验证。Stage5.2 CUDA跨batch拆分毫米级数值限制继续保留，评价固定相同batch128/目标顺序/目标epsilon，不修改容差或指标挑选。

没有独立确认集、额外seed或容量控制性能，不能宣布独立社会因果贡献或第四章新创新成立。本阶段无Ego-Capacity-Control训练，未运行项不填0。只完成本地commit，用户自行`git push -u origin feat/stage5p3-social-attribution`；完成后停止。

## 最终实测补充

严格NoSocial的训练后输入隔离检查全部通过；历史SHA及正式训练验收见`delivery_verification.json`。容量控制未训练，社会收益不能独立归因。正常行走的指定问题已在05/06列出同目标CV/Full/NoSocial及两种权重，不能将相对CV收益直接当社会贡献。

## 主结果的证据边界核对

主K1 best规则下Full在两个录像及两种权重均达到预设继续核查门槛。窗口ADE优势仅ZARA2约6.3mm（2.26%）、HOTEL约8.0mm（2.03%）；这是开发结果，不是显著性结论。Full相对NoSocial的优势不能排除585792个结构有效参数上界差异，不能直接命名为社会信息收益。

K20窗口口径：HOTEL Full优于NoSocial（0.2025/0.3253 vs 0.2461/0.4166m），ZARA2 Full反而较差（0.1575/0.3048 vs 0.1551/0.2942m）。ZARA2行人等权K20 ADE略有Full优势、FDE略退化；这与整体K1同方向收益不是同一结论。单首采样也有权重/录像冲突，均已保留。

正常行走问题：ZARA2无需邻居时仍保留窗口口径约74.8%、行人等权约58.9%的相对Last2改善；Full只比NoSocial进一步降低ADE约0.0210/0.0401m。该比例是描述性差值，不是社会因果贡献份额。HOTEL正常组Full 0.7992m、NoSocial 0.8925m都弱于Last2 0.5333m；移除社会输入未消除退化，故不能将原退化简单归因为社会分支，Full在该组反而较小。这也不证明个体主干/后验目标的具体失效原因，仍需独立核查。

同一目标下仍有退化组：Full历史静止组相对NoSocial在HOTEL/ZARA2的ADE分别增加0.0689/0.0037m；HOTEL无合法邻居组增加0.0115m；ZARA2新进入邻居及速度不可观测组增加0.0073/0.0107m。低速组两录像ADE小改善、FDE却小退化。不能只展示正常行走组的成功案例。HOTEL静止组106窗口只有4个ID，正常组91窗口覆盖24个ID，因而窗口与行人等权强调的对象不同；模型间目标集合本身完全相同。

按两个录像各1/2的宏平均，best窗口K1为Full 0.3299/0.6795 vs NoSocial 0.3370/0.6956m；Full仍不如Last2宏0.3129/0.6393m。last100时ZARA2 NoSocial K1明显好于Full，而HOTEL Full较好，说明主best规则结论不能推广到所有训练终点。last宏FDE也出现Full小退化。全部曲线/终点已保留，没有反向修改选择规则。

下一步仅值得先核查容量/训练泛化，不能据此增加GNN、高阶、关系GRU、注意力或门控。Ego-Capacity-Control的提案不是已实现/已复现结果；其额外ego路径改变归纳偏置，若未来实施仍需独立结构与梯度测试。在性能容量混杂未排除、无额外seed/独立确认时，不确认第四章新方法路线或社会因果收益。本阶段停止，不自动追加训练。
