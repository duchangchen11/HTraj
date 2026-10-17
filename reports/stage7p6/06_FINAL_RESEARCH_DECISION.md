# 最终科研判定：NO-GO

停止AgentFormer“历史合法上下文+未来缺标签联合后验”这条具体创新开发路线；不自动开展Stage7.7，不增加GNN/attention/关系GRU/门控，不启动训练。保留原始AgentFormer作为已经复现的公开基准，保留所有历史失败与有效工程证据。

|门槛|状态|依据|
|---|---|---|
|真实问题得到现有数据支持|PASS|Stage7.1 mixed-label cohorts and posterior mask-only counterexamples; prevalence scope limited|
|与近年公开方法存在明确区别|NOT_MET|C/S separation and masked supervision established; no distinct new inference mechanism established; two requested full texts unresolved|
|数学训练目标自洽|PASS_CONDITIONAL|Observed-label conditional ELBO possible under stated likelihood/missingness assumptions; original loss is augmented, not exactELBO|
|不依赖虚构未来GT|PASS_DESIGN_ONLY|Omit U future tokens; U latents from history prior, no invented labels; not implemented|
|公平基线和消融|DESIGN_POSSIBLE_NOT_RUN|Separate complete20 main and C8 auxiliary tracks; retrain all protocol-matched baselines. No comparable results yet|
|资源预算具备可行性|NOT_ESTABLISHED|N73 inference only; original N57 training already9677MiB whole-card; no mixed-label training memory measurement|
|解释与官方基准差异|PASS|Unchanged target/K/units insufficient if context eligibility differs; separate rankings mandatory|


## 回答五个核心问题

1. 问题真实吗？是，Stage7.1实际窗口中存在历史完整但未来不完整邻居，原FutureEncoder mask-only负探针失败。没有证据说明新增邻居一定带来预测收益。
2. 已有论文解决了吗？宽泛的C/S分离、有效性mask与部分条件推断已有明确先例；精确到AgentFormer联合latent+缺失未来边缘化并未核实有完全相同实现。两篇指定近文完整方法仍不可得，不能以未发现当成新颖性通过，也不能宣布整个领域已无开放问题。
3. 有区别的数学机制吗？本轮给出了自洽但受限的变分构造，U从prior、S仅用真实标签。它体现常规缺失变量处理，可能产生较松ELBO；尚没有不同于已有思想且能解释预测收益的新机制。后验/身份映射/聚合需改，不能称纯适配。
4. RTX3080能做吗？原协议训练已实测，73人仅推理通过。新混合标签训练显存、完整epoch和可承担消融预算未建立，不宣称已不可运行，也不能填写资源GO。
5. 能保留标准主协议吗？可保留官方完整20帧五折主表，把C8实验另列；C8内必须重新训练匹配基线。仅8/12、K20相同不足以严格公平，不能把不同上下文条件下的结果直接排名。

## 为什么不启动一个“先试试看”训练

GO是所有条件同时成立，不能由“问题真实”替代“新颖性、独立归因、资源可行”。此时训练一个修改后模型即使提升，也难区分上下文更多、后验变更、目标权重与不同训练分布。没有独特机制支撑，训练预算投入无法补足研究论证。

本轮没有证明原AgentFormer预测失败，没有推翻Stage7.3权重复现，没有将未完整future节点补成静止。原Stage7.1在“保持核心语义不变”限制下NO-GO仍然正确；本轮允许数学讨论也不等于那次准入突然成功。

## 论文方向调整建议（仅建议，等待用户决策）

优先将已积累材料整理为硕士论文中的**协议边界与可复现性/动态观测鲁棒性评价**：明确离线上下文资格、历史可见上下文、有效标签和部署输入之间的边界，保留真实反例、同目标CV和公开模型复现。这是可以证据化的研究内容，但单靠协议重命名和mask工程未必达到独立算法小论文要求。

若第四章必须有新算法，先由用户决定是否调整数据任务或重新立项：需要有原生可见性/标签缺失标注、能排除评测与选择偏差的数据，并把自然缺失与人工mask区分。不能在目前已充分曝光的ETH/UCY小开发集上连续换模块直到出现最低误差。该建议不是新模型方案，不宣布数据已取得、算法已实现或性能已提升。

当前不具备本具体AgentFormer创新路线的正式训练启动条件。不再自行更换模型或调整运动特征/候选数量。没有新增ADE/FDE，未运行项明确保留未验证。
