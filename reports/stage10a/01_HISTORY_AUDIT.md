# Stage10A：历史研究审计

本轮基于已推送Stage9 `d8e753f718647db49202da3ac6d43e4b37cfc59b`；本地与origin无历史差异，工作区原本干净。冻结旧结论，重新授权可控图关系实验并不倒改旧阶段NO-GO。不把失败解释为网络“不够复杂”。

## 已完成过的动态图与失败边界

|证据|已做的结构/实验|真实结论与限制|
|---|---|---|
|[Stage2.6](../stage2p6/STAGE2P6_REPORT.md)、`src/models/temporal_graph.py:select_candidates`|4m/8邻居，历史相对位置/相对速度、有限4.8s CPA、按`DCA+.05*distance`排序；MLP社会权重；Dynamic-GNN-RN、独立H1/H2/H3及容量控制|3seed，1024训练/96开发窗口，30轮；Dynamic-RN ADE .440075±.002073，PedBase .439398±.003578，未建立稳定ADE增益。H3弱于H2；干预改变预测说明计算依赖，不能证明独立泛化价值。此处数值仅引用旧任务，不是新任务成绩。|
|[Stage3](../stage3/STAGE3_REPORT.md)、`src/models/relation_routing.py`|分离receiver/carry连续性，未知速度标记、位置回退；逐帧关系、关系GRU、均匀/随机关系控制|动态关系历史、缺失运动处理、关系门控均非本轮首次出现。旧种子/小预算/重用开发集结论不外推全部社会模型。|
|[Stage4.2最终审计](../stage4p2_final/FINAL_AUDIT_REPORT.md)|消除节点索引卷积，匹配逐人时间归一化PTN；G0/G1同预算从30延长到100轮，配对三CV|ZARA2 G0/G1 ADE .514221/.605474，CV-Last2 .325740；HOTEL .980086/.933634，CV-OLS8 .241865。两个神经模型均未超过三CV，G1只HOTEL改善G0。不是所有GNN的否定。|
|[PTN审查](../stage4p2_final/04_PTN_SENSITIVITY_AUDIT.md)|全体目标、固定权重1mm扰动，静止/低速/正常运动分组|局部PTN高增益及静止目标重复敏感性成立；G0也有，不能归咎社会边或证明PTN唯一因果责任。窗口重复会放大少数长轨迹影响。|
|[Stage5.1](../stage5p1/01_CAUSAL_DATA_PROTOCOL.md)、[原始输入反事实](../stage5p1/02_RAW_INPUT_LEAKAGE_AUDIT.md)|严格8帧历史；后向速度差分、显式有效性；未来邻居padding；训练/验证录像隔离|旧官方SocialVAE loader首出现速度补值确有未来数值污染反例；新B适配阻断该路径。未来监督目标资格与输入泄漏必须分开。|
|[Stage6](../stage6/STAGE6_REPORT.md)|Full/Placebo多seed结果、HOTEL运动误差、SGCN代码检查|Full社会内容优势未跨录像/seed通过；HOTEL持续运动的预测速度偏低是描述性证据，不证明唯一原因。SGCN空间mask索引问题成立，TXP节点卷积猜测已被纠正。|
|[Stage7](../stage7/STAGE7_REPORT.md)、后续AgentFormer协议|换序、梯度、单/多人和完整20帧上下文审查|官方离线资格可依赖未来存在性，不能冒充严格历史协议。排列等变与准入测试是工程性质，不是创新。AgentFormer公开基准成功复现继续保留，本轮不读取其test误差。|
|[Stage8](../stage8/STAGE8_REPORT.md)|Resonance/MoFlow协议、配对歧义及公开研究查重|有表示歧义反例，但没有预测损害与新颖性证据；当时NO-GO，不改写。|
|[Stage9](../stage9/08_FINAL_RESEARCH_DECISION.md)、07容量/机制审计|A原Re、B同容量联合MLP、C位置—运动交叉统计，各200轮，seed42；离线Complete20|C对B窗口ADE改善.001763m，小于.005m门槛；行人等权ADE/FDE退化.006571/.030109m；歧义组仅.00002085m收益。增加配对表达能力并未通过机制/预测门槛。不是“所有位置运动关系无价值”的证明。|

## 本轮实质差异与重复部分

重复部分：相对速度、CPA/DCA、风险排序、动态图、GRU、消息MLP、残差/门控/稀疏关系都不是新的原理。尤其旧TOHI已实现CPA选边，不能称本轮第一次从距离切换到风险。

本轮重新设计的是**归因条件**：新独立命名空间、严格历史完整目标协议、平移等变的速度GRU、无PTN/节点卷积、单跳当前ego消息、无学习关系门控；D/R候选池/每接收者边数/消息特征/参数完全相同，只有排序规则不同。S是单独静态时间规则控制，E有效容量较小。本轮不是复活高阶/关系GRU，也不能把这些工程修订本身当创新。

尚未验证：在该特定受控骨干和Causal8x12任务内，DCA风险排序能否比距离排序提升同目标预测；是否超过强CV及容量控制；跨seed/录像是否稳定；风险较高是否对应真正行为影响。这些均需未来独立授权训练，当前没有性能证据。本轮风险是恒速几何代理，不是行为意图、真实碰撞概率或社会影响因果标签。
