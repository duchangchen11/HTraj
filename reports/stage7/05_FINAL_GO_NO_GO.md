# 最终Go/No-Go及最低正式实验

**Go仅限AgentFormer-CVAE维护版在明确离线任务上的最小复现准备；No-Go用于论文新算法开发和严格历史因果全链路复现的完成声明。** 本阶段不启动训练。不是强行宣布一个已获真实提升的正式论文主干；若“合格”要求Stage5.1严格历史上下文整链路，本轮没有完全合格对象，不能直接开训。

唯一保留对象：`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`，原ETH/UCY pre配置，所有原模型层和MSE/KL/sample目标不变。选择理由是来源/训练入口完整、维护版已隔离旧scene_orig未来值、单人与多人/先验/排列/反向/恢复及N57资源检查通过。PECNet不推荐为并行备选，旧TOHI/G0/G1/SocialVAE/采样路线继续关闭。

## 已冻结但未执行的最小方案

精确设计与SHA在configs/stage7/minimal_reproduction_plan.json，协议名HTraj-AgentFormer-A_offline_complete20-v1。依据Stage5.1冻结录像与时间支持，从原始五个非ETH文件生成新manifest，保留N1；同一监督目标集重新分组为同步完整20帧上下文。未来存在性用于离线资格，不使用GT坐标计算运动特征/连接，不冒称Causal8x12。单人/完整目标数如下：

| 留出录像 | split | 窗口 | 目标 | 单人窗口 | 最大人数 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | train | 2121 | 27145 | 151 | 57 |
| biwi_hotel | validation | 94 | 318 | 25 | 8 |
| crowds_zara02 | train | 1684 | 23545 | 196 | 57 |
| crowds_zara02 | validation | 172 | 1092 | 3 | 13 |


训练集排除整个验证录像，仅原来源train区域，旧validation不混训练；ETH封存。HOTEL、ZARA2两个曝光开发划分，不叫独立确认。需从原始数据重建packet，不沿用自定义模型缓存。数据source SHA/新manifest SHA已经保存；Stage7未计算新真实预测指标。

四run：原始CVAE与逐人独立同结构对照×两录像×seed17。均随机初始化，相同窗口/监督目标访问、每窗口一次优化（独立人loss先聚合），最多30epoch，前5轮包含在内；Adam1e-4、StepLR10/gamma0.5、原loss三项和K20，均不调参、不使用旧权重。逐人独立处理必须同时消除global scene中心及注意路径，不能只设对角mask后称无社会。

有效容量和单人KL clipping聚合与Full可能不同，scene中心也不同；前置审计必须说明，不能因名义参数相同宣布完全控制。这个对照足以检验个体预测能力，**不足以单独证明真实社会信息贡献**。若后续要求独立归因而不能控制这些差异，停止方法开发，不自动增加容量/Placebo实验。

同窗口/目标ID/GT重算CV-Last2、OLS4、OLS8，固定Last2为主，不事后逐录像挑CV。主指标K1 learned-prior-mean（不是recon、不是轨迹数学期望），选最低窗口ADE/早epoch，同checkpoint FDE；同ID噪声首样本、独立minADE20/minFDE20另报，所有联合候选生成完再对GT计算指标。窗口/录像内行人等权/录像宏平均全部报告，不将重叠窗口当独立样本。

开始前仍须通过真实数据入口smoke、np.int机械兼容、目标/GT/预算/损失聚合/有效容量审计；不默默运行原test.py的strict=False。串行RTX3080，N57保留显存已约9.66GB，遇OOM立即停止，不能删拥挤难例换可运行。当前没有真实吞吐，不能抄SocialVAE训练耗时估算AgentFormer整轮。

停止规则冻结：任何非有限/泄漏/目标错配/OOM即停该run不盲重试；30轮到期停，best=30标截断不延长。继续运动预测研究需两录像两权重相对Last2 ADE增益≥max(.005m,1%)、配对FDE退化≤max(.005m,1%)；独立社会研究还必须相对逐人对照达到同门槛且解决容量/中心混杂，否则不增加社会模块。K1不过但K20改善仅记录多模态条件收益，不自动转第五章。全部条件通过也只建议后续独立验证，不能宣称创新。

完整两阶段AgentFormer/DLow、官方权重真实性与官方原论文数值未复现；严格历史上下文适配未完成。以上是冻结的有限预算方案，不是当前训练授权或已完成结果。最终方向仍由下一条用户指令确定。
