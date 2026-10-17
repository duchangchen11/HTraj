# 最终研究判断

**NO-GO: stop explicit cross-statistic hypothesis**

|训练前门槛|通过|
|---|---|
|GPU_under12hours|True|
|ambiguity_ADE_improves|True|
|at_least_two_sources|True|
|capacity_matched|True|
|cohort_GT_paired|True|
|no_severe_group_degradation|False|
|person_ADE_threshold|False|
|person_FDE_tolerance|False|
|window_ADE_threshold|False|
|window_FDE_tolerance|True|

停止显式交叉统计这项假设，不增加模块挽救，不自动开启下一阶段。

先审BvsA（窗口ADE差+0.000654米），再审CvsB（-0.001763米）。若B/C接近，优先简单联合编码B的后续验证，停止宣称交叉统计独特优势；这仍须B本身对A有值得验证的效果，不能因实现简单强行推荐。

即使全部门槛通过，也只是单seed单fold开发结果，不是跨场景稳定性、SCI新颖性或真实邻居因果收益证明。聚合辨识性、容量数、最终性能必须分开解释。官方Re数据入口与本受控适配不同，不与论文表直接排名。

历史SHA核验87721项通过，未修改Stage1–8或读取test误差。Stage8 NO-GO历史结论保留，本轮新授权实验不倒改既有结论。

## 数据支持的最终解释

C相对B的窗口ADE只降低0.001763米（0.885%），未达到预设0.005米门槛；行人等权ADE增加0.006571米，FDE增加0.030109米（6.585%）。不能仅按窗口ADE的小幅最优宣布C更好。

歧义组3,869个目标窗口中，C的ADE仅比B低0.00002085米，约0.02085毫米。布尔门槛ambiguity_ADE_improves虽为True，但这不是具有说服力的机制收益。非歧义组贡献约99.12%的净窗口ADE改善（分区互斥加权分解，非因果归因）。不支持收益特异性集中于配对歧义。

C在5/7来源的窗口ADE较B低，但只有3/7来源的行人等权ADE更低；5/7来源的行人等权FDE更差。来源内及两种计权方向并不一致。biwi_eth的行人等权FDE由B的0.774807增至C的0.972702米，其26/249身份权重明显高于99/5203窗口权重，是汇总方向冲突的重要数值贡献之一；这不是对模型失败成因的因果证明。

普通联合编码B相对A的窗口ADE/FDE也没有改善；行人等权ADE虽小幅更好，却不足以据此把B宣布为可靠改进。当前没有必要将B/C任何一项作为已验证新方法。若只比较结构复杂度可优先B，但本轮不建议以此启动追加训练。

决定：停止本轮显式交叉统计改造，不追加模型、seed、epoch或测试集评价。这个结论没有否认原聚合非单射的数学事实，也不能证明所有位置—运动配对建模都无效；它表示本次具体实现未获得预设的独立预测收益证据。
