# 语义研究命题与下一步决策

| hypothesis | proposition | status |
| --- | --- | --- |
| H1 | 语义输入敏感性 | SUPPORTED |
| H2 | 语义空间结构影响Goal | SUPPORTED |
| H3 | 固定Goal时Path响应 | SUPPORTED |
| H4 | 真实误差发生变化 | SUPPORTED |
| H5 | 真实语义独立预测价值 | UNVERIFIED |
| H6 | 分割错误导致预测失败 | UNVERIFIED |


| condition | ADE_better_recordings | ADE_worse_recordings | macro_ADE_delta | macro_FDE_delta |
| --- | --- | --- | --- | --- |
| S1 | 0 | 5 | 12.158664 | 8.549101 |
| S2 | 0 | 5 | 0.608704 | 0.804255 |
| S3 | 0 | 5 | 0.597225 | 0.902923 |


本阶段是输入干预敏感性研究，不是语义增强算法实验。冻结网络遇到全零、均值或错位概率图，可能有输入分布偏移；即使S0更好，也不能直接证明同预算训练时语义有独立增益。S0更差也不能证明移除语义训练更好。H5需要同数据、同预算、从头训练的匹配无语义对照；H6缺少语义GT与可靠类别映射，没有直接质量测量，因此UNVERIFIED。

研究建议：MATCHED_NO_SEMANTIC_TRAINING_CONTROL_WORTH_PROPOSING。这只是可提案的下一步，不授权自动执行。唯一建议的确认性实验是匹配数据/初始化/预算/评价对象的从头无语义训练对照；单独统计有效容量与输入归一化差异。需预注册多seed、录像口径与失败门槛，并明确仍是已曝光开发数据。不得从当前结果直接开发可靠性门控、声称新颖性，或改写论文已有结论。

不依据测试集选择方向；不启动Stage11C-1。若各录像方向不一致，保留条件性解释；不包装统一语义可靠性机制。硕士小论文方向是否调整，待用户审查本阶段证据和训练成本后决定。


GPU完整进程wall合计1517.301774s，包括预检、S0回归、失败首轮、重跑完整A、全部B、模型加载/缓存/退出。硬上限7200s，安全阈值6600s。失败首轮6.330436s未剔除。峰值及阶段时间见各worker日志和GPU_card_*.csv；预算估计来自训练smoke，实际值没有用历史耗时代替。训练时间0s。CPU评分55.743803s，SHA、全测试及作图另计。

证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`，原S0预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，作者MIT源码 `a345ada1557f53dc2acb4c07f71d340d366980da`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。全仓pytest：671 PASS、1 SKIP、0 FAIL、0 ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。 92464项历史文件SHA全部未改变。
