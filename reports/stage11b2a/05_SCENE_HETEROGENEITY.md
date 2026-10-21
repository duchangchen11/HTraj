# 五录像异质性与留一敏感性

数据：998个固定开发验证目标、5录像、seed17、Epoch30，K20/10000 proposals；误差原参考图像像素。来源缓存SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。所有结果均为**开发集选模之后的描述性分析**，不是新算法、正式测试或独立确认。本阶段CPU离线计算，模型前向0次、训练0步、测试池未加载。阈值与规则在 `db88936f33529f91ae50260f0b7e2fda33574d1e` 提交后才执行分组。


|scene|targets|diagonal|minADE20|minADE20_diag|hist_speed_mean|future_path_mean|goal_miss25_fraction|YNet_ADE1_minus_OLS8_ADE1|
|---|---|---|---|---|---|---|---|---|
|bookstore_2|423|1785.028011|7.922687|0.004438|22.919106|110.053765|0.056738|2.248884|
|deathCircle_3|443|2363.306159|27.077129|0.011457|37.922796|171.391028|0.300226|-2.975008|
|gates_7|24|2389.116992|6.791864|0.002843|25.147723|103.238748|0.041667|-1.721685|
|hyang_7|45|2422.003303|8.021491|0.003312|22.376375|107.259401|0.066667|2.557423|
|nexus_4|63|2177.782588|7.934808|0.003644|25.173575|109.970058|0.047619|-5.383790|

deathCircle_3的443目标minADE20=27.077129px，图像对角线归一化为0.011457；其他录像约0.002843—0.004438。该录像历史平均速度37.922796px/s、未来平均路程171.391028px，高于其他四录像约22—25px/s和103—110px，但跨录像像素速度不是统一物理速度。

25px目标覆盖失败中，133/164（81.10%）来自deathCircle_3，其录像内失败率30.02%，其他录像4.17%—6.67%。图像大小、运动幅度、交通对象组成、未来路线、标注质量及预训练曝光均可能混杂，不能据此认定哪个原因已经证明。

## 按运动幅度的描述性标准化

|scene|factor|all_three_bins_present|standardized_minADE20_diag|
|---|---|---|---|
|bookstore_2|hist_speed_group|True|0.004374|
|bookstore_2|future_amplitude_group|True|0.004307|
|deathCircle_3|hist_speed_group|True|0.011447|
|deathCircle_3|future_amplitude_group|True|0.010755|
|gates_7|hist_speed_group|True|0.002905|
|gates_7|future_amplitude_group|True|0.003115|
|hyang_7|hist_speed_group|True|0.003766|
|hyang_7|future_amplitude_group|True|0.004147|
|nexus_4|hist_speed_group|True|0.003537|
|nexus_4|future_amplitude_group|True|0.004630|

采用同一全队列三分位比例权重、每录像每bin实际误差，不补缺bin。按历史速度标准化后deathCircle_3约0.011447；按未来路程事后标准化后约0.010755，均仍高于其余录像。这支持“尺度和粗粒度运动幅度不能完全消除观察到的差异”，不是严格匹配、因果控制或语义残差证明。完整scene_motion_strata.csv保留每个bin样本数与运动均值，分层内部仍可能有不平衡。

## 同目标OLS8与留一录像

|excluded_recording|targets|paired_ADE1_delta|paired_FDE1_delta|recording_macro_ADE1_delta|recording_macro_FDE1_delta|
|---|---|---|---|---|---|
|NONE|998|-0.633333|-3.289174|-1.054835|-3.663859|
|bookstore_2|575|-2.753641|-8.347183|-1.880765|-5.476415|
|deathCircle_3|555|1.235789|1.586607|-0.574792|-2.230408|
|gates_7|974|-0.606515|-3.242774|-0.888123|-3.286758|
|hyang_7|953|-0.783998|-3.576127|-1.957900|-5.276785|
|nexus_4|935|-0.313248|-2.828674|0.027404|-2.048926|

差值均Y-Net减OLS8，负值改善。整体窗口加权ADE差-0.633333px；去掉deathCircle_3后变成+1.235789px，FDE差变成+1.586607px。去掉nexus_4后，录像宏平均ADE差亦轻微反转为+0.027404px。完整录像中Y-Net相对OLS8的ADE方向为3/5改善、2/5退化，窗口改善只有408/998目标，不能称跨录像稳定提升。

5次留一分析只作敏感性，完整998总体不变，不删除该录像后挑选新指标。没有窗口级bootstrap、p值或有限5簇的可靠总体显著性结论。参考图像对角线归一化不是米制标定，也没有识别出真正物理难度的倍数关系。

## 录像对整体配对差值的贡献

|scene|targets|mean_ADE1_delta|weighted_ADE1_contribution|mean_FDE1_delta|weighted_FDE1_contribution|
|---|---|---|---|---|---|
|bookstore_2|423|2.248884|0.953184|3.586369|1.520074|
|deathCircle_3|443|-2.975008|-1.320570|-9.397659|-4.171506|
|gates_7|24|-1.721685|-0.041403|-5.172262|-0.124383|
|hyang_7|45|2.557423|0.115315|2.787848|0.125705|
|nexus_4|63|-5.383790|-0.339859|-10.123590|-0.639064|

weighted contribution=该录像全部配对差值之和/998；各录像贡献之和恢复完整总体差值，不是删除录像后的新成绩。负值贡献改善，正值贡献退化。
