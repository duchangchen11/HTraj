# Goal分布与候选集合敏感性

H2：SUPPORTED，仅指当前冻结模型和指定空间扰动。

窗口加权（JS/熵nats、几何px、覆盖0—1）：

| condition | JS_nats | total_variation | logits_mean_abs_difference | logits_RMS_difference | logits_max_abs_difference | Original_entropy_nats | Alternative_entropy_nats | Original_support_pixels | Alternative_support_pixels | Softargmax_displacement | Hungarian_mean_distance | Hausdorff_distance | GoalCover10 | GoalCover25 | GoalCover50 | GoalFDE1 | minGoalFDE20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.723934 | 6.723934 | 2068.413828 | 2068.413828 | 0.000000 | 0.000000 | 0.000000 | 0.510020 | 0.835671 | 0.900802 | 60.614379 | 25.749053 |
| S1 | 0.288441 | 0.617339 | 2.648077 | 3.549769 | 10.227552 | 6.723934 | 7.622299 | 2068.413828 | 4421.200401 | 50.542667 | 61.355360 | 99.768082 | 0.175351 | 0.707415 | 0.864729 | 82.494628 | 33.704687 |
| S2 | 0.053660 | 0.223597 | 1.940891 | 2.587631 | 7.549866 | 6.723934 | 6.826766 | 2068.413828 | 2304.415832 | 16.527136 | 23.141738 | 50.614035 | 0.485972 | 0.816633 | 0.901804 | 62.500443 | 26.958707 |
| S3 | 0.035894 | 0.166819 | 2.286909 | 3.314496 | 9.476682 | 6.723934 | 6.726675 | 2068.413828 | 2103.301603 | 13.155058 | 19.613464 | 43.397530 | 0.496994 | 0.800601 | 0.899800 | 61.324382 | 27.294732 |


五录像等权：

| condition | JS_nats | Hungarian_mean_distance | Softargmax_displacement | GoalCover25 | minGoalFDE20 |
| --- | --- | --- | --- | --- | --- |
| S0 | 0.000000 | 0.000000 | 0.000000 | 0.897417 | 18.488185 |
| S1 | 0.284940 | 57.894396 | 47.879814 | 0.770596 | 26.641322 |
| S2 | 0.042940 | 20.099071 | 13.567336 | 0.877874 | 19.259744 |
| S3 | 0.031222 | 17.947969 | 11.112399 | 0.871837 | 19.324391 |


以上热图差异直接来自缓存中实际Goal logits/实际采样权重。归一化Sigmoid和实际阈值后categorical不是同一个量，JS只比较后一种。S2/S3保留原通道或空间值的部分统计，仍属于冻结模型的分布扰动，未证明当前分割内容正确或错误。

每目标4组goal_distribution_sensitivity.csv、goal_set_comparison.csv保留全部分布和集合统计；每组998个生成记录含proposal、中心、Goal、Path、history与语义SHA。S0精确GoalCover25=0.835671343，不是用83.5671%的舍入数反推。

证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`，原S0预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，作者MIT源码 `a345ada1557f53dc2acb4c07f71d340d366980da`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。全仓pytest：671 PASS、1 SKIP、0 FAIL、0 ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。 92464项历史文件SHA全部未改变。
