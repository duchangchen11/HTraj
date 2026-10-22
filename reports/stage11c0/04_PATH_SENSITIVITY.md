# 完整路径与固定Goal路径敏感性

实验A，窗口加权（所有路径真实运行）：

| condition | ADE1 | FDE1 | minADE20 | minFDE20 | PathPairwiseDispersion | GoalPathEndpointMeanDistance |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 32.093100 | 60.539021 | 16.403136 | 25.634029 | 33.150624 | 2.679707 |
| S1 | 48.058254 | 82.480070 | 28.482479 | 33.758803 | 56.962552 | 3.343418 |
| S2 | 33.103457 | 62.482418 | 17.142452 | 26.747282 | 35.468104 | 2.736717 |
| S3 | 32.447002 | 61.226753 | 17.092719 | 27.126451 | 33.256627 | 2.657791 |


实验B，固定Goal的窗口加权（改变/不变head为每目标平均数量；总数另见下表）：

| condition | ADE1 | FDE1 | minADE20 | minFDE20 | Path_point_mean_displacement | Path_endpoint_mean_displacement | Path_changed_heads | Path_unchanged_heads | Head0_point_mean_displacement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 38.322681 | 60.922061 | 24.302381 | 25.852956 | 15.558313 | 2.738649 | 20.000000 | 0.000000 | 13.243552 |
| S2 | 32.156102 | 60.694886 | 16.493232 | 25.585390 | 3.121930 | 1.471624 | 20.000000 | 0.000000 | 3.021045 |
| S3 | 32.097883 | 60.499180 | 16.454221 | 25.604401 | 1.950132 | 0.806323 | 20.000000 | 0.000000 | 1.775156 |


| condition | global_max_point_move | mean_per_target_max | changed | unchanged | median_point_move | targets_mean_move_gt1px |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 286.134480 | 72.036933 | 19960 | 0 | 14.904062 | 998 |
| S2 | 130.321430 | 8.781352 | 19960 | 0 | 2.769688 | 949 |
| S3 | 176.374736 | 5.561206 | 19960 | 0 | 1.554042 | 740 |


这里global_max_point_move才是全20×998×12点的最大位移；scene_comparison.csv中的Path_max_displacement列按统一均值聚合，表示每目标最大位移的平均值，不把它冒充全局最大值。极端最大值也不能推广为全部目标。

每个B组Goal全部逐位等于S0。三种干预×3个预定ID×正/倒序的18次真实B解码，与保存的对应路径全部逐位一致；未来占位字段0/1e6不改变历史前缀投影，固定Goal的重复性独立通过，详见fixed_repeat_reverse.json/rows.csv。H3=SUPPORTED，H4=SUPPORTED。不把微小bitwise变化直接当有意义变化，给出真实px尺度和1e−4/1/5/25px head计数。固定3个真实ID的被动hook显示Encoder改变级数6—6、Path logits改变head数20—20，原始SHA/范数见internal_probe_comparison.csv与baseline/fixed_internal_probes.json，证明检查了扰动是否到达中间计算。

完整预测和固定Goal是两个不同的条件性实验，不将A的变化全归因于Path，也不将B的变化归因于采样。ADE1、FDE1仅指第0个Softargmax Goal解码Path，不是概率分布期望；minADE20/minFDE20分别取最优head，是GT oracle指标，不是部署候选选择器。

证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`，原S0预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，作者MIT源码 `a345ada1557f53dc2acb4c07f71d340d366980da`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。全仓pytest：671 PASS、1 SKIP、0 FAIL、0 ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。 92464项历史文件SHA全部未改变。
