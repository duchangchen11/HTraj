# 图件证据、统计单位与显示约定

核心结论：冻结Epoch30的剩余误差集中于部分goal覆盖失败目标，同时存在少量“目标准确而中段错误”，不能把它们解释为已证明的语义根因。图1—9为quantitative grid，案例为image plate + quant，图10为固定案例组合；使用Python/matplotlib，PNG预览与可编辑文字SVG。所有数据均来自真实缓存/已冻结曲线，没有新模型训练或推理。

模型版本：Y-Net官方`a345ada1557f53dc2acb4c07f71d340d366980da`，seed17，Epoch30。checkpoint SHA：

`0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`

原预测缓存SHA：`2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`。

协议：HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY，8历史/12未来，0.4秒，5录像998目标。此队列已用于checkpoint选择，所有结果仅为开发集分析。原误差/坐标单位像素，对角线归一化不是米制。K1为softargmax goal对应的第一完整轨迹；K20 ADE/FDE分别取独立oracle，可能来自不同head。

|图件前缀|证据与统计单位|源表（results/tables/stage11b2a/）|
|---|---|---|
|01_K1_K20_distributions|998目标真实经验分布；symlog保留0与全部极端值|motion_features.csv|
|02_goal_path_relationship|19960同候选Goal/Path配对；非19960独立样本。右为998目标最近goal路径与ADE oracle|goal_path_decomposition.csv、goal_path_summary.csv|
|03_goal_coverage_error|10/25/50px覆盖比例与条件均值，组样本数标注；关联不是因果|motion_error_groups.csv|
|04_history_motion_error|原始历史幅度/图像对角线与误差；来源训练历史速度三分位|motion_features.csv、motion_error_groups.csv|
|05_turning_error|全部历史/未来转向组均值与n；未定义方向保留；未来组仅事后|motion_error_groups.csv|
|06_recording_motion_comparison|5录像原误差、对角线归一化与未来运动幅度，各录像n标注|scene_normalized_analysis.csv|
|07_failure_types_and_overlap|非互斥A25/B/C/D/E数量、全部成对交集；不加总成100%|failure_type_summary.csv、failure_overlap.csv|
|08_paired_OLS8_error|998同目标Y-Net减OLS8的ADE/FDE差；负值改善，保留全部样本|motion_features.csv、leave_one_recording_out.csv|
|09_frozen_validation_trends|已有七个真实节点的误差、覆盖、多样性；连线不代表中间轮已验证|frozen_validation_trends.csv|
|10_real_failure_case_composite|每个固定类型选例中的中间一例，共7例；所有轨迹/goal和基线均显示|failure_case_manifest.csv、failure_case_coordinates.csv|
|case_*_full / case_*_detail|7类各3例，共21目标，42视图；没有因美观重新选择|同上、failure_case_selection.json|

统计：除明确标注录像均值外，均值/经验分布以目标窗口为单位；无置信区间、显著性检验或窗口bootstrap，不能把共享录像/重叠窗口当独立场景。相关系数为有限配对的平均秩Spearman，既报告合并也报告逐录像；不赋予因果解释。

案例选择：每类按冻结score和sample_id排序取25/50/75%秩，本次21例无重复。完整病例metadata包括sample_id、scene、reference/history/GT/checkpoint SHA、运动指标、失效类型和误差。原RGB来源reference.jpg的SHA在病例manifest中；未修改图像颜色、亮度或对比度。全景保留原图范围并扩展到全部预测和CV/OLS8坐标；局部视域包含历史、GT、全部20路径和20goal、CV及OLS8，再加固定余量，**不会裁掉失败候选**。

所有轨迹坐标来自冻结缓存，CV/OLS8为历史解析结果且逐例误差与旧基线核对；未来GT仅用于离线评分、类别和oracle解释。绿色ADE oracle、紫色最近goal路径均使用GT判断，不能部署时直接得到；默认第一候选单独标色。GoalFDE不同于完整路径FDE。图10为初次固定组合渲染，完整单案例图额外突出最近goal对应路径；二者都保留所有候选。

质检：人工检查图3/5/7/9/10及目标正确路径偏离案例的初稿/修订稿。初稿案例caption与x轴标签重叠，已增加底部留白；不是重新选样或重算统计。初次逐位CSV核查因pandas默认解析精度失败，改为float_precision=round_trip后，21例的全部缓存路径与CSV逐位一致，坐标CSV SHA未改。失败日志和当时源码快照保留，修订映射见case_layout_QA.json；原初稿截图也作为QA证据保存。

SVG只清理行末空白，归一化XML的元素、属性、文本、嵌入图像和顺序相同；不改任何统计或路径坐标。图中不存在经确认的语义类别标签或碰撞结论。单个极端案例不能外推总体机制。

本轮统一标为EXPLORATORY_DEVELOPMENT_ANALYSIS，冻结分析规则不改变该队列曾被用于选模的事实。
