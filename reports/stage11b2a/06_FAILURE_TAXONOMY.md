# 非互斥失效类型与几何核查

数据：998个固定开发验证目标、5录像、seed17、Epoch30，K20/10000 proposals；误差原参考图像像素。来源缓存SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。所有结果均为**开发集选模之后的描述性分析**，不是新算法、正式测试或独立确认。本阶段CPU离线计算，模型前向0次、训练0步、测试池未加载。阈值与规则在 `db88936f33529f91ae50260f0b7e2fda33574d1e` 提交后才执行分组。


类型定义完整保存在冻结analysis_protocol.json。A25为所有goal距GT终点>25px；A10/A50同时报告。B为默认ADE>=25、oracle ADE<=10、绝对gap>=15且相对gap>=0.5。C为至少一个**同候选**goal误差<=10且路径ADE>=25。D为oracle ADE>=25且历史转向/速度变化/可靠方向证据弱之一，属于关联标签。E为候选越出参考范围或最大步长>max(50px,历史最大步长4倍)。绝对阈值受像素尺度限制，不能当通用物理标准。

|type|targets|fraction|ADE1|minADE20|cohort_minADE20_error_share|
|---|---|---|---|---|---|
|GOAL_COVERAGE_MISS|164|0.164329|102.652124|60.971240|0.610817|
|DEFAULT_ORACLE_GAP|66|0.066132|35.641418|7.631313|0.030767|
|GOAL_PATH_MISMATCH|7|0.007014|55.277570|42.332616|0.018102|
|MOTION_EXTRAPOLATION_DIFFICULTY|66|0.066132|149.670963|97.973383|0.394998|
|GEOMETRIC_ANOMALY|11|0.011022|18.949659|10.645096|0.007153|

各类型不互斥。A25与D重叠59例，A25与B重叠2例，C与D重叠3例，C与E重叠1例。详细5×5交集、每录像数量与精确标签组合分别在failure_overlap.csv、failure_type_summary.csv、failure_exact_combinations.csv；比例不能加总为100%。

## 几何越界

核实原有51个越界预测点，涉及11目标、23条候选，而非新增预测；占239520点的0.0213%。这些目标平均minADE20=10.645096px，贡献总体minADE20误差总和约0.7153%，因此不是当前总体大误差的主要组成。该描述不等于“删除越界候选不会影响oracle”，本轮没有删除后重新选择候选或重排名。

逐点sample_id/head/未来步/x/y/图像范围与对应候选误差均在geometric_out_of_image_points.csv。全部候选的outside_points和goal是否越界也有记录。11目标没有触发冻结的相对历史大步长阈值；这不代表所有预测速度都合理，只说明该定义下没有额外标记。图像范围外可能是模型外推、图像边界或坐标支持问题，单凭当前缓存无法判定原因。

|sample_id|outside_points|outside_heads|
|---|---|---|
|SDD:bookstore_2:track100:frame5328|4|1|
|SDD:bookstore_2:track101:frame5568|3|1|
|SDD:bookstore_2:track102:frame5808|1|1|
|SDD:bookstore_2:track158:frame1224|2|1|
|SDD:bookstore_2:track159:frame1464|3|2|
|SDD:bookstore_2:track169:frame1620|21|7|
|SDD:bookstore_2:track194:frame4608|2|1|
|SDD:bookstore_2:track208:frame10356|6|3|
|SDD:bookstore_2:track230:frame6504|3|2|
|SDD:bookstore_2:track326:frame13200|2|2|
|SDD:hyang_7:track16:frame0|4|2|

## 真实案例

7类各3例，共21个固定选择、21个不同目标，无跨类重复。本次选择不是为了美观；每类按预先冻结score排序取25/50/75%秩。全景和局部共42幅案例图，保留20路径/20goal、第一候选、ADE oracle、对应最近goal路径、CV和OLS8；各图坐标都来自原缓存或核验过的解析基线。

|case|sample_id|ADE1|minADE20|minGoalFDE20|failure_types|
|---|---|---|---|---|---|
|goal_miss25_1|SDD:deathCircle_3:track535:frame10680|55.780964|19.986845|38.201421|GOAL_COVERAGE_MISS|
|goal_miss25_2|SDD:deathCircle_3:track479:frame4500|184.608472|160.674516|55.665398|GOAL_COVERAGE_MISS;MOTION_EXTRAPOLATION_DIFFICULTY|
|goal_miss25_3|SDD:deathCircle_3:track744:frame2796|115.771844|52.013039|107.852961|GOAL_COVERAGE_MISS;MOTION_EXTRAPOLATION_DIFFICULTY|
|default_oracle_gap_1|SDD:bookstore_2:track149:frame492|29.391855|8.678367|9.636897|DEFAULT_ORACLE_GAP|
|default_oracle_gap_2|SDD:bookstore_2:track410:frame8580|31.011986|6.254237|7.652471|DEFAULT_ORACLE_GAP|
|default_oracle_gap_3|SDD:bookstore_2:track111:frame12060|36.247704|6.085407|9.217523|DEFAULT_ORACLE_GAP|
|goal_path_mismatch_1|SDD:deathCircle_3:track316:frame8760|40.377247|21.001474|8.016977|GOAL_PATH_MISMATCH|
|goal_path_mismatch_2|SDD:bookstore_2:track208:frame10356|42.813069|33.012156|9.842018|GOAL_PATH_MISMATCH;GEOMETRIC_ANOMALY|
|goal_path_mismatch_3|SDD:deathCircle_3:track480:frame4740|60.143615|37.316989|6.589083|GOAL_PATH_MISMATCH;MOTION_EXTRAPOLATION_DIFFICULTY|
|history_straight_future_turn_1|SDD:deathCircle_3:track51:frame5412|55.777512|27.110953|40.372836|GOAL_COVERAGE_MISS;MOTION_EXTRAPOLATION_DIFFICULTY|
|history_straight_future_turn_2|SDD:bookstore_2:track296:frame3804|38.009077|7.969881|13.027520|DEFAULT_ORACLE_GAP|
|history_straight_future_turn_3|SDD:deathCircle_3:track405:frame3408|40.277963|17.081182|21.333175|未定义|
|geometry_anomaly_1|SDD:bookstore_2:track194:frame4608|4.784443|3.457253|5.401553|GEOMETRIC_ANOMALY|
|geometry_anomaly_2|SDD:bookstore_2:track159:frame1464|4.525803|4.525803|6.298746|GEOMETRIC_ANOMALY|
|geometry_anomaly_3|SDD:bookstore_2:track102:frame5808|14.955841|11.881088|9.627586|GEOMETRIC_ANOMALY|
|YNet_better_OLS8_1|SDD:deathCircle_3:track56:frame6612|22.247899|9.665765|8.633585|未定义|
|YNet_better_OLS8_2|SDD:deathCircle_3:track488:frame684|26.178196|15.379320|12.807202|未定义|
|YNet_better_OLS8_3|SDD:deathCircle_3:track376:frame11868|29.560565|13.597539|17.792551|未定义|
|YNet_worse_OLS8_1|SDD:bookstore_2:track368:frame8004|3.315362|3.306236|5.837807|未定义|
|YNet_worse_OLS8_2|SDD:bookstore_2:track376:frame948|64.885052|21.091921|24.169663|未定义|
|YNet_worse_OLS8_3|SDD:hyang_7:track43:frame240|14.846107|3.887842|8.954882|未定义|

三个优先核查类型：A25目标覆盖失败；B默认与oracle差距；C对应goal准确但中段偏离。C类7个目标中的bookstore_2 track208可见未来回转，准确终点不保证正确中途路线；RGB只能显示背景，不能给出道路、草地、树冠的可通行真值。未来直行/转向、oracle高亮与最近goal路径全部是离线GT诊断，不是部署推理。

## EXPLORATORY：参考边界与解码画布支持

额外静态/CPU几何核查：作者image_utils.py的resize使用cv2.resize(fx=fy=.25)，pad补到32的倍数；路径softargmax解码在padding后的完整画布内。51个原图越界点中，51点仍处于原网络合法padding画布范围，51点超出缩放后原图像素中心范围。逐点证据见EXPLORATORY_padding_support.csv。此现象与padding提供的额外坐标支持相容，不能据此证明padding是成因，也不能称NaN/数值爆炸或语义碰撞。本轮没有裁剪、重算预测或改动指标。该解释性扩展明确标为EXPLORATORY，未改变冻结分类阈值。
