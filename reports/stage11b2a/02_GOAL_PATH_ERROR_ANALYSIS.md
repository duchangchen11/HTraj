# Goal/Path逐候选对应关系

数据：998个固定开发验证目标、5录像、seed17、Epoch30，K20/10000 proposals；误差原参考图像像素。来源缓存SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。所有结果均为**开发集选模之后的描述性分析**，不是新算法、正式测试或独立确认。本阶段CPU离线计算，模型前向0次、训练0步、测试池未加载。阈值与规则在 `db88936f33529f91ae50260f0b7e2fda33574d1e` 提交后才执行分组。


记第k条路径为p_ik、goal为g_ik，真实未来为y_i。ADE_ik=mean_t||p_ik,t-y_i,t||；FDE_ik=||p_ik,12-y_i,12||；GoalFDE_ik=||g_ik-y_i,12||；Gap_ik=||g_ik-p_ik,12||。分别保留argmin_Goal、argmin_ADE、argmin_FDE，精确同值保留原顺序较早head。**不将不同head的minGoalFDE与minADE相减当成误差分解。**

窗口均值：GoalFDE1=60.614379，minGoalFDE20=25.749053；ADE1=32.093100，minADE20=16.403136；路径FDE1=60.539021，minFDE20=25.634029。

最近goal对应head与ADE最优head相同为660/998，与FDE最优相同为864/998。最近goal对应完整路径ADE均值17.471469，FDE均值25.931501。这两个值可与oracle说明对应路径差距，但最近goal也依赖GT，不能在线使用。所有19960个候选配对保存在goal_path_decomposition.csv；每个候选还保存前/中/后4步误差和goal-path间距。

## 覆盖条件与错误

|family|group|targets|minADE20|SelectionGap_ADE|
|---|---|---|---|---|
|goal_coverage_10px|covered|509|5.675975|6.886727|
|goal_coverage_10px|missed|489|27.569035|24.853252|
|goal_coverage_25px|covered|834|7.639144|10.579040|
|goal_coverage_25px|missed|164|60.971240|41.680884|
|goal_coverage_50px|covered|899|8.590998|12.139420|
|goal_coverage_50px|missed|99|87.343658|47.931777|

25px邻域下，164/998个目标全部goal均未覆盖，平均minADE20=60.971240，贡献全体minADE20误差总和的61.08%。834个覆盖目标平均minADE20=7.639144。这支持“较大的剩余误差与goal覆盖不足共现”，不是goal模块因果贡献百分比分解。未覆盖目标只占16.43%，不能称为多数目标普遍失败。

## 目标正确而中段错误

严格使用同一候选：GoalFDE<=10px且ADE>=25px，共7个目标，最近goal候选也均满足此条件。其中7个具有FDE<=10px且前11步平均误差>=25px，终点FDE>=25px的子类型为0。说明这7例主要是路径中段偏离，而非goal与解码末点的普遍不一致。该类型总样本少，不能代替总体结论，不能推断由语义错误造成。

|sample_id|closest_goal_head|closest_goal_ADE|closest_goal_FDE|minGoalFDE20|minADE20|
|---|---|---|---|---|---|
|SDD:bookstore_2:track208:frame10356|17|34.108663|8.563501|9.842018|33.012156|
|SDD:deathCircle_3:track480:frame4740|7|47.136734|4.941184|6.589083|37.316989|
|SDD:deathCircle_3:track499:frame5004|7|33.343045|9.796161|7.804826|33.343045|
|SDD:deathCircle_3:track229:frame5124|1|77.530301|5.634293|6.490650|77.530301|
|SDD:deathCircle_3:track316:frame8760|14|30.720358|7.479777|8.016977|21.001474|
|SDD:deathCircle_3:track688:frame9576|3|25.940696|0.735047|0.616915|22.403700|
|SDD:deathCircle_3:track543:frame9912|1|78.715881|2.376712|3.506350|71.720644|

图2、3与病例对应路径只作离线解释；图中ADE oracle、FDE oracle和最近goal head有明确不同含义。
