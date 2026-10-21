# 第一候选与GT oracle差距

数据：998个固定开发验证目标、5录像、seed17、Epoch30，K20/10000 proposals；误差原参考图像像素。来源缓存SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。所有结果均为**开发集选模之后的描述性分析**，不是新算法、正式测试或独立确认。本阶段CPU离线计算，模型前向0次、训练0步、测试池未加载。阈值与规则在 `db88936f33529f91ae50260f0b7e2fda33574d1e` 提交后才执行分组。


SelectionGap_ADE=ADE1-minADE20，SelectionGap_FDE=FDE1-minFDE20；全部998目标均非负。相对gap=gap/max(第一候选误差,1px)，不是独立概率校准指标。下面的分位数使用冻结linear方法，不删极端值。

|metric|valid_n|mean|median|p25|p75|p90|p95|
|---|---|---|---|---|---|---|---|
|SelectionGap_ADE|998|15.689964|9.553162|3.480583|21.203151|33.837228|43.936333|
|SelectionGap_FDE|998|34.904992|22.522168|7.112197|48.919890|77.748109|95.262641|
|SelectionGap_ADE_relative|998|0.484034|0.525466|0.343399|0.663833|0.755660|0.798041|
|SelectionGap_FDE_relative|998|0.576487|0.633136|0.438999|0.773814|0.849122|0.895503|

第一候选已是ADE oracle的目标为69/998（6.91%）；ADE和FDE最优head相同708/998。绝对ADE gap>=25px为183/998。严格B类同时满足默认ADE>=25、oracle ADE<=10、gap>=15、相对gap>=0.5，为66个目标。全部目标分布和每录像分布均保留，不能只分析B类。

goal在10px内有覆盖的509个目标中，仍有13个ADE gap>=25px；其平均gap=6.886727px。覆盖良好时仍可能存在默认候选差距，但较大gap并非全部由这种情况构成。

|scene|targets|SelectionGap_ADE|SelectionGap_FDE|
|---|---|---|---|
|bookstore_2|423|11.206512|24.714632|
|deathCircle_3|443|21.116031|47.426927|
|gates_7|24|12.098486|25.369539|
|hyang_7|45|12.308063|26.971246|
|nexus_4|63|11.422241|24.574424|

## 历史特征的关联限度

|x|finite_pairs|Spearman|
|---|---|---|
|hist_path_length|998|0.533384|
|hist_speed_diag|998|0.506399|
|hist_speed_CV|794|0.052553|
|hist_max_turn_degrees|803|0.139630|
|hist_straightness|916|-0.257702|
|hist_reliable_motion_steps|998|0.489724|

历史幅度和归一化速度与gap存在描述性关联；速度CV和转角的关联较弱。逐录像相关也在descriptive_correlations.csv保存，避免把合并相关当稳定跨录像规律。本轮没有训练任何可部署选择器，不能声称历史输入已能找出oracle head。

源码scripts/stage11b1r/predictor.py把softargmax goal放在第0位，其后19个KMeans中心不构成训练过的置信度排序。没有本研究候选排序头。因此不能把gap称为“排序头失效”，更不能用GT选出的候选冒充在线预测能力。Y-Net已有TTST和goal条件路径，gap本身不构成新算法创新。

机制证据限制：原缓存保存采样goal和完整路径，没有保存可供本轮核查的完整goal概率图。不能仅凭坐标gap就证明softargmax跨模态平均、概率失准或语义特征错误是原因。当前能确定的是候选生成角色不同及真实oracle差距，其内部成因仍UNVERIFIED。后续若检验概率机制，需另行冻结概率图留存/分析协议，不能把未保存的张量重新生成后冒充本轮缓存。
