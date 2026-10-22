# 冻结解码器逐位回归与全量推理

解码工程准入PASS。固定3个sample_id（配置中已冻结，按原清单位置0/499/997），原20Goal走新入口，正序及倒序重复共6次；每次完整20×12×2路径与原缓存逐位一致，最大坐标差0。原始行级证据decoder_regression_rows.csv；模型全部参数stateSHA `62453170be18a7ecea3948a921954004e86dc950fe1a91dba1d91997aacfc741`保持不变。

公式保持：原历史×0.25 → 原trajectory_maps与语义 → pred_features；原像素Goal×0.25 → 原trajectory_maps → 原多尺度avg_pool pyramid → concat共享特征 → pred_traj → 原softargmax → 除0.25恢复原像素。单元测试还核对作者内层循环AST。没有调用Goal Predictor、TTST或KMeans，没有改变网络/损失/参数。eval与no_grad固定，语义缓存SHA验证。

全998新路径shape=(20,998,12,2)，均有限，新文件SHA `fa3befae6118c186144a2f7185506d9b85a592ab154c828dbea24d33ba1310e0`，Path张量SHA `11af6e4a4a7f663c975e4dfac5a6eb8198a4b0617acc325ae6765ab382657d50`。全部998个Head0 Goal和Path均逐位不变，ADE1/FDE1逐目标完全相同；不存在将Head0误差变化当收益的情形。新实际解码内容与原缓存相同也保留记录。

GPU回归进程完整wall 8.035311s，998解码进程完整wall 133.163284s；两进程返回码均0。完整进程wall含启动/加载/同步/保存/正常退出/监控线程退出，退出开销没有独立拆分，不用估值冒充实测。

数据与结论范围：同一998个SDD开发验证目标窗口，5个既有录像，8帧历史/12帧未来、0.4秒，单位为SDD原参考像素px，resize=0.25。不是ETH/UCY，不是独立确认，不与米制结果排名。998个录像内行人ID；重叠窗口不能视为独立样本，本报告只给配对描述统计，不给窗口独立性p值。

原始模型SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`；原预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`；作者源码MIT版本 `a345ada1557f53dc2acb4c07f71d340d366980da`。所有计算沿用已冻结协议；无训练、无新采样、无KMeans重跑、无测试集访问。GT只用于全部新预测保存后的离线评分和绘图。

原始证据目录：[results/tables/stage11b2e](../../results/tables/stage11b2e/)。全仓pytest：652 PASS、1 SKIP、0 FAIL、0 ERROR；原始记录full_tests.txt/XML。SKIP不算PASS。 保护后92362项历史文件SHA全部未变（history_after.json）。
