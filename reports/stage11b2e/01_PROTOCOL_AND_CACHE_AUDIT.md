# 冻结协议与缓存身份审计

结论：缓存准入PASS；实验只检验一次普通代表点替换。基于Stage2D `c3e5d09354130bc51d22bfa325342ae6a0831de6` 建立当前分支。冻结提交`d221b22279731d94655c1cba5134efc9876f2b0b`在998目标选择前成功；CPU选择执行SHA与GPU/评分执行SHA分别记录在selection_audit.json与generation_execution.json。工程准入提交`c90bd50978c4234f6daf2d7f5db9d00cbbc2a02c`在GPU前成功。数值脚本、评价公式和四项门槛的文件SHA见protocol_frozen.json。

cache_audit.json确认998个sample_id唯一、原10000个proposal的下标/坐标/顺序SHA一致，原19中心、Softargmax、原20Goal与原20Path逐位一致，全部18962个分区非空，人口和每目标10000。没有用重新生成的随机样本替代缓存。保护清单configs/stage11b2e/protected_files.csv.gz覆盖92362项历史文件。

CPU选择接口只接收白名单sample_id、proposals、centers、Softargmax；实际GT尚未装载。原记录中未来GT和oracle/A25标签被替换的3个反事实样本，白名单输入和结果均逐位一致；全998倒序重算逐位一致。GPU配置为GT-free白名单历史、冻结语义、模型、已选Goal和3个旧预测回归参考；worker不载入GT/失败标签/已有误差。先完整保存(20,998,12,2)新Path，再由独立CPU score.py载入GT。score_integrity.json与逐目标history_SHA/GT_SHA核验通过。

当前使用既有纯历史缓存及五录像冻结语义，不新执行分割网络；原分割网络仍随模型严格加载并处于eval。大型tensor、模型、原图片只在忽略缓存，不提交。2D中的最终Voronoi不等同原KMeans最后迭代标签；本阶段不改变此定义。

数据与结论范围：同一998个SDD开发验证目标窗口，5个既有录像，8帧历史/12帧未来、0.4秒，单位为SDD原参考像素px，resize=0.25。不是ETH/UCY，不是独立确认，不与米制结果排名。998个录像内行人ID；重叠窗口不能视为独立样本，本报告只给配对描述统计，不给窗口独立性p值。

原始模型SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`；原预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`；作者源码MIT版本 `a345ada1557f53dc2acb4c07f71d340d366980da`。所有计算沿用已冻结协议；无训练、无新采样、无KMeans重跑、无测试集访问。GT只用于全部新预测保存后的离线评分和绘图。

原始证据目录：[results/tables/stage11b2e](../../results/tables/stage11b2e/)。全仓pytest：652 PASS、1 SKIP、0 FAIL、0 ERROR；原始记录full_tests.txt/XML。SKIP不算PASS。 保护后92362项历史文件SHA全部未变（history_after.json）。
