# 冻结干预协议与因果解释范围

四组定义：S0原六通道；S1包括padding的六通道全零；S2逐通道在有效RGB缩放区域取float32 CPU均值并铺满该区域，padding原值；S3六通道共同在有效区域水平翻转，无插值、无通道交换，padding原值。保存20份实际语义张量、文件与张量SHA，详见intervention_manifest.csv。干预前后shape/dtype不变，history heatmaps8通道在每次真实前向的encoder pre-hook中逐位检查，六通道输入也与指定干预张量逐位一致。

先做训练2目标、四组重复/倒序/未来字段反事实smoke，再提交完整数值协议`cbe69cc220894960f0c051c26b0c98e5feefb026`。S0固定原清单索引0/499/997，正序与倒序两次：logits、Sigmoid、阈值与实际采样权重、proposal/indices、19中心、Softargmax、20Goal和20Path全部逐位复现。正式S0基线只读复用998旧缓存。

首次完整预测在首目标的元数据dict构造时出现重复semantic_SHA TypeError；未读GT，失败原文、6.330436秒GPU wall及第一份真实缓存完整保留。只删除重复关键字并添加trace SHA与实际输入相等断言；干预、网络、采样与评价数值公式不变。新增回归测试后提交`1a9bf5e2b5ddca6d8a0c02dd4577a5ebe44bf54a`，更新源码SHA再继续。原冻结清单保存为protocol_frozen_failed_attempt1.json，不抹去失败历史；失败进程完整时间计入总预算。

实验A分别对S1/S2/S3运行原AuditPredictor：原Encoder/GoalDecoder、temperature1、rel_threshold0.01、CPU TTST10000 replacement=True、作者KMeans19+Softargmax1、原PathDecoder，K20不变。同sample_id使用原SHA派生CPU Generator与KMeans独立种子；干预改变分布后proposal/中心变化属于整个链路响应，不能全归因于分割类别。

实验B保持S0全部20Goal逐位相同，不运行Goal预测、TTST或KMeans，仅用相同历史与各组语义重算Encoder特征及原PathDecoder。2994×20条路径均实际重新解码，非旧路径冒用。A/B分别评价，不作线性可加的因果分解。

JS使用原/新实际阈值后采样权重，离线float64数学归一化，使用自然对数；p=0的KL项定义0，中间分布m=(p+q)/2，不用任意epsilon改变分布。支持像素>0；面积=count×16px²是原坐标网格面积，不是语义可行走面积。熵不是校准不确定性。KMeans head无语义对应，除head0外不按同编号解释模式位移；使用无GT的离线Hungarian/Chamfer/Hausdorff集合几何比较，不改变输出或评分。

尺度解释事先冻结：1e−4px数值级、1px描述性实际尺度，另保留5/25px比例；H1—H4的判断公式见experiment_protocol.json。不是优化收益门槛，也不通过窗口独立性p值声称显著性。GT仅在全部A/B组完整保存、身份/有限性/固定Goal核验后由独立CPU score.py加载。

证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`，原S0预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，作者MIT源码 `a345ada1557f53dc2acb4c07f71d340d366980da`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。全仓pytest：671 PASS、1 SKIP、0 FAIL、0 ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。 92464项历史文件SHA全部未改变。
