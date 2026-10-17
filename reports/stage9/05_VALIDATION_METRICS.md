# 最终共同checkpoint验证指标

这是HOTEL折其他7个来源的官方validation区域，不是HOTEL留出test结果，也不是官方Re成绩复现。目标5,203个，来自688个重叠窗口。所有A/B/C的sequence/frame/person/history_SHA/GT_SHA逐项一致；K20随机噪声种子与batch顺序完全配对。

|模型|窗口minADE20|窗口minFDE20|行人等权minADE20|行人等权minFDE20|
|---|---|---|---|---|
|A|0.198469|0.335136|0.273614|0.462428|
|B|0.199123|0.338067|0.270441|0.457260|
|C|0.197360|0.339161|0.277012|0.487370|

单位米。窗口加权为每个目标窗口等权；行人等权先在同(source,person_id)的重叠窗口内平均，再对来源内身份等权，防止重叠窗口被误当独立行人。逐来源宏平均仅作补充，不能代替两个主要口径。

minADE20与minFDE20分别取完整12步候选ADE和终点FDE的最小值，可能不同head。没有逐时间步换head、GT候选重排或FPC。预测归档包含固定原始K20轨迹与GT，独立NumPy float64重新计算并核对逐目标GT_SHA；数值差异见independent_metric_verification.csv。此oracle评价不代表在线挑选head能力。

窗口ADE：B−A=+0.000654米，C−B=-0.001763米。负值代表改善。不得由A/B额外容量差异直接断言社会关联贡献。
