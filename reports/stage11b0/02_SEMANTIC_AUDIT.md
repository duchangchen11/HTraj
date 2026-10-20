# 语义类别与分割预训练来源

状态仍为**SEMANTIC_CLASS_MAPPING_UNVERIFIED**与**SEGMENTATION_EXPOSURE_UNVERIFIED**，但本轮新增了作者原文依据，不能再笼统说作者没有说明。

成功从[CVF正式论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html)确认并下载[官方补充ZIP](https://www.openaccess.thecvf.com/content/ICCV2021/supplemental/Mangalam_From_Goals_Waypoints_ICCV_2021_supplemental.zip)（约10.6MB）。补充PDF SHA和访问失败记录在semantic_supplement_addendum.json/semantic_audit.json。首个错误大小写链接404、canonical ZIP TLS失败均保留；没有绕过访问限制。

补充材料4.1（PDF第4页）明确五种stuff——pavement/tree/terrain/structure/road，以及用于padding和inD黑区的额外dummy；因此六通道的概念来源有直接原文支持。60张SDD mask只观测到ID1..5，真正SDD的编号→类别和ignore规则仍没有表格可核实。不能把inD README直接移植，不能给0命名background，也不把tree/terrain直接设障碍。

补充材料还明确声明分割预训练仅使用相应轨迹训练场景。应区分：**方法级声明有依据**，**当前公开checkpoint确切成员清单未取得**，不是已经证明它用了test，也不能声称文件级已证明完全没见过test。新HTraj val来自作者原训练池，其语义标签曝光仍可能存在；不能宣称独立未知场景语义泛化。

原文分割架构Unet+ResNet101、ImageNet骨干、Dice、lr1e-4/batch4、1500轮后lr1e-5、前200轮冻结骨干；这里只审查，不重新训练分割器。作者公开Git和ZIP没有提供可绑定checkpoint的完整分割训练manifest/脚本/日志。当前文件SHA沿用Stage11A+，原权重不变。

本轮仅将作者分割权重作**冻结eval特征提取器**，无梯度、BN不更新；手工mask不作为Y-Net输入，不进行语义违规评价。未来若要求严格未知场景语义，须提出单独批准、无评测标签曝光的分割来源/训练协议。
