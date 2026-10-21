# Stage11B-1C 图件说明与完整元数据

所有图使用Python/matplotlib和真实训练或验证记录。作者Y-Net版本`a345ada1557f53dc2acb4c07f71d340d366980da`；训练启动commit`6c9ed2413c0086df7d39b393e414460b671b6fe8`，seed17。协议`HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY`，历史8帧、未来12帧、0.4秒。训练25录像7496目标，验证5录像998目标，无测试评分。best为epoch30，SHA256：

`0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`

误差与轨迹坐标为原始参考图像像素；训练BCE损失无单位。每图PNG为预览、SVG为矢量图件，SVG文字可编辑。图件是开发研究报告图，不宣称满足任何特定期刊的最终版式。

| 文件前缀 | 图要表达的证据 | 原始数据 |
|---|---|---|
| 01_training_total_loss | 真实30轮原始总loss下降 | results/tables/stage11b1c/epochs.csv |
| 02_training_goal_path | Goal和Path原始BCE分项 | 同上 |
| 03_validation_minADE20 | 唯一主选模指标在实际7个节点的变化 | validation_metrics.csv及validation_epoch*.json |
| 04_validation_minFDE20 | 与ADE配对checkpoint的独立FDE oracle | 同上 |
| 05_validation_K1 | 第一完整候选ADE/FDE，不是分布期望 | 同上 |
| 06_recording_comparison | 同一best checkpoint的5录像K20差异 | scene_metrics.csv |
| 07_training_loss_scale_view | 首轮大值与后期变化的尺度差异 | epochs.csv；左为log纵轴，右仅展示第2—30轮线性纵轴 |
| case_low_1..3 | 冻结低minADE20分位数案例 | qualitative_cases.json及qualitative_case_coordinates.csv |
| case_high_1..3 | 冻结高minADE20分位数案例 | 同上 |
| case_large_goal_1..3 | 冻结高minGoalFDE20分位数案例 | 同上 |
| case_*_detail | 同9例的空间放大，不删任何轨迹/goal | 同上及case_detail_viewports.json |

曲线标记仅为实际观测，连线只连接观测点，不代表中间epoch验证已执行。没有平滑、补点、置信区间或独立样本显著性检验。均值口径：主曲线是998目标窗口加权，录像图的各bar是单录像目标均值，数量在坐标标签中标注。只有一个seed；重叠窗口/共享录像不能视为独立行人重复实验。

案例均来自best=epoch30，同一图的全部指标配对同SHA。完整20条轨迹和20个goal保留，淡色重叠可能遮住个别线条，但坐标CSV完整。绿色是GT依赖的ADE oracle高亮，不是部署选择器，minFDE20可来自另一head。GoalFDE和路径FDE分开标注，GT只用于离线评分与显示。

全景RGB来自对应验证录像原始reference.jpg，源SHA记录在qualitative_cases.json；没有亮度、对比度、色相或局部图像调整。放大图仅按统一坐标包围盒裁切显示范围，覆盖全部历史、GT、20路径与20goal，未移动任何坐标，不改变原始像素比例关系；图片放大不增加原始分辨率。本次9例无跨组重复，不能依据图中植被纹理判定语义碰撞。

人工视觉QA查看了loss尺度图、minADE20曲线、录像图，以及低误差/高误差各一例的全景与放大图，文字与SHA完整，全部坐标范围规则由生成脚本验证。其他案例按相同冻结规则生成，未因美观或误差重新选取。

交付格式检查曾因Matplotlib SVG尾随空格失败；仅去除行末空白，逐文件验证全部归一化XML元素、属性、文字、嵌入图像与顺序完全一致。PNG未改，轨迹和统计坐标未改。证据见svg_formatting_audit.json及保留的原始失败日志。
