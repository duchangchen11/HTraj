# 真实语义图、参考图与轨迹对齐

60张SDD掩码均为单通道L模式，实际像素ID集合**仅1、2、3、4、5**，无0、无调色板。47个有参考RGB的录像全部尺寸一致；另13个仅有mask，没有该包内RGB/短期目标。真实掩码与普通RGB严格区分，未生成伪标签。

短期yaml明确semantic_classes=6，正式分割head为Conv2d(16,6,3)+Softmax(dim=1)，预测模型第一层输入14通道=6+8。因此6输出通道对应0..5，但实际提供mask仅出现1..5。**不能据此给未出现的0命名background/ignore**。包内inD专用README写0:unlabeled、1:pavement、2:road、3:structure、4:terrain、5:tree；这是inD的已知映射，只能作为SDD的待核查线索，不能无声明迁移。SDD类别名称、ignore、训练loss如何处理0仍UNVERIFIED。

Y-Net论文描述人工标注五种stuff，作者包提供离散区域图并与该描述一致；没有逐文件标注者、自动生成/修订历史或分割训练脚本，不能独立证明每个像素的制作过程。SDD_segmentation checkpoint实际用过哪些train/test标签未知；包里有所有17个test录像mask，这不证明权重一定训练过它们，也不能假定未曝光。

六组**作者训练池**真实案例：bookstore_0、coupa_3、deathCircle_0、gates_0、hyang_4、nexus_0。每组包含RGB、语义ID图、8历史点、12个train GT离线叠加，scene/video/frame/person/meta、reference_SHA/semantic_SHA及变换见alignment_cases.json与alignment_points.csv，PNG在alignment_figures。GT只在离线对齐脚本读取，预测函数根本没有future参数。

采用作者原像素坐标，RGB→mask同尺寸identity；推理使用标称0.25倍坐标、cv2.resize(fx=fy=.25)，然后只在下/右侧补齐32倍数。图像整数舍入引起的实际尺寸比例与标称.25细微差异不能叫精确世界定位。无米制比例尺、单应矩阵或estimated_scales.yaml。

视觉检查：可见道路/建筑/草地大区域在RGB与mask上大体对应，但边界为粗粒度手绘多边形；gates_0与nexus_0的轨迹接近或跨越标签边界，不能在名称/边界未核准前当作真实碰撞。当前证明的是作者配套图像的几何同网格和实例可读性，**未证明原视频的精确时空配准，也未证明逐像素语义正确**。语义合规率本轮不计算。
