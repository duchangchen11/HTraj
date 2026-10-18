# 场景语义资源审计

**人工语义标签、palette与分割预训练权重均未取得。** 原SDD的bbox轨迹/类别标注不是道路/建筑物等逐像素语义。六幅reference为真实RGB示例帧，不是语义图。

Y-Net [原文数据与实现部分](https://openaccess.thecvf.com/content/ICCV2021/papers/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.pdf)描述为SDD手工标注5种stuff：pavement、terrain、structure、tree、road；并使用预训练后在具体数据集微调的分割网络。[作者README](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/README.md)把数据/config/权重指向单一Google Drive补充包。链接网页存在不等于已得到其中每种文件。本轮实际只有元数据及约949M包的下载确认页，未提交大型下载确认；语义文件/权重在当前环境仍未取得。

源码有RGB经SMP分割分支，以及seg_mask输入按类别ID one-hot分支；YNetTorch默认semantic_classes=6而论文stuff为5，可能背景/实现配置不同，**未取得配置和palette前无法确定映射**。本轮合成6通道只是工程接口参数，不当作真实类别定义。不能用任意RGB值或灰度自动解释成这些类别。

训练、验证、测试的手工语义图数量、质量、label制作流程与分割训练划分均UNVERIFIED。分割模型是否见过评测场景的人工标签也未知；以后必须披露分割预训练与目标预测各自split，不能把包含测试场景监督的分割器称为完全未知场景泛化。也不能在未知事实下指控原作者泄漏。

可行走性不是简单二分：structure可能对应实体障碍；tree包含树冠遮挡，不能直接视作地面不可走；terrain/road/草地也不应一律排除。场景标签未提供出入口/通行连通性/动态障碍的可靠GT。当前无法计算真实scene compliance或“语义违规率”，更不能称其为安全性。

参考图只表达某时刻背景，可能含动态参与者，不能等同整段录像的动态语义。未尝试用Semantic Drone Dataset替代标签，未运行任何批量伪标签，也未把现有通用分割器输出当人工GT。备用方向只是未来在独立环境审计公开分割模型和人工核对一小部分数据；需要单独授权和来源/质量验证。

当前阻断项：作者语义资源实际取得；ID/颜色/ignore类定义；与每reference的像素尺寸及变换；分割模型来源和标签split；障碍/未知类别规则。任何一项未解决，都不能启动Q2/Q3定量验证。
