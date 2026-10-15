# Stage 5：公开模型候选清单

核查日期：2026-10-16。只调查EqMotion、HighGraph及另外两个模型SocialVAE、SGCN。有限功能检查仅涉及EqMotion和SocialVAE；SGCN没有实例化或加载权重，HighGraph没有定位到可冻结实现。没有进行论文成绩排名，也未读取ETH测试成绩。

## 发表、代码和版本

|候选|可核实发表|官方代码状态|本轮冻结版本|权重与训练入口|
|---|---|---|---|---|
|EqMotion|CVPR 2023，1410–1420|真实官方源码，MIT|`5aec2e0b61c511fa93a24138dd90da59a089084b`|有官方权重和训练入口，但两个已核查权重严格加载均失败|
|HighGraph|CVPR 2024，15251–15260|本轮官方页面及定向搜索未定位公开源码|未定位，不填伪造commit|没有可验证的独立可运行发布；它本身是插入宿主网络的模块|
|SocialVAE|ECCV 2022，511–528|真实官方源码，MIT|`eb23cbae562bac2c2dcf0d81529e6ab2ac280962`|模型、配置、训练入口、非ETH官方权重均齐全；两个权重严格加载成功|
|SGCN|CVPR 2021，8994–9003|当前main已删除模型和训练文件；官方历史可取回，MIT|完整发布`5153702d8ae32a2b0586b093d4e094aee08b0f9b`|历史提交同时有代码和权重，尚未运行；不能把当前main说成开箱可用|

SGCN最初审查的源码提交为`ec2ac7353ccba7c9691b00c0e978a0631667fa94`；对照上述完整发布，model/train/test/utils/metrics/LICENSE六项Git blob完全相同。当前main为`0d4ac44f44a0688a839f25070be1a4faeed5ade1`。权重只列文件元数据，没有读取保存的成绩。证据：`source_files.json`、`fetch_verification.json`、`sgcn_complete_release.json`。

## 技术与数据可行性

|候选|运动与社会建模|ETH/UCY协议/输出|RTX3080与训练成本|公平消融难点|
|---|---|---|---|---|
|EqMotion|不变运动pattern、等变几何更新、关系类别推断|8/12，20学习头；完整轨迹筛选、目标中心邻居预算因fold不同|实例化约3.03M/5.54M参数；本轮没有实际预测和训练吞吐|HOTEL/ZARA2预处理只有ego；权重版本不匹配；只关类别推断不等于无社会|
|HighGraph|碰撞感知关系、多阶邻接传播|沿用宿主模型的输出/采样规则；不是统一独立模型|无源码，不能确认本地成本|宿主与模块必须同时冻结；本轮不具备复现入口|
|SocialVAE|速度/加速度、自身GRU、几何社会注意力、逐步潜变量与后验训练|8/12；目标完整20帧，邻居可部分出现；K20可结合大量预采样后的FPC|实测2,144,002总参数；batch16一步可运行，约0.186秒；不等于完整收敛|社会初始化与逐帧注意力两条路径均须隔离；采样预算、loader边界和选模规则须固定|
|SGCN|稀疏有向空间图/时间图、双变量高斯|8/12完整20帧离线行人；K20独立minADE/minFDE|CUDA代码及约0.13MB历史权重文件可定位；显存/参数量/吞吐未实测|空间mask与预测卷积沿行人索引运算，存在排列敏感结构风险；不能简单置单位图|

SocialVAE的实际小样本指标见03号报告，不与HTraj旧指标或论文表格横向比较。它已有运动状态、社会选择和随机预测机制，不把这些思想重新命名为HTraj创新。SGCN已研究稀疏关系，EqMotion已研究等变运动与不变交互，HighGraph已研究间接高阶关系。

## 来源和检索限制

- EqMotion：[正式发表记录](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)、[官方源码](https://github.com/MediaBrain-SJTU/EqMotion/tree/5aec2e0b61c511fa93a24138dd90da59a089084b)。
- HighGraph：[正式发表记录](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.html)、[作者页](https://sungjune-kim.github.io/)、[实验室发表页](https://kuaicv.com/publications/)。这些页面未给出本轮可定位的实现；GitHub精确查询两次均0项；旧项目URL返回404。不能由有限搜索断言任何地方永远没有代码。
- SocialVAE：[作者项目/发表记录](https://motion-lab.github.io/SocialVAE/)、[官方源码](https://github.com/xupei0610/SocialVAE/tree/eb23cbae562bac2c2dcf0d81529e6ab2ac280962)、[论文](https://arxiv.org/abs/2203.08207)。
- SGCN：[正式发表记录](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html)、[完整历史发布](https://github.com/shuaishiliu/SGCN/tree/5153702d8ae32a2b0586b093d4e094aee08b0f9b)。

使用nature-academic-search的多源检索流程。学术MCP未暴露；HTTP预检3/3可达，但OpenAlex实际检索429，EqMotion issues API为403限流；CVF两个直接打开请求403，采用其检索到的出版元数据及作者/arXiv来源交叉核查。失败均留档，不伪装为成功检索。完整来源及BibTeX在`results/tables/stage5/primary_sources.json`、`references.bib`。

三套源码的MIT许可不自动赋予ETH/UCY原始数据的再分发权。原始数据、权重、源码缓存保留本机忽略目录，本次Git只保存包装器、来源、hash和轻量结果；未来分发改动源码须保留原许可。
