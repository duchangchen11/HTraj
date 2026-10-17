# 2023—2026方法与协议核查

检索日期2026-10-17。使用nature-academic-search多源流程及nature-downloader开放全文核验；这是针对本轮问题的可追溯检索，不是穷尽所有文献的系统综述。学术MCP未挂载；OpenAlex备用脚本遇HTTP429，CrossRef与出版社/作者/arXiv/CVF互补。GitHub API限流，改用公开git ls-remote冻结版本并读取固定commit文件。所有失败均留档，不把检索失败等同论文/代码不存在。

AgentFormer、TUTR、MoFlow、Joint Metrics Matter、SocialCircle及PerReg+已取得合法PDF并读方法和实验，SHA在literature_sources.json。SocialTrans与Inter-GraphFormer身份已定位，但全文审阅未完成：出版社入口403；OpenReview请求浏览器验证，不绕过；Inter-GraphFormer出版社声明的文本接口429。未取得个人图书馆会话或作者全文，相关loss/协议保留待核实。**因此指定五篇中有两篇未达到完整方法精读，不能宣布新颖性检索全部通过。**

分类严格区分：原方法直接覆盖、相近思路、不同任务、尚未核实。相近思路并不证明技术完全相同，缺少公式也不证明不同。

### AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting

2021，ICCV。[论文](https://arxiv.org/html/2103.14023v3)；作者代码：https://github.com/Khrylx/AgentFormer；版本`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`。阅读范围：全文§3/4与固定维护源码。

- 方法：agent-aware时空attention；CVAE联合潜在意图；DLow多候选。
- 输入/输出：多行人历史；ETH/UCY无地图 → K条多行人未来轨迹。
- 训练与评价：CVAE重建/KL/sample；DLow重建/KL/diversity；ETH/UCY、nuScenes；ETH/UCY 8/12帧、K20 marginal minADE/FDE；nuScenes另有K1/5/10。
- 与本项目：本项目冻结基础。重合判定：DLow条件化与多样性思想已存在，不能重新命名。
- 证据定位：paper§3.1/3.2/4；model/agentformer.py、dlow.py。

### Trajectory Unified Transformer for Pedestrian Trajectory Prediction (TUTR)

2023，ICCV。[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf)；作者代码：https://github.com/lssiair/TUTR；版本`655a756af8e961598a1a22a07cb48eb8534c722a`。阅读范围：全文§3.2/4.1与model.py/train.py。

- 方法：训练轨迹归一化聚类mode；mode/social两级Transformer；轨迹与概率双输出。
- 输入/输出：目标和邻居历史、来源训练集mode中心 → 轨迹与概率；推理按概率TopK。
- 训练与评价：最近训练mode监督的Huber回归+软标签交叉熵；ETH/UCY、SDD；8/12帧；best-of20 ADE/FDE与brier指标。
- 与本项目：解决未排序多候选/后处理开销，相近问题而非AgentFormer简单增层。重合判定：普通概率评分头与归一化mode想法已覆盖。
- 证据定位：paper§3.2 Eq1–6/§4.1；model.py:37–75；train.py:248–260有test选best需审慎。

### SocialTrans: Transformer based social intentions interaction for pedestrian trajectory prediction

2025，Physica A 663:130435。[论文](https://doi.org/10.1016/j.physa.2025.130435)；作者代码：https://github.com/XiaodZhao/SocialTrans；版本`未取得`。阅读范围：出版社方法/实验概要；全文未取得，不能称精读完成。

- 方法：概要：SIE/NPIE并行意图状态、终点优化及聚类。
- 输入/输出：概要：目标与邻居观测运动；准确特征/资格待核实 → 多未来轨迹；精确候选数待核实。
- 训练与评价：完整公式/权重待核实；ETH/UCY、SDD；ADE/FDE；K、筛选、选模协议待全文核实。
- 与本项目：目标/邻居意图协同、终点修正为相近思路。重合判定：尚未核实，不能作为不存在某机制的证据。
- 证据定位：publisher Proposed approach/Experiments；OpenReview PDF遇浏览器验证，不绕过。

### MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation

2025，CVPR。[论文](https://arxiv.org/html/2503.09950v2)；作者代码：https://github.com/DSL-Lab/MoFlow；版本`f1b89b0e80ce95646214923e271e3c280d305cfa`。阅读范围：全文§4/5与官方README/模型源码静态检查。

- 方法：相关K-shot flow matching回归+分类；IMLE/Chamfer一步蒸馏。
- 输入/输出：历史多行人位置/相对运动上下文、噪声 → K个相关未来；一步student。
- 训练与评价：最佳候选回归+CE；student匹配teacher样本的Chamfer；ETH/UCY、SDD、NBA；ETH/UCY 8/12、min20ADE/FDE；NBA10/20帧；SDD像素。
- 与本项目：多候选多样性/推理效率相近；生成机制不同。重合判定：候选覆盖、相关头和概率辅助已覆盖；本阶段不采用flow/扩散。
- 证据定位：paper§4 Eq4.3/4.4,§5.1/5.2；README明确ETH/UCY有LED与original不同版本。

### Inter-GraphFormer: A spatial–temporal graph transformer-based method for pedestrian trajectory prediction

2026，Physica A 694:131566 (July2026)。[论文](https://doi.org/10.1016/j.physa.2026.131566)；作者代码：未定位可验证作者源码；版本`未取得`。阅读范围：CrossRef身份及出版社Introduction/方法概要；全文未取得。

- 方法：概要：稀疏有向图、并行时空Transformer、dual-GLU、CVAE。
- 输入/输出：历史运动与社会上下文；精确状态/资格待核实 → 多个未来轨迹。
- 训练与评价：CVAE提及；精确loss/权重待核实；ETH/UCY、SDD；完整8/12、K及选模细节未核实，不推定。
- 与本项目：显式运动/非对称交互是相近思路。重合判定：相近思路已公开；与具体草案是否相同尚未核实。
- 证据定位：publisher Introduction贡献段；CrossRef DOI元数据；text-mining全文接口429。

### Joint Metrics Matter: A Better Standard for Trajectory Forecasting

2023，ICCV。[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf)；作者代码：https://github.com/ericaweng/joint-metrics-matter；版本`894182b5ea7dac3c93bbe0bcc18c156c932398b7`。阅读范围：全文§3/4/5与官方README。

- 方法：JADE/JFDE及collision系统评价；Joint AgentFormer训练项。
- 输入/输出：历史多行人；联合候选 → 原预测器联合轨迹。
- 训练与评价：原AgentFormer基础上增加joint sample reconstruction；ETH/UCY、TrajNet SDD；marginal与joint K20、collision；8/12、stride1、包含单人。
- 与本项目：直接改进AgentFormer联合度量。重合判定：仅增加联合oracle指标/损失属于高度重合工作。
- 证据定位：paper§3/§4.1/§5；README标准保存格式与数据差异说明。

### SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction

2024，CVPR。[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf)；作者代码：https://github.com/cocoon2wong/SocialCircle；版本`383ff0671f011806a8da9422d4a3839206847b82`。阅读范围：全文§3/4；作者仓库身份/版本。

- 方法：角分区聚合速度(历史净位移)、距离、方向，作为虚拟时间序列接入backbone。
- 输入/输出：历史目标/邻居运动 → 沿用所接backbone的K未来。
- 训练与评价：沿用backbone原loss，不新增统一loss；ETH/UCY、SDD；8/12帧、minADE20/FDE20；ETH/UCY米、SDD像素。
- 与本项目：显式相对运动表示近邻先例。重合判定：直接添加距离/速度/角度社会特征不是新机制。
- 证据定位：paper§3 Eq1–11,§4；官方仓库TensorFlow与beta PyTorch版本须分开。

### Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting

2025，CVPR。[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Messaoud_Towards_Generalizable_Trajectory_Prediction_using_Dual-Level_Representation_Learning_and_Adaptive_CVPR_2025_paper.pdf)；作者代码：本次未定位作者独立实现，不声称已复现；版本`未取得`。阅读范围：全文§3/4。

- 方法：PerReg+自蒸馏/段重建、register mode queries与冻结主干prompt适应。
- 输入/输出：车辆历史与车道地图 → 多模态GMM未来轨迹及概率。
- 训练与评价：自蒸馏+GMM+重建，多任务权重；nuScenes、AV2、WOMD；2s历史/6s未来、K6、minADE/FDE/B-FDE等。
- 与本项目：跨域运动泛化相关，但属于不同输入/协议问题。重合判定：泛化/模式query已被研究，不能直接迁移其指标到行人8/12。
- 证据定位：paper§3.1–3.5/§4.1；车辆/地图输入限制。

## 与当前证据最接近的先例

TUTR §3.2已将训练mode中心、概率和轨迹联合建模：训练用GT形成监督标签是正常监督，不是部署oracle；在线TopK来自模型概率。官方train.py却逐轮调用名为test的数据loader并按ADE+FDE保存best，不能把这种入口直接移植为本项目严格validation选模方式。

MoFlow §4.2/4.3直接研究相关多候选及覆盖/多样性，代码声明ETH/UCY存在两种预处理版本，不能仅因同为8/12、K20就直接公平排名。Joint Metrics Matter已经把marginal最优不等于joint最优作为核心问题；只把该指标加进HTraj不构成新算法。

MoFlow固定版本的`models/context_encoder/eth_encoder.py:102–111`还使用节点索引位置编码与agent_query_embedding；这是静态结构风险提示，本轮没有执行其排列测试，不能宣称它已通过HTraj排列准入。没有新增候选模型复现或训练。

SocialCircle的“速度”meta分量实际用历史端点净位移长度，和本阶段平均相邻步速不完全相同。可区别具体定义，不能将常规运动特征不同定义本身宣布为创新。SocialTrans与Inter-GraphFormer的未核实项仍是新颖性判断风险，不由摘要补全。

本报告不使用任何这些论文的test数值来挑选本项目改进，不新建跨方法性能排行榜。
