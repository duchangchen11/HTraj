# 候选生成、压缩与新颖性审查

检索日期2026-10-22。使用nature-academic-search多源工作流；专业academic MCP未挂载。使用作者/CVF/arXiv原文及作者代码链接，OpenAlex降级检索因系统Python兼容性失败，改用既有Python3.10后HTTP429，未反复重试。记录见discovery_*failure.txt和literature_search_audit.json。12篇主论文全文下载到忽略缓存并阅读相关方法/公式；PDF、文本SHA与方法定位提交，全文不上传。不是穷尽性系统综述，不提供“已证明原创”结论。

### YNet（2021）

[From Goals, Waypoints & Paths to Long Term Human Trajectory Forecasting](https://arxiv.org/abs/2012.01526)；Karttikeya Mangalam; Yang An; Harshayu Girase; Jitendra Malik；ICCV 2021; methods source arXiv v1 (2020)。输入：历史坐标+场景图像；输出：Goal/waypoint热图+完整路径。TTST:10000 -> K-1 KMeans + 1 Softargmax。与本题：本项目完整生成链路来源；区别：现研究冻结链路的局部失覆盖，不是原论文新模型。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Test-Time Sampling Trick”、文本行227;283;618及PDF/text SHA见literature_review.csv；版本arXiv:2012.01526v1 [cs.CV] 2 Dec 2020。官方冻结代码a345ada；相对阈值和KMeans源码已读；全文为较早arXiv版本，不冒充最终版逐字相同。
### PECNet（2020）

[It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction](https://arxiv.org/abs/2004.02025)；Karttikeya Mangalam; Harshayu Girase; Shreyas Agarwal; Kuan-Hui Lee; Ehsan Adeli; Jitendra Malik; Adrien Gaidon；ECCV 2020。输入：历史+社会邻居；输出：端点CVAE+条件轨迹。核心截断trick不是聚类。与本题：有限候选与概率多样性取舍；区别：潜空间方差/截断，不是冻结proposal压缩。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Truncation trick”、文本行273;451;477;479;505及PDF/text SHA见literature_review.csv；版本arXiv:2004.02025v3 [cs.CV] 18 Jul 2020。代码仅通过作者项目定位，未执行。
### DLow（2020）

[DLow: Diversifying Latent Flows for Diverse Human Motion Prediction](https://arxiv.org/abs/2003.08386)；Ye Yuan; Kris Kitani；ECCV 2020。输入：过去关节运动；输出：相关潜码解码K种未来。不是KMeans后处理。与本题：有限候选多样性不是新问题；区别：动作潜空间，不是2D端点采样云。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Latent Mapping Functions”、文本行259;276;278;324;326及PDF/text SHA见literature_review.csv；版本arXiv:2003.08386v2 [cs.CV] 22 Jul 2020。K头仿射共享噪声；RBF成对能量+KL；与DPP论文分开。
### AgentFormer（2021）

[AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting](https://arxiv.org/abs/2103.14023)；Ye Yuan; Xinshuo Weng; Yanglan Ou; Kris Kitani；ICCV 2021。输入：多行人历史+可选地图；输出：CVAE+DLow完整轨迹。无TTST KMeans核心链路。与本题：历史条件有限候选与DLow已有；区别：修改完整模型不是固定proposal对照。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Trajectory Sampler”、文本行84;291;380;382;384及PDF/text SHA见literature_review.csv；版本arXiv:2103.14023v3 [cs.AI] 7 Oct 2021。已读CVAE/式7及采样器式9；不重算已有测试指标。
### TUTR（2023）

[Trajectory Unified Transformer for Pedestrian Trajectory Prediction](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.html)；Liushuai Shi; Le Wang; Sanping Zhou; Gang Hua；ICCV 2023。输入：历史+邻居+训练集运动模式；输出：并行轨迹与概率。训练前聚类模式；推理不聚类。与本题：聚类后处理概率丢失已有明确问题定义；区别：模型级并行输出，不是固定采样云代表点。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Global Prediction”、文本行48;105;135;150;164及PDF/text SHA见literature_review.csv；版本driving. arXiv preprint arXiv:1909.07707, 2019. 1                 [29]   Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-。全文§3；训练模式聚类不可误写为完全无聚类。
### DPP（2020）

[Diverse Trajectory Forecasting with Determinantal Point Processes](https://arxiv.org/abs/1907.04967)；Ye Yuan; Kris M. Kitani；ICLR 2020。输入：历史条件特征；输出：DSF潜码与多样轨迹集。非KMeans后处理；可做集合选择。与本题：质量/概率和多样性权衡直接重合；区别：学习潜码集合，不是已冻结19簇内替换代表。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Quality.”、文本行328;356及PDF/text SHA见literature_review.csv；版本arXiv:1907.04967v2 [cs.CV] 23 Dec 2019。方法§4，核Diag(r)S Diag(r)；质量不是直接用轨迹似然以免只保大模式。
### DenseTNT（2021）

[DenseTNT: End-to-End Trajectory Prediction From Dense Goal Sets](https://arxiv.org/abs/2108.09640)；Junru Gu; Chen Sun; Hang Zhao；ICCV 2021。输入：目标历史+道路向量；输出：稠密Goal分布到有限Goal/路径集。优化/集合预测替代NMS。与本题：概率加权有限K代表性问题直接先例；区别：地图车辆与可训练集合头，不是SDD冻结cloud。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Optimization for Offline Goal Set Prediction”、文本行273及PDF/text SHA见literature_review.csv；版本arXiv:2108.09640v2 [cs.CV] 24 Nov 2021。方法§3；f(Y)=sum h(c) min_y ||c-y||；不能把该目标包装新颖。
### NPSN（2022）

[Non-Probability Sampling Network for Stochastic Human Trajectory Prediction](https://arxiv.org/abs/2203.13471)；Inhwan Bae; Jin-Hwi Park; Hae-Gon Jeon；CVPR 2022。输入：目标/邻居过去路径；输出：N个目的性潜采样/轨迹。无必要KMeans后处理。与本题：历史运动引导候选/低差异采样已有；区别：新采样而非保留既有10000点。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“NPSN Architecture”、文本行254及PDF/text SHA见literature_review.csv；版本CVF official PDF。§4 式8/9、QMC与Box-Muller；不可称引入历史状态即新。
### HYPER（2022）

[HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling](https://arxiv.org/abs/2110.02344)；Xin Huang; Guy Rosman; Igor Gilitschenski; Ashkan Jasour; Stephen G. McGill; John J. Leonard; Brian C. Williams；ICRA 2022; arXiv v1 2021。输入：过去轨迹+道路图；输出：混合意图轨迹样本及似然。最高似然初始化FPS选终点分离样本。与本题：直接使用FPS进行预测样本压缩；区别：车辆混合意图，不是YNet热图聚类。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Trajectory Sample Selection”、文本行283;339及PDF/text SHA见literature_review.csv；版本arXiv:2110.02344v1 [cs.RO] 5 Oct 2021。§IV-D/V；FPS已经用于轨迹预测，不能声称只属于数学背景。
### LRVQ2024（2024）

[Trajectory Forecasting through Low-Rank Adaptation of Discrete Latent Codes](https://arxiv.org/abs/2405.20743)；Riccardo Benaglia; Angelo Porrello; Pietro Buzzega; Simone Calderara; Rita Cucchiara；ICPR 2024 (author official README); downloaded arXiv v2。输入：历史+社会上下文；输出：VQ-VAE/扩散轨迹到K中心。N=200路径 -> K=20 k-means质心。与本题：2024仍直接研究过采样到KMeans压缩；区别：完整轨迹聚类，非10000Goal19+Soft。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Enforcing Effective Multi-modal Forecasting”、文本行383及PDF/text SHA见literature_review.csv；版本arXiv:2405.20743v2 [cs.CV] 29 Aug 2024。§4.4：实际方法全文，不只摘要；质心均值降量化噪声的论点需实验对照。
### GoalFlow2025（2025）

[GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://arxiv.org/abs/2503.05689)；Zebin Xing; Xingyu Zhang; Yang Hu; Bo Jiang; Tong He; Qian Zhang; Xiaoxiao Long; Wei Yin；CVPR 2025; downloaded arXiv v6 2025-10-01。输入：传感器BEV+自车状态；输出：Goal引导flow候选与评分。非YNet TTST19簇。与本题：场景条件Goal与候选选择相近；区别：任务/输入/指标不同，不冒充直接行人同题。风险MEDIUM。证据FULLTEXT_METHOD_REVIEWED，锚点“Goal Point Scorer”、文本行201;300及PDF/text SHA见literature_review.csv；版本arXiv:2503.05689v6 [cs.CV] 1 Oct 2025。已读Goal构建和规划模块；不作为当前预测指标比较。
### GMM2026（2026）

[Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs](https://arxiv.org/abs/2606.26424)；Qiyuan Wu; Katie Z Luo; Bharath Hariharan; Wei-Lun Chao; Mark Campbell；作者arXiv注明Accepted by ECCV 2026；最终proceedings未逐字核验。输入：既有模式轨迹与后验概率；输出：合并后的有限模式。测试时概率加权KMeans合并。与本题：概率保持/加权聚类2026直接冲突；区别：WTA模式而非重复离散Goal样本；并非完全等价。风险HIGH。证据FULLTEXT_METHOD_REVIEWED，锚点“Test-Time Merging”、文本行404;410;671及PDF/text SHA见literature_review.csv；版本arXiv:2606.26424v1 [cs.LG] 24 Jun 2026。§5.1：按终点分配，全轨迹按模式后验加权均值；不能声称概率加权为原创。

## 必须分清的先例

DenseTNT的目标是sum_x p(x) min_c ||x-c||，直接涉及概率加权有限Goal集合；HYPER在轨迹预测中实际使用FPS，故FPS不只是无关数学背景；DPP明确讨论高概率主要模式与少数模式的质量取舍；NPSN已经以历史路径条件化有限采样；TUTR已经讨论聚类后处理丢失输出概率。TUTR在训练数据上聚类运动模板，不能误写为任何阶段都无聚类。

2024 LRVQ §4.4使用200路径到20个KMeans质心。发表来源由作者官方README核为ICPR2024，不沿用初步ECCV猜测。2026 GMM工作§5.1按模式概率加权合并，§5.2一步EM需要训练集GT做M-step；不能把两者都称为无训练无GT。其实验为nuScenes和WOMD，不把参考文献里的Argoverse2误作实验集。GoalFlow为自车规划、PDMS任务，相近思路不等于行人同任务。检索中的SIPTraj同名资料有歧义，未用不同论文互相佐证。

## 六方向创新冲突矩阵

| direction | proposal | closest_work | current_evidence | innovation_risk |
| --- | --- | --- | --- | --- |
| A | 直接替换KMeans19 | YNet;LRVQ2024;HYPER | 86有失覆盖；不证明替代有效 | HIGH |
| B | 增加候选多样性 | DLow;DPP;NPSN;AgentFormer | 几何离散度不等于性能收益 | HIGH |
| C | 按概率质量调整代表性 | DenseTNT;GMM2026;TUTR | 低质量局部被遗漏；未证明权重重复使用合理 | HIGH |
| D | 高低概率区域代表性 | DPP;NPSN;HYPER;DenseTNT | 失败组低局部质量；保护所有尾部可能保噪声 | HIGH |
| E | 历史运动条件压缩 | NPSN;TUTR;LRVQ2024 | 仅有事后分组关联，未证独立收益 | HIGH |
| F | 语义可靠性候选生成 | YNet;GoalFlow2025 | 本阶段没有语义致错直接证据 | UNKNOWN |

A—E均HIGH：直接替换KMeans、增多样性、加概率权重、保护尾部、引入历史条件都有明显先例；这不等于本次证明每一种特定实现完全相同。F为UNKNOWN新颖性且没有当前语义因果证据，因此不准入。LOW在本次矩阵没有使用。K-center/FPS、k-medoids、DPP、量化的不同目标见mathematical_prior_art.csv；书/经典论文只作为数学背景，未冒充已在同一行人任务实现。

当前链路重复样本已经包含经验概率质量，不能用“原聚类不含概率”建立虚假的创新缺口。单纯改名、套FPS、加权KMeans或常规medoid不构成原创。尚未找到区别明确、证据充分的新数学机制（UNVERIFIED）；不强凑第二个创新假设。
