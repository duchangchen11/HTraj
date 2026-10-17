# 近期研究与查重

检索截至2026-10-18。使用nature-academic-search多源流程；无学术MCP，OpenAlex备用检索HTTP429，因此通过作者/arXiv/CVF与固定源码补充。不是穷尽性系统综述。仅Resonance、MoFlow进入工程审查；TUTR/SocialCircle/Rev/KDTF仅作方法先例，没有创建第三模型复现任务。

|论文|机制与阅读范围|对本轮问题的重合/限制|
|---|---|---|
|[AgentFormer](https://arxiv.org/abs/2103.14023) 2021 ICCV|Agent-aware temporal/social attention, learned prior, diversity learning；Reused full methods and frozen source|Ordinary social attention, latent diversification alreadycovered|
|[TUTR](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf) 2023 ICCV|Mode/social Transformer, regression+classification；Prior fullPDF/source; method/experimental sections reread|Scoring/anchors/mode selection not new; no new model admission|
|[SocialCircle](https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf) 2024 CVPR|Per-angular-sector speed/displacement,distance,angle means；Full method Eq1-11 and existing frozen source identity|Ordinary velocity/distance/angle additions and partition features alreadycovered|
|[MoFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_MoFlow_One-Step_Flow_Matching_for_Human_Trajectory_Forecasting_via_Implicit_CVPR_2025_paper.html) 2025 CVPR|K-shot flow matching regression+CE, IMLE/Chamfer one-step distillation；Full method/experiments, frozen teacher/student/train/eval/data source|Fewer steps or generic distillation/coverage improvement not new|
|[Resonance](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_Resonance_Learning_to_Predict_Social-Aware_Pedestrian_Trajectories_as_Co-Vibrations_ICCV_2025_paper.pdf) 2025 ICCV|Haar/differential motion, spectral pair product, angle aggregation；Final CVF fullpaper+supplement; official frozen network+Qpid+Luna|Motion/social decomposition, frequency features and social averaging alreadycovered|
|[Inter-GraphFormer](https://doi.org/10.1016/j.physa.2026.131566) 2026 PhysicaA694:131566|Sparse directedweighted spatial graph, parallel temporal/spatial attention, dualGLU；Publisher detailedIntroduction+metadata only; fullmethod unavailable|Dynamic/asymmetric interaction ideas overlap; precise formula comparison unknown, cannot claim novelty cleared|
|[Reverberation](https://arxiv.org/abs/2511.11164) 2026 IEEE TPAMI earlyonline Oct7; author repo stillsays underreview|Retains temporal-spectral x angular social tensor; two reverberation kernels；Full available arXiv methods/supplement, formal identity DOI10.1109/TPAMI.2026.3741469|Adding temporal angle tensors or generic delayed interaction to Re overlaps directly; within-sector identity pooling remains in inspected formulas|
|[Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Multi-modal_Knowledge_Distillation-based_Human_Trajectory_Forecasting_CVPR_2025_paper.pdf) 2025 CVPR|Intra/inter-agent embedding distillation withlocal/globalencoders；Full methods3.2/3.3 and protocol4.1|Different fromMoFlow distribution distillation, but generic KDnewness already insufficient; extra modalities not available for standard coordinateonly baseline|


## 已有工作解决到什么程度

研究问题A：AgentFormer已联合编码时空交互，TUTR已有mode/social两级编码；SocialCircle已编码角分区内运动、距离和方向。Resonance以中心化运动谱乘积和角聚合建模社会修正，普通速度/距离/角度或社会残差分解本身均非空白。Rev全文§III/IV Eq12–14进一步保留Th×Nθ角向时序谱，学习双核处理交互延迟；“把Re的时间维留下”“增加记忆/时延权重”与它高度重合。新检索定位2026年连续动态图/patch-wise图、2025多时刻交互等论文，但未完整阅读的只作排重线索，不能以摘要判定确切机制不同。

本轮具体问题是**位置—运动配对信息被独立聚合抹除**，并不是笼统说模型没有相对速度或时间模块。原代码和作者权重给出可复核的同表示反例；这仍不证明真实预测受损，也不证明简单的联合特征就是新算法。原论文补充已比较SocialPooling、GCN、SocialCircle与ResonanceGathering，不能重新添加普通图模块就宣称创新。

研究问题B：MoFlow §4.2/4.3已联合考虑K候选覆盖/一致性、回归与分类，并用IMLE/Chamfer进行一步蒸馏；少步生成、teacherstudent、集合匹配都已直接研究。KDTF §3.2/3.3将完整轨迹/姿态/文本教师的个体和交互知识蒸馏给较少模态学生，属于额外输入下的知识转移，不同于MoFlow分布蒸馏，也不能直接拿其指标作为纯坐标公平对照。当前没有取得本地teacher/student配对预测误差与同设备延迟—覆盖证据，因此不提出“再减层/再减步”的创新。

## 版本与全文边界

Resonance采用最终ICCV2025 CVF版（三位作者）及补充，不将早期arXiv四作者版本混作同一最终方法细节。Rev作者仓库README仍写underreview，而[公开身份](https://pubmed.ncbi.nlm.nih.gov/42848575/)已列TPAMI2026-10-07 earlyonline、DOI10.1109/TPAMI.2026.3741469；本文以此区分公开/正式状态，不把旧README当最新发表状态。

Inter-GraphFormer只取得出版社详细Introduction与元数据：有向加权稀疏图、并行时空Transformer、dualGLU、CVAE、图Transformerdecoder。完整后验/损失/资格公式和作者代码仍未取得，出版社请求失败，不绕过验证。因此指定文献中这篇**未达到完整方法精读**；其缺失细节构成新颖性未通过的限制，不能写成“它没有这个机制”。来源、PDF与版本SHA见literature_evidence.json及source_and_weight_SHA256.csv.gz。
