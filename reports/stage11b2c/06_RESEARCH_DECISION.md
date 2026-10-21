# 机制与科研决策

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

**Q1 原始热图支持 — SUPPORTED**：998目标均保存真实权重与支持。25px实际阈值后支持为0的目标77个；熵不代表校准。

**Q2 GT25px概率质量 — SUPPORTED**：均值0.23531017、中位数0.17348301，逐目标见probability_mass.csv。

**Q3 10000抽样覆盖 — SUPPORTED**：920/998=92.184369%，来自真实抽样点。

**Q4 最终K20覆盖 — SUPPORTED**：834/998=83.567134%，旧缓存逐位一致。

**Q5 164失败中的原始命中 — SUPPORTED**：86/164；GT仅离线用于距离计算。

**Q6 最终失败与支持缺失 — SUPPORTED**：77/164共现，不等于原始Sigmoid为0或语义因果。

**Q7 原始命中后的覆盖丢失 — SUPPORTED**：86目标；不证明KMeans实现有错。

**Q8 Softargmax补充作用 — SUPPORTED**：25px补救0目标。其多峰均值失效原因仍UNVERIFIED。

**Q9 deathCircle机制 — PARTIALLY_SUPPORTED**：主事件分布[{'scope': 'deathCircle', 'counts': {'zero_support_final_miss': 67, 'proposal_hit_final_miss': 65, 'positive_support_sampling_miss_final_miss': 1}, 'largest': 'zero_support_final_miss'}, {'scope': 'other4', 'counts': {'proposal_hit_final_miss': 21, 'zero_support_final_miss': 10}, 'largest': 'proposal_hit_final_miss'}]；仅开发录像描述，不能泛化到全部SDD。

**Q10 高速/转向/变速关联 — PARTIALLY_SUPPORTED**：全部历史组见motion_mechanism_summary.csv；组均值有录像/窗口构成混杂，不证明运动状态导致覆盖损失。

**Q11 优先研究哪一环节 — PARTIALLY_SUPPORTED**：MECHANISM_INCONCLUSIVE；依据预先冻结的最终失败最大类及deathCircle/other4一致性，不是创新GO。

**Q12 语义不可靠直接证据 — UNVERIFIED**：没有受控语义干预或可靠语义GT，不能从当前热图推断。

**Q13 提出新方法的机制证据 — PARTIALLY_SUPPORTED**：仅具备本冻结链路的实证观察；新机制效果、可部署性与新颖性尚未验证，不能直接提出新算法。

**Q14 还需阅读哪些工作 — UNVERIFIED**：需复核Y-Net原TTST、TUTR聚类后处理替代、DLow集合多样性目标，以及受限候选预算下量化/覆盖代表点选择与概率保持方法。当前不是完整查重。

**Q15 下一阶段最低成本实验 — PARTIALLY_SUPPORTED**：若获批准，最低成本先使用当前缓存做簇级分配、簇半径与命中样点到最终中心距离的离线分解，并复核公开压缩/覆盖方法；同时区分已有相对阈值切除与网络分布低质量。不改变阈值、不重采样、不训练；有明确机制和查重后再冻结同K预算的独立对照。

[Y-Net原生成源码](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/test.py)已明确使用大量抽样、KMeans和Softargmax；因此这些常规构件不能换名作为创新。[TUTR正式论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf)已讨论聚类后处理的代价及概率信息问题；[DLow作者论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540324.pdf)研究多样化潜变量集合，但原任务是3D人体运动，不能直接称为相同SDD候选压缩实验。来源与访问限制见source_references.json；本轮Y-Net CVF链接返回403，未将其误记为全文审读成功。

最终机制建议 `MECHANISM_INCONCLUSIVE`；算法创新UNVERIFIED；新算法开发/训练NO-GO，自动下一阶段NO-GO。停止，等待审查。

历史组实际对照（25px，全部为既有历史分组，非因果）：

- hist_speed_group / low（n=266）：抽样命中后最终失覆盖率0.7519%；实际支持为0率3.3835%；最终失败率4.1353%。
- hist_speed_group / high（n=497）：抽样命中后最终失覆盖率12.6761%；实际支持为0率10.4628%；最终失败率23.3400%。
- hist_turn_group / straight（n=322）：抽样命中后最终失覆盖率7.4534%；实际支持为0率5.5901%；最终失败率13.3540%。
- hist_turn_group / turning_ge30deg（n=270）：抽样命中后最终失覆盖率13.3333%；实际支持为0率14.4444%；最终失败率27.7778%。
- hist_speed_change_group / stable_CVle0.15（n=394）：抽样命中后最终失覆盖率9.8985%；实际支持为0率6.0914%；最终失败率16.2437%。
- hist_speed_change_group / variable_CVge0.4（n=164）：抽样命中后最终失覆盖率9.7561%；实际支持为0率18.2927%；最终失败率28.0488%。

速度high/low是来源训练历史对角线归一化速度的冻结三分位组，不是绝对米/秒。上述群体有录像构成/重复窗口混杂，不构成独立因果或统计显著性判断。具体数值见history_motion_contrasts.csv。

明确的NOT_SUPPORTED判断：只有1/164最终失败属于“正质量但原始抽样未命中”，当前证据不支持随机抽样次数不足是主要失效来源；新算法直接启动条件也未得到支持。语义因果及多峰平均致错保持UNVERIFIED，而非已被否定。全部主张状态见scientific_claim_status.csv。
