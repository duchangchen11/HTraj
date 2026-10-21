# 收敛与研究决策

适用范围：PROTOCOL_DEVELOPMENT_ONLY / CONVERGENCE_DIAGNOSTIC_ONLY。单seed17、5个开发验证录像、998个重叠目标窗口，并非998个独立场景。单位为SDD参考图像像素，不能与ETH/UCY米制排名。分割语义ID映射与预训练成员身份未完全核实，验证可能存在语义预训练曝光。未读取封存测试集，未新增模型、loss、调参或数据增强。

**Q1 是否完成Epoch50**：COMPLETED_EPOCH50；最终训练50，最终完整验证50。

**Q2 minADE20变化**：16.403136→18.562667px；变化+2.159531px（+13.17%，正值为退化）。

**Q3 A25失败率**：16.432866%→21.042084%。

**Q4 A25误差份额**：61.081748%→65.470210%。

**Q5 原164个恢复数量**：31

**Q6 新失败数量**：77

**Q7 deathCircle_3**：最终A25=143/443，minADE20=29.160778px。与全部录像比较见scene_metrics.csv；不作语义归因。

**Q8 K1/K20差距**：ADE均值15.689964→15.902010px；FDE34.904992→35.612900px。

**Q9 Goal正确但Path错误**：7→2；身份转移见target_cohort_transitions.csv。

**Q10 是否平台期**：训练loss缓慢下降，但4次新增验证均未超过Epoch30，未见验证随训练持续改善。该结果更接近泛化波动/退化，不能证明充分收敛或区分过拟合与目标函数不一致；未追加Epoch60。

**Q11 有限预算H1**：H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET

**Q12 是否值得直接开发目标候选方法**：仅可决定是否值得下一步机制/文献审查，不具备直接创新开发GO。

**Q13 语义不确定性直接证据**：没有；无可靠语义GT/映射及受控语义干预，不能因录像困难或路径偏离作因果归因。

**Q14 下一步一个待验证假设**：持续覆盖失败是否主要表现为GT终点附近热图概率质量不足，还是固定TTST proposal→KMeans压缩丢失覆盖？需要先作固定权重的概率/抽样分解和已有方法查重；本阶段未执行，待科研决策批准。

证据索引：Q1/2/8/10见epochs.csv、validation_metrics.csv和training_execution.json；Q3/4/11见goal_coverage_epochs.csv；Q5/6见a25_transition.csv；Q7见scene_metrics.csv；Q9见candidate_errors_epoch*.csv.gz及target_cohort_transitions.csv；Q12—14为限定性研究判断/待验证假设。所有判断仅适用于本开发协议和本次预算。

工程恢复GO=True，执行状态=COMPLETED_EPOCH50；H1=H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET。进一步Goal覆盖问题审查=True；直接算法开发GO=False；语义因果证据=False；自动追加训练GO=False；4次新增验证均不及Epoch30，不建议立即追加相同训练到Epoch60。完成后停止，不进入Stage11C。
