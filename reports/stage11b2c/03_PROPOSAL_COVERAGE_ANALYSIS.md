# 原始10000次抽样覆盖

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

每sample_id独立CPU Generator和冻结SHA seed，有放回10000次，重复样点未去重再聚类。唯一位置数均值1446.78，重复抽样次数均值8553.22；10000不是10000个不同位置。

| radius | proposal_cover_count | proposal_cover_fraction | positive_mass_sampling_miss_count | theoretical_P_hit_mean |
| --- | --- | --- | --- | --- |
| 10 | 902 | 0.903808 | 3 | 0.902540 |
| 25 | 920 | 0.921844 | 1 | 0.920427 |
| 50 | 944 | 0.945892 | 1 | 0.945599 |

25px实际覆盖920/998=92.184369%；正质量但实际抽样未命中1个。每目标实际最近距离、各邻域命中次数和索引/坐标SHA见proposal_coverage.csv。P_hit=1-(1-P_mass)^10000使用稳定expm1/log1p计算，仅是固定离散分布独立有放回假设下的理论量；不能用它替代真实命中事件。

所有sample_id原probability/weights/proposal/goal/path SHA与旧Epoch30一致，全部998反序二次生成的原始精度中间张量逐位相同。采样与GT评分相隔独立进程，GT未选择随机数或样点。
