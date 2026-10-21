# Goal权重与GT邻域离散概率

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

保留原logits、Sigmoid权重、阈值后未归一化权重、实际torch.multinomial使用的float32权重。Sigmoid不等于全局概率。统计中p=w/sum64(w)仅为数学离散分布定义，不反馈给原采样器；原float32权重及其总和单独留存。GT-neighborhood坐标为network/.25，包含原画布全部支持，同时区分真实图像与padding。

25px质量均值0.23531017、中位数0.17348301；分位数P10/P90=0.0021964246/0.59074197。零质量目标77/998。以固定10000次有放回理论P_hit<.5为解析低质量参考时，共78个；这不是事后调出的分类阈值，也不等同实际未命中。

| radius | targets | P_mass_mean | P_mass_median | support_zero_count | low_mass_reference_count | theoretical_P_hit_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 998 | 0.049425 | 0.029857 | 93 | 96 | 0.902540 |
| 25 | 998 | 0.235310 | 0.173483 | 77 | 78 | 0.920427 |
| 50 | 998 | 0.509531 | 0.555310 | 53 | 54 | 0.945599 |

各目标支持像素数、熵（自然对数）、max概率、top1%画布概率集中度、原Sigmoid最大值/和、阈值权重和、实际float32权重double求和、padding质量及GT画布内外标记均见probability_mass.csv和sample_level_mechanism.csv。高熵并不证明校准正确；质量为0表示实际阈值后离散支持缺失，不表示原始Sigmoid必为0，更不能证明语义错误。GT是否在画布内、是否在真实图像内分开标注。

原始张量保持float32/int64，无量化，目录在忽略的experiments/stage11b2c/cache/tensors；每文件及各中间张量SHA见tensor_manifest.csv。CPU评分只在generation_execution.json和process_extract.json完整通过后启动。

固定阈值支持细分：77个实际25px零质量目标中，原始Sigmoid邻域权重和>0且被原相对阈值切除的为77个，原Sigmoid邻域权重和为0的为0个。前者未阈值化Sigmoid的数学归一化邻域质量中位数7.7329512e-05，范围1.7393457e-23—0.0028658572。这些是已记录张量的静态分解，没有更改阈值、重采样或计算新候选成绩；不能据此推断降低阈值/修改Goal网络的收益。详见threshold_support_details.csv/json。
