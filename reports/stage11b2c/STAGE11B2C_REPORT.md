# Stage11B-2C：Goal分布与候选生成机制审计

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

**完成998目标正序导出和全部998目标反序逐位回归。** 原probability/weights/proposal/goal/path五项SHA及20目标/20路径逐位一致，额外导出未改变Epoch30预测。模型状态SHA `62453170be18a7ecea3948a921954004e86dc950fe1a91dba1d91997aacfc741` 未变化，无训练步。

| radius | proposal_cover_count | proposal_cover_fraction | final_cover_count | final_cover_fraction | compression_loss_count | final_recovery_count | support_zero_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 902 | 0.903808 | 509 | 0.510020 | 393 | 0 | 93 |
| 25 | 920 | 0.921844 | 834 | 0.835671 | 86 | 0 | 77 |
| 50 | 944 | 0.945892 | 899 | 0.900802 | 45 | 0 | 53 |

25px原始10000抽样覆盖920，最终K20覆盖834；最终失败164个中，零支持77、正支持但抽样未命中1、抽样命中后最终失去覆盖86。不得将三组误差均值差解释成模块因果贡献。

GT25px概率质量均值0.23531017、中位数0.17348301。Softargmax在25px补救KMeans19遗漏0个。最终反而命中而原样点未命中0个，本次没有观测到最终覆盖突破原始抽样覆盖的反例；算法定义仍不保证子集关系，不能把未观测当成数学约束。

机制优先建议 `MECHANISM_INCONCLUSIVE`；跨deathCircle与其他4录像的最大类一致性=False。这是观察性链路审计，不能证明新算法可提升预测，也不能证明语义因果。

GPU新增总进程wall 658.470秒，包含启动、加载、采样、聚类、完整解码、张量导出、回归和退出，低于7200硬限/6600安全线。原精度大张量仅保存在忽略cache，Git不含原始数据或权重。模型/数据/原缓存及全部中间SHA、时间、显存与失败日志分别见input_export.json、tensor_manifest.csv、generation_execution.json、gpu_budget.json和cache_integrity.json。

报告01为原链路与GT隔离；02概率质量；03原始抽样；04压缩与Softargmax；05全录像及历史分组；06逐项科研问答与来源。图件PNG/SVG与全部冻结规则案例及sidecar身份/SHA保留，缺失类型不虚构案例。

验收记录见full_tests_final.txt/xml、history_after.json和delivery_audit.json。已有语义预训练与协议限制未解除；没有测试集评分、新训练、改采样或新模型。新算法开发NO-GO；仅可在后续批准后开展机制与文献独立复核。

固定阈值支持细分：77个实际25px零质量目标中，原始Sigmoid邻域权重和>0且被原相对阈值切除的为77个，原Sigmoid邻域权重和为0的为0个。前者未阈值化Sigmoid的数学归一化邻域质量中位数7.7329512e-05，范围1.7393457e-23—0.0028658572。这些是已记录张量的静态分解，没有更改阈值、重采样或计算新候选成绩；不能据此推断降低阈值/修改Goal网络的收益。详见threshold_support_details.csv/json。

Softargmax的作用依赖半径：10px下独立补救KMeans19遗漏14目标（合计509 vs495），25px/50px下独立补救均为0。不能用25px无新增覆盖推断它在所有精度要求下无作用；1/19/20候选来源比较不是公平重训练实验。

最终验收：全仓pytest为614 passed / 1 skipped，原CUDA专项跳过原因保留。首次全仓收集因与旧阶段测试文件同名而冲突，失败txt/xml保留，仅将新文件改为唯一名称后通过；没有修改旧测试。原始数据文件及派生纯历史输入的完整SHA索引另见data_provenance.json。

资源测量：额外被动导出与SHA校验合并开销约0.0168秒/目标；纯概率图D2H未单独计时，未伪填独立数值。原Goal前向、采样、KMeans、完整20路径解码与缓存写入分别测量，完整范围见resource_summary.json。继承的父计时器通用文案虽列举train等阶段，本轮实际只有precheck/export/reverse regression，模型训练步和参数更新均为0。
