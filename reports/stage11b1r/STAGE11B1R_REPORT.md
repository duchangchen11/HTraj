# Stage11B-1R Y-Net K20随机采样复现性修复

**结论：SAMPLER_ENGINEERING_GO。仅采样工程准入，不自动启动30epoch，不代表预测性能、独立泛化或算法创新通过。**

分支feat/stage11b1r-sampling-reproducibility；基础5c5b6cac4ac8839fd0cd7ff31578a89ac44c17b2。协议与统计阈值训练前/观察前提交b3d9cb5冻结，官方Y-Net a345ada1557f53dc2acb4c07f71d340d366980da与所有Stage1—11B-1文件保持。90361项历史SHA通过，旧失败日志保留。

## 根因

独立最小复现固定概率图和四类RNG状态，原GPU proposal仍不同；固定proposal的KMeans及固定goal的decoder各5次逐位相同。旧完整K20 goal最大差0.009033203125原像素。本次问题定位在CUDA multinomial路径，非遗漏随机种子；具体低层kernel未单独跟踪。

## 修复

独立HTraj adapter保持原设备float32阈值和归一化权重，仅CPU本地Generator执行10000次有放回抽样，然后还原索引回传模型。19原KMeans中心+softargmax与20完整轨迹不变。KMeans NumPy choice/CPU空簇补点使用独立SHA随机域并恢复调用者状态。原源码、结构、权重和损失未改。CPU/GPU序列不要求相同；新adapter内部必须逐位重复。

## 真实检查

固定图5次；3目标各5次完整调用；20目标倒序、未来GT替换、场景缓存/重算，合计92次新K20输出；独立解释器3目标严格恢复/预测。所有proposal/goal/path SHA一致、最大坐标差0，形状及有限性通过；原像素还原独立验证。561项单元测试通过，含强制18空簇随机回退测试。没有放宽任何bitwise判据。

分布核查预注册4图×50次×10000抽样×2设备，总400万；16格同时频率界epsilon=0.003528871310，全部通过；原阈值权重逐位保留、无支持外采样。不将统计近似称为CPU/GPU bitwise等价。

## 成本

20目标新完整K20平均0.517899秒，旧对照0.487650秒，均值增加6.203%；CPU抽样约2.750ms。998目标估计8.619分钟，加10%余量9.481分钟。候选与聚类初始化不同会改变KMeans耗时，不能把全部变化归因CPU或外推为所有checkpoint速度。

峰值allocated/reserved 503.421/622.000MiB（新旧控制共同进程）。GPU worker总88.175208秒，含CPU测试的保守计费126.675208秒，预算3600秒内。模型总参数53155339、预测核心1641381，全部参数和缓冲不变，0训练更新。

## 证据与限制

configs/stage11b1r/sampling_protocol.json；results/tables/stage11b1r/minimal_original_case.json、fixed_probability_five_repeats.csv、real_K20_checks.csv、fresh_process_checks.csv、distribution_trials.csv、distribution_bins.csv、paired_target_timing.csv、GPU进程账单、pytest XML、历史SHA与checkpoint/工件SHA。原始案例和随机工程checkpoint仅在忽略cache保存，不上传权重/数据。

逐位结论限当前RTX3080/PyTorch/NumPy与串行执行，不保证跨硬件、驱动、库版本。16格Monte Carlo不是对全部类别的穷尽证明。32位SHA种子有限池碰撞情况单独记录。未经协调的外部线程仍可能干扰原KMeans全局RNG，应使用串行或进程隔离。

仍为HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY；原语义ID映射、分割权重训练清单、新验证潜在语义曝光及短池来源限制不因采样修复消失。所有工程前向使用训练目标与随机预测核心；未计算任何新的验证/测试ADE/FDE，不能判断Y-Net是否超过CV或是否欠收敛。

本轮完成后停止。后续有限训练需新的明确指令，并将本修复后的随机协议纳入训练/评价冻结配置；不自动恢复旧Stage11B-1训练入口、不增加epoch或任何新网络模块。
