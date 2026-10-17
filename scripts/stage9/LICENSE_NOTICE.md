# 第三方来源与许可

本阶段 `common.py` 中 `PairedLayer` 继承并调用作者的 ResonanceLayer，属于GPL-3.0代码适配，按GPL-3.0提供。完整许可见仓库中冻结的 `experiments/stage8/cache/Re/LICENSE`（原文件名以来源清单为准）。原作者 Conghao Wong，https://github.com/cocoon2wong/Re ，commit 7358b3d4fd3fcf160c18be9425445a67ee8211ee。保留原仓库，不修改其文件。Qpid c7233b3f7e7305294b1d5644e695b44727ac0bea；Luna b25d44354a75f7652c25c37aa9c16b353578ad79。

本阶段不是官方成绩复现：使用AgentFormer冻结的HOTEL折训练/验证区域和完整20帧目标；三个模型随机初始化，不载入作者权重。适配器仅改变数据入口；B/C仅在社会聚合表示处增加受控残差，其他网络直接调用官方实现。
