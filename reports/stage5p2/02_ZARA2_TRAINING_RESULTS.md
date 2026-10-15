# ZARA2 正式训练结果

已真实完成 seed17 的100个完整epoch；每轮 23545 个训练目标窗口，累计 18400 次 optimizer 更新。仅训练集更新参数，验证录像整体排除。模型独立随机初始化，不加载任何checkpoint。训练代码提交 `eb8d494bb1373462fcd07da46abeb5e7bb2533f7`。

| 选择 | epoch | 评价口径 | ADE/m | FDE/m |
| --- | --- | --- | --- | --- |
| best | 30 | K1-prior | 0.2726 | 0.6075 |
| best | 30 | K1-first-sample | 0.4099 | 0.9473 |
| best | 30 | K20-independent-minima | 0.1575 | 0.3048 |
| last | 100 | K1-prior | 0.3290 | 0.7398 |
| last | 100 | K1-first-sample | 0.3459 | 0.7790 |
| last | 100 | K20-independent-minima | 0.2506 | 0.5458 |


best 按验证 K1 latent-prior ADE 唯一选择，第 30 轮；FDE、首样本与 K20 都来自同一个 checkpoint，不按各自最低值拼接。last 为第100轮。K20 两项分别按候选取最小值，可能来自不同候选；FPC关闭。

## 训练与泛化诊断

- 参数量：2,144,002；未引入新参数。原始训练目标仍为 reconstruction + MC latent KL，不是精确边缘NLL。
- 第1轮训练总loss 0.419614，第100轮 0.008465（rec 0.005351 + KL 0.003113）。训练曲线是更新中各batch的目标数加权均值。
- best 固定权重完整训练集 K1 ADE/FDE：0.4559/0.9964m；验证：0.2726/0.6075m。训练集也含重叠窗口，不能当独立个体样本。
- best 验证 posterior loss/rec/KL：0.015872/0.006351/0.009521。这是使用未来GT的监督posterior诊断；预测与分组不读取未来GT。
- 完整run墙钟 2087.09s（34.78min），训练GPU计算合计 1717.27s；峰值allocated 1.847GiB，reserved 3.297GiB；梯度L2最大值 7.1470；全部逐轮参数有限性检查通过。
- 记录了100份不可覆盖checkpoint及SHA；第1、10……100轮和best的模型/optimizer/RNG恢复测试通过。该run未发生NaN/Inf导致的停止或重试。

| 指标 | 81–90轮均值 | 91–100轮均值 | 后十轮−前十轮 | 末20轮线性斜率/轮 |
| --- | --- | --- | --- | --- |
| K1_prior_ADE_m | 0.327556 | 0.329114 | 0.001558 | 0.000387 |
| K1_prior_FDE_m | 0.740411 | 0.739679 | -0.000732 | 0.000593 |
| minADE20_m | 0.231111 | 0.248875 | 0.017764 | 0.001841 |
| minFDE20_m | 0.500443 | 0.539924 | 0.039481 | 0.004320 |
| posterior_loss | 0.036882 | 0.047482 | 0.010600 | 0.001070 |
| train_kl | 0.004378 | 0.003444 | -0.000933 | -0.000094 |
| train_loss | 0.010296 | 0.008985 | -0.001311 | -0.000133 |
| train_rec | 0.005918 | 0.005541 | -0.000377 | -0.000039 |


最优点不在最后10轮，但这不足以证明所有指标已收敛。训练loss下降不保证验证ADE下降；随机posterior目标与确定性latent-prior轨迹ADE不同。末段斜率仅描述这一固定预算，未据此调参或延长训练。

## 可追溯文件

- `results/tables/stage5p2/crowds_zara02/provenance.json`：来源、manifest SHA、初始权重fingerprint、环境、code commit。
- `training_curve.csv` 与 `epochs/epochNNN.json`：全部100轮；训练集K1仅在预注册轮次计算，其他轮留空。
- `epochs/epochNNN_steps.csv.gz`：每个batch的原始loss分项、梯度和耗时；`epochNNN_targets.csv.gz`：完整验证集逐目标误差。
- `checkpoint_inventory.csv`、`selection.json`、`checkpoint_restore_checks.json`：权重位置/SHA、配对选择和恢复。
- best `experiments/stage5p2/checkpoints/crowds_zara02/epoch030.pth`，SHA `e60594d67deeee83c73660a1a61fc387e10acc1cd4c54679a12d3df48cd75dd5`；last `experiments/stage5p2/checkpoints/crowds_zara02/epoch100.pth`，SHA `feeaceba2bb1024d14d2863fd980c421925f86d731dfd9b45d18cfc45afa38aa`。权重仅本地保存且被Git忽略。

## 最终验收补充（真实数据）

同一训练cohort的确定性K1 ADE从best轮0.4559降至第100轮0.1156m，验证却从0.2726升至0.3290m；验证posterior目标从0.015872升至0.050282。这支持后期过拟合/录像迁移下泛化退化，不能仅用best轮训练误差与验证误差的大小关系判断。

末20轮训练loss仍下降；HOTEL验证K1虽末段回落但远弱于早期best，ZARA2末段K1基本平台且K20继续恶化。最优轮不在预算边界，未发现必须自动延长的依据；不宣称所有优化量已收敛。

完整单元测试302项通过（`final_tests.xml`）。额外固定权重CUDA诊断8项中：同batch置换K1/K20的4项预测最大差异均为0；跨batch拆分4项均未通过预设atol/rtol=1e-5，失败标记保留，未放宽阈值。ZARA2 K1/K20最大坐标差为0.000204086/0.001112938m，HOTEL为0.000275135/0.000776291m。标准高斯epsilon按目标ID完全一致；这说明控制随机数并不消除浮点批处理与padding布局带来的数值差异。官方网络/epsilon/主评价batch=128均未修改，没有重训或重选checkpoint。原始失败现场与完整诊断见`CUDA_initial_tolerance_failure.txt`、`CUDA_sampling_invariance_diagnostic.json`。这些只是各录像16个固定索引的功能诊断，不是全验证集最大误差上界，也不证明差异完全由某一GPU算子造成。
