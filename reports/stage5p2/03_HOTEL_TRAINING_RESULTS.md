# HOTEL 正式训练结果

已真实完成 seed17 的100个完整epoch；每轮 27145 个训练目标窗口，累计 21300 次 optimizer 更新。仅训练集更新参数，验证录像整体排除。模型独立随机初始化，不加载任何checkpoint。训练代码提交 `eb8d494bb1373462fcd07da46abeb5e7bb2533f7`。

| 选择 | epoch | 评价口径 | ADE/m | FDE/m |
| --- | --- | --- | --- | --- |
| best | 3 | K1-prior | 0.3872 | 0.7515 |
| best | 3 | K1-first-sample | 0.5303 | 1.0754 |
| best | 3 | K20-independent-minima | 0.2025 | 0.3253 |
| last | 100 | K1-prior | 0.6470 | 1.4422 |
| last | 100 | K1-first-sample | 0.6720 | 1.4965 |
| last | 100 | K20-independent-minima | 0.4703 | 1.0387 |


best 按验证 K1 latent-prior ADE 唯一选择，第 3 轮；FDE、首样本与 K20 都来自同一个 checkpoint，不按各自最低值拼接。last 为第100轮。K20 两项分别按候选取最小值，可能来自不同候选；FPC关闭。

## 训练与泛化诊断

- 参数量：2,144,002；未引入新参数。原始训练目标仍为 reconstruction + MC latent KL，不是精确边缘NLL。
- 第1轮训练总loss 0.424568，第100轮 0.008494（rec 0.005210 + KL 0.003285）。训练曲线是更新中各batch的目标数加权均值。
- best 固定权重完整训练集 K1 ADE/FDE：0.5135/1.1081m；验证：0.3872/0.7515m。训练集也含重叠窗口，不能当独立个体样本。
- best 验证 posterior loss/rec/KL：0.034527/0.018103/0.016424。这是使用未来GT的监督posterior诊断；预测与分组不读取未来GT。
- 完整run墙钟 2338.62s（38.98min），训练GPU计算合计 1983.80s；峰值allocated 1.848GiB，reserved 3.260GiB；梯度L2最大值 17.8853；全部逐轮参数有限性检查通过。
- 记录了100份不可覆盖checkpoint及SHA；第1、10……100轮和best的模型/optimizer/RNG恢复测试通过。该run未发生NaN/Inf导致的停止或重试。

| 指标 | 81–90轮均值 | 91–100轮均值 | 后十轮−前十轮 | 末20轮线性斜率/轮 |
| --- | --- | --- | --- | --- |
| K1_prior_ADE_m | 0.692395 | 0.676345 | -0.016050 | -0.002331 |
| K1_prior_FDE_m | 1.538324 | 1.506598 | -0.031726 | -0.005091 |
| minADE20_m | 0.476105 | 0.481433 | 0.005328 | 0.000064 |
| minFDE20_m | 1.034594 | 1.044719 | 0.010125 | -0.000282 |
| posterior_loss | 0.057899 | 0.062747 | 0.004848 | 0.000578 |
| train_kl | 0.004569 | 0.003611 | -0.000958 | -0.000095 |
| train_loss | 0.010341 | 0.008956 | -0.001385 | -0.000139 |
| train_rec | 0.005771 | 0.005345 | -0.000426 | -0.000044 |


最优点不在最后10轮，但这不足以证明所有指标已收敛。训练loss下降不保证验证ADE下降；随机posterior目标与确定性latent-prior轨迹ADE不同。末段斜率仅描述这一固定预算，未据此调参或延长训练。

## 可追溯文件

- `results/tables/stage5p2/biwi_hotel/provenance.json`：来源、manifest SHA、初始权重fingerprint、环境、code commit。
- `training_curve.csv` 与 `epochs/epochNNN.json`：全部100轮；训练集K1仅在预注册轮次计算，其他轮留空。
- `epochs/epochNNN_steps.csv.gz`：每个batch的原始loss分项、梯度和耗时；`epochNNN_targets.csv.gz`：完整验证集逐目标误差。
- `checkpoint_inventory.csv`、`selection.json`、`checkpoint_restore_checks.json`：权重位置/SHA、配对选择和恢复。
- best `experiments/stage5p2/checkpoints/biwi_hotel/epoch003.pth`，SHA `77a6d129a9bda4b932414ecd465f3c2b8daab1ea5161529aabe099edb418aa51`；last `experiments/stage5p2/checkpoints/biwi_hotel/epoch100.pth`，SHA `ea6ad6f2f70d1952c6cd74ba8d22ded19343c5832ddc04952fde426792a5c5e3`。权重仅本地保存且被Git忽略。

## 最终验收补充（真实数据）

同一训练cohort的确定性K1 ADE从best轮0.5135降至第100轮0.1146m，验证却从0.3872升至0.6470m；验证posterior目标从0.034527升至0.068391。这支持后期过拟合/录像迁移下泛化退化，不能仅用best轮训练误差与验证误差的大小关系判断。

末20轮训练loss仍下降；HOTEL验证K1虽末段回落但远弱于早期best，ZARA2末段K1基本平台且K20继续恶化。最优轮不在预算边界，未发现必须自动延长的依据；不宣称所有优化量已收敛。

完整单元测试302项通过（`final_tests.xml`）。额外固定权重CUDA诊断8项中：同batch置换K1/K20的4项预测最大差异均为0；跨batch拆分4项均未通过预设atol/rtol=1e-5，失败标记保留，未放宽阈值。ZARA2 K1/K20最大坐标差为0.000204086/0.001112938m，HOTEL为0.000275135/0.000776291m。标准高斯epsilon按目标ID完全一致；这说明控制随机数并不消除浮点批处理与padding布局带来的数值差异。官方网络/epsilon/主评价batch=128均未修改，没有重训或重选checkpoint。原始失败现场与完整诊断见`CUDA_initial_tolerance_failure.txt`、`CUDA_sampling_invariance_diagnostic.json`。这些只是各录像16个固定索引的功能诊断，不是全验证集最大误差上界，也不证明差异完全由某一GPU算子造成。
