# Stage 5.2 正式SocialVAE基线总报告

已完成两个预注册独立seed17 run，各100个完整epoch；训练/验证精确沿用Stage5.1因果manifest。官方网络与原始重建+MC-KL损失保持冻结，FPC关闭，未读取ETH指标，未启动额外训练。训练配置与代码先提交，再执行正式参数更新；执行提交 `eb8d494bb1373462fcd07da46abeb5e7bb2533f7`。

| 录像 | 模型/指标 | best epoch | ADE/m | FDE/m |
| --- | --- | --- | --- | --- |
| HOTEL | CV-Last2 | — | 0.3295 | 0.6110 |
| HOTEL | CV-OLS4 | — | 0.2378 | 0.4389 |
| HOTEL | CV-OLS8 | — | 0.2201 | 0.4088 |
| HOTEL | SocialVAE K1-prior | 3 | 0.3872 | 0.7515 |
| HOTEL | SocialVAE K1-first-sample | 3 | 0.5303 | 1.0754 |
| HOTEL | SocialVAE K20-independent-minima | 3 | 0.2025 | 0.3253 |
| ZARA2 | CV-Last2 | — | 0.2963 | 0.6675 |
| ZARA2 | CV-OLS4 | — | 0.3319 | 0.7157 |
| ZARA2 | CV-OLS8 | — | 0.3832 | 0.7872 |
| ZARA2 | SocialVAE K1-prior | 30 | 0.2726 | 0.6075 |
| ZARA2 | SocialVAE K1-first-sample | 30 | 0.4099 | 0.9473 |
| ZARA2 | SocialVAE K20-independent-minima | 30 | 0.1575 | 0.3048 |


上表SocialVAE三个口径共用该录像K1-prior ADE选定的best checkpoint；K20独立最小ADE/FDE。完整last及逐轮结果没有省略，见02/03报告。三个CV使用完全相同的窗口、目标和GT，重新计算并核对旧SHA；没有选择性只展示弱CV。

## 交付索引

- `01_FROZEN_TRAINING_PROTOCOL.md`：预注册配置、采样、阈值、选择与停止规则。
- `02_ZARA2_TRAINING_RESULTS.md`、`03_HOTEL_TRAINING_RESULTS.md`：best/last、完整预算、趋势、耗时/显存、checkpoint SHA。
- `04_CV_COMPARISON.md`：三个CV、K1/K20分离、窗口/行人等权与录像宏平均。
- `05_ERROR_AND_GENERALIZATION.md`：全部历史运动与交互分组，含退化/不可观测/空组。
- `06_RESEARCH_DECISION.md`：十项问题、证据边界及停止决定。
- `results/tables/stage5p2/`：全部真实配置来源、逐轮逐batch、逐目标配对/分组、采样seed/noise SHA、模型/数据SHA、完整执行日志和测试结果。

## 限制与验收

两个录像已有历史研究曝光，只有seed17；重叠窗口不是独立行人。K20相对CV的收益不是社会模块独立贡献，也不能替代确定性K1结论。VAE训练目标不称精确边缘NLL，latent-prior轨迹不称预测数学期望。

验收：完整单元测试见 `results/tables/stage5p2/final_tests.xml` 与 `final_tests.txt`；旧2569文件SHA和200份新checkpoint SHA复核见 `delivery_verification.json`、`checkpoint_sha_verification.csv`。保护结果以这些真实运行记录为准，不覆盖旧权重/指标。权重/原始数据/缓存不进入Git，只完成本地commit，不自动push。

用户手动推送：`git push -u origin feat/stage5p2-socialvae-baseline`。本阶段完成后停止，等待下一步科研决策。

## 最终验收补充（真实数据）

主指标按目标窗口加权：ZARA2 K1相对Last2改善，HOTEL退化；K20两录像均优于三种CV，但这是20样本GT最优口径。**按行人等权时HOTEL的K20为0.4212/0.8036m，仍差于OLS4的0.3546/0.6914m和OLS8的0.3816/0.7696m**。不能将窗口加权下的结论推广至所有权重或所有行人。

两个录像后期训练误差下降而验证恶化，存在明显过拟合/跨录像泛化退化证据；best位于30/3轮，未由100轮边界截断。训练目标仍下降，不等于数值收敛，但没有证据支持本阶段自动延长。SocialVAE可暂作为多模态开发基线，尚不支持独立社会收益或第四章创新成立；下一步仅建议预注册完整NoSocial，当前不执行。

完整单元测试302项通过（`final_tests.xml`）。额外固定权重CUDA诊断8项中：同batch置换K1/K20的4项预测最大差异均为0；跨batch拆分4项均未通过预设atol/rtol=1e-5，失败标记保留，未放宽阈值。ZARA2 K1/K20最大坐标差为0.000204086/0.001112938m，HOTEL为0.000275135/0.000776291m。标准高斯epsilon按目标ID完全一致；这说明控制随机数并不消除浮点批处理与padding布局带来的数值差异。官方网络/epsilon/主评价batch=128均未修改，没有重训或重选checkpoint。原始失败现场与完整诊断见`CUDA_initial_tolerance_failure.txt`、`CUDA_sampling_invariance_diagnostic.json`。这些只是各录像16个固定索引的功能诊断，不是全验证集最大误差上界，也不证明差异完全由某一GPU算子造成。
