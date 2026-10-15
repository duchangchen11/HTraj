# Stage 5.1：SocialVAE严格因果数据与训练入口总报告

**HTraj-SocialVAE-Causal8x12-v1已通过原始数据隔离、覆盖率和5轮工程训练检查，具备下一阶段正式基线训练的工程条件。当前没有性能优势或社会交互独立贡献的结论。**

项目`/home/lrj/HTraj`，分支`feat/stage5p1-socialvae-causal`，基础`d25da2fa743034338c337bee5d726885b91a2ef1`。官方SocialVAE固定`eb23cbae562bac2c2dcf0d81529e6ab2ac280962`。2,463项Stage1至Stage5源码、结果、配置、权重等文件已建立SHA保护清单；最终零变化核查见`delivery_checks.json`。原始网络与loss未修改，新loader/入口独立管理。

## 已完成与真实结果

- 建立A `SocialVAE-Official-Offline`和B `HTraj-SocialVAE-Causal8x12-v1`两套独立manifest。A仅保留原样参照，明确不宣称因果；B全部导数限当前8帧，邻居ID仅来自历史。
- 缺连续两点的邻居运动packet按有效性屏蔽；首帧只允许原模型的纯位置初始化。原始有效性以bitmask留档，不把unknown当测得静止。目标首项加速度采用明确的0边界先验，valid=False，不使用窗口前隐藏数据。
- 36个从原始文本重新运行loader的反事实全部通过，输入及预测最大差0。Stage5反例在A下仍变化0.487452656米，B下为0。padding不同batch最大绝对差1.19e-7米。
- 两个录像隔离划分固定，训练保留原train区域、没有旧validation回流。HOTEL验证318目标/94有监督窗口，ZARA2验证1092目标/172窗口。A/B监督目标一致，输入协议不混用。
- batch16/32/64/128均真实完成前向、loss、反向、更新和model/optimizer/RNG严格恢复。batch128计时计算约0.102秒/步，加载约0.019秒/步，约1055目标/秒端到端；峰值已分配显存约1890MiB。
- ZARA2单seed17、随机初始化、5个完整epoch完成，共920更新、117725次目标曝光，实际117.193秒。loss由0.419614降至0.035121，所有梯度/预测有限，所有checkpoint严格恢复一致。没有加载官方权重初始化训练，没有更换LR或无限延长。

## 相同目标的主指标

|验证录像|对象|目标数|K1 ADE|配对FDE|
|---|---|---:|---:|---:|
|HOTEL|CV-Last2|318|0.329516|0.611038|
|HOTEL|CV-OLS4|318|0.237814|0.438866|
|HOTEL|CV-OLS8|318|0.220144|0.408760|
|HOTEL|新SocialVAE|—|未运行|未运行|
|ZARA2|CV-Last2|1092|0.296306|0.667523|
|ZARA2|CV-OLS4|1092|0.331875|0.715683|
|ZARA2|CV-OLS8|1092|0.383175|0.787222|
|ZARA2|SocialVAE smoke epoch5|1092|0.307006|0.674250|

单位米。CV全部本轮重算，逐目标窗口/ID/未来hash一致，没有引用旧表代替。神经best由最低验证K1 prior ADE选择，best=last=5；对应首样本ADE/FDE为0.451537/0.985597，minADE/minFDE@20为0.189208/0.346880，FPC关闭。K1 latent-prior均值不是轨迹分布数学期望；K20与CV K1不能直接当同预算比较。

当前神经K1比最强ZARA2 CV-Last2差0.010700 ADE、0.006727 FDE，不能声称成熟性能已成立。训练少，仍无法判断充分收敛；没有社会Full/NoSocial训练，因此没有独立社会收益结论。只有5轮工程结果，不能转成论文性能表。

## 资源判断及停止

基于实际batch128和五个完整epoch，ZARA2 30完整epoch估计约11.72分钟，100轮约39.06分钟；对齐原配置常见6万/8万更新量级，当前非ETH协议估计约2.15/2.87小时。不是小batch线性推算，也不是原论文全数据复现承诺；上下文/硬件负载/验证策略变化需重新测量。全部估计均标记未运行。

本阶段不进行正式基线长训练、额外seed、Full/NoSocial、新预测模块、FPC或ETH评价。建议下一阶段先冻结正式训练预算及终点，再验证成熟基线，不先宣布创新。首次启动的0更新配置字段错误和一次结果JSON编码错误均保留日志；修订没有改模型、超参数或真实指标。

## 文件与Git

七份报告：

- `reports/stage5p1/01_CAUSAL_DATA_PROTOCOL.md`
- `reports/stage5p1/02_RAW_INPUT_LEAKAGE_AUDIT.md`
- `reports/stage5p1/03_DATASET_AND_SPLIT_AUDIT.md`
- `reports/stage5p1/04_SOCIALVAE_TRAINING_SMOKE.md`
- `reports/stage5p1/05_CV_BASELINE_RESULTS.md`
- `reports/stage5p1/06_RESEARCH_DECISION.md`
- `reports/stage5p1/STAGE5P1_REPORT.md`

配置/数据manifest位于`configs/stage5p1/`；原始与源码SHA、全量压缩目标索引、可观测性、36个反事实实际预测、资源每步日志、5轮曲线/逐目标误差、CV配对、九个新checkpoint hash及完整测试在`results/tables/stage5p1/`。预训练前293项完整测试通过，最终完整测试另存`unit_tests.xml`；权重、数据和缓存不入Git。源代码版本及保护核查结果在`smoke_config.json`与`delivery_checks.json`。

最终本地commit以`git log -1`为准，不自动推送。用户自行：

```bash
git push -u origin feat/stage5p1-socialvae-causal
```

完成本地交付后停止。
