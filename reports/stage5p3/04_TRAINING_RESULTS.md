# 正式训练与冻结Full对照

两个NoSocial run各自seed17从头训练100完整epoch，未加载Full训练权重，未调参、未追加seed/epoch。Full直接引用Stage5.2冻结正式结果，两个模型均按验证K1-prior ADE最低选择best、FDE同checkpoint。原始数据、训练池、目标ID、GT SHA、评价函数、目标噪声和K20预算相同，核查见`cohort_verification.json`。这仍是历史曝光开发录像，非独立确认；ETH未评价。

| 录像 | 模型 | 选择 | epoch | 口径 | 窗口ADE | 窗口FDE | 行人等权ADE | 行人等权FDE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HOTEL | CV-Last2 | 解析 | — | 确定性 | 0.3295 | 0.6110 | 0.4580 | 0.8553 |
| HOTEL | CV-OLS4 | 解析 | — | 确定性 | 0.2378 | 0.4389 | 0.3546 | 0.6914 |
| HOTEL | CV-OLS8 | 解析 | — | 确定性 | 0.2201 | 0.4088 | 0.3816 | 0.7696 |
| HOTEL | Full | best | 3 | K1-prior | 0.3872 | 0.7515 | 0.7361 | 1.4786 |
| HOTEL | Full | best | 3 | K1-first-sample | 0.5303 | 1.0754 | 0.9754 | 2.0391 |
| HOTEL | Full | best | 3 | K20-independent-minima | 0.2025 | 0.3253 | 0.4212 | 0.8036 |
| HOTEL | Full | last | 100 | K1-prior | 0.6470 | 1.4422 | 1.2578 | 2.9163 |
| HOTEL | Full | last | 100 | K1-first-sample | 0.6720 | 1.4965 | 1.2879 | 2.9584 |
| HOTEL | Full | last | 100 | K20-independent-minima | 0.4703 | 1.0387 | 0.9399 | 2.2430 |
| HOTEL | NoSocial | best | 2 | K1-prior | 0.3952 | 0.7778 | 0.8168 | 1.7250 |
| HOTEL | NoSocial | best | 2 | K1-first-sample | 0.5734 | 1.1535 | 0.9504 | 1.9647 |
| HOTEL | NoSocial | best | 2 | K20-independent-minima | 0.2461 | 0.4166 | 0.5098 | 1.0500 |
| HOTEL | NoSocial | last | 100 | K1-prior | 0.7054 | 1.5088 | 1.3135 | 3.0191 |
| HOTEL | NoSocial | last | 100 | K1-first-sample | 0.7892 | 1.6777 | 1.5402 | 3.4019 |
| HOTEL | NoSocial | last | 100 | K20-independent-minima | 0.3470 | 0.6433 | 0.6169 | 1.3026 |
| ZARA2 | CV-Last2 | 解析 | — | 确定性 | 0.2963 | 0.6675 | 0.4191 | 0.8453 |
| ZARA2 | CV-OLS4 | 解析 | — | 确定性 | 0.3319 | 0.7157 | 0.4728 | 0.9087 |
| ZARA2 | CV-OLS8 | 解析 | — | 确定性 | 0.3832 | 0.7872 | 0.5242 | 0.9607 |
| ZARA2 | Full | best | 30 | K1-prior | 0.2726 | 0.6075 | 0.3598 | 0.7009 |
| ZARA2 | Full | best | 30 | K1-first-sample | 0.4099 | 0.9473 | 0.5506 | 1.1923 |
| ZARA2 | Full | best | 30 | K20-independent-minima | 0.1575 | 0.3048 | 0.1871 | 0.3097 |
| ZARA2 | Full | last | 100 | K1-prior | 0.3290 | 0.7398 | 0.4606 | 0.9415 |
| ZARA2 | Full | last | 100 | K1-first-sample | 0.3459 | 0.7790 | 0.4810 | 0.9871 |
| ZARA2 | Full | last | 100 | K20-independent-minima | 0.2506 | 0.5458 | 0.3320 | 0.6329 |
| ZARA2 | NoSocial | best | 55 | K1-prior | 0.2789 | 0.6134 | 0.3727 | 0.7345 |
| ZARA2 | NoSocial | best | 55 | K1-first-sample | 0.4013 | 0.9016 | 0.5487 | 1.1732 |
| ZARA2 | NoSocial | best | 55 | K20-independent-minima | 0.1551 | 0.2942 | 0.1912 | 0.3084 |
| ZARA2 | NoSocial | last | 100 | K1-prior | 0.2955 | 0.6564 | 0.4106 | 0.8249 |
| ZARA2 | NoSocial | last | 100 | K1-first-sample | 0.3833 | 0.8680 | 0.5475 | 1.1711 |
| ZARA2 | NoSocial | last | 100 | K20-independent-minima | 0.1552 | 0.3051 | 0.2030 | 0.3334 |


K1-prior不是分布数学期望；首样本为同一K20中的候选0；minADE/minFDE分别选最优候选，不保证候选相同。全部模型FPC关闭。CV三者保留，Last2为固定主参考，不按录像挑弱对手。Ego-Capacity-Control未实现/未训练，不填写性能。原始loss为重建+MC KL，非精确边缘NLL。

## NoSocial生命周期容量实测

| 录像 | 名义 | 曾连接loss图的张量参数 | 全程曾非零梯度标量 | 全程从未非零标量 | 结构有效上界 |
| --- | --- | --- | --- | --- | --- |
| HOTEL | 2144002 | 1656514 | 1455838 | 688164 | 1459906 |
| ZARA2 | 2144002 | 1656514 | 1456224 | 687778 | 1459906 |


梯度并集记录了该run全部optimizer步骤；它依赖有限训练数据和数值精度，不等于函数类维数。结构死参数逐轮保持原始值。Full只有旧Adam二阶矩代理，不伪造其逐步梯度历史。

## 曲线与泛化

### HOTEL Full

参数名义2144002；best第3轮，last第100轮。实际墙钟2338.62s（38.98分钟），峰值allocated 1.848GiB。训练源码提交`eb8d494bb1373462fcd07da46abeb5e7bb2533f7`。

第1轮训练loss 0.424568 → last 0.008494（rec 0.005210 + MC KL 0.003285）；best验证posterior loss 0.034527 → last 0.068391。best固定权重完整训练K1 ADE 0.5135 → last训练 0.1146m；验证 0.3872 → 0.6470m。

| 末20轮指标 | 81–90均值 | 91–100均值 | 后段−前段 |
| --- | --- | --- | --- |
| K1_prior_ADE_m | 0.692395 | 0.676345 | -0.016050 |
| K1_prior_FDE_m | 1.538324 | 1.506598 | -0.031726 |
| minADE20_m | 0.476105 | 0.481433 | 0.005328 |
| minFDE20_m | 1.034594 | 1.044719 | 0.010125 |
| posterior_loss | 0.057899 | 0.062747 | 0.004848 |
| train_kl | 0.004569 | 0.003611 | -0.000958 |
| train_loss | 0.010341 | 0.008956 | -0.001385 |
| train_rec | 0.005771 | 0.005345 | -0.000426 |

### HOTEL NoSocial

参数名义2144002；best第2轮，last第100轮。实际墙钟2019.67s（33.66分钟），峰值allocated 0.113GiB。训练源码提交`52de9edc344b1603ba7cb2a7944239b678e1c0ee`。

第1轮训练loss 0.425328 → last 0.017819（rec 0.006696 + MC KL 0.011123）；best验证posterior loss 0.044610 → last 0.034233。best固定权重完整训练K1 ADE 0.5091 → last训练 0.3982m；验证 0.3952 → 0.7054m。

| 末20轮指标 | 81–90均值 | 91–100均值 | 后段−前段 |
| --- | --- | --- | --- |
| K1_prior_ADE_m | 0.710518 | 0.681958 | -0.028561 |
| K1_prior_FDE_m | 1.545046 | 1.472731 | -0.072315 |
| minADE20_m | 0.313232 | 0.326297 | 0.013065 |
| minFDE20_m | 0.563429 | 0.590851 | 0.027422 |
| posterior_loss | 0.028732 | 0.030521 | 0.001789 |
| train_kl | 0.011628 | 0.011299 | -0.000328 |
| train_loss | 0.018388 | 0.017941 | -0.000446 |
| train_rec | 0.006760 | 0.006642 | -0.000118 |

### ZARA2 Full

参数名义2144002；best第30轮，last第100轮。实际墙钟2087.09s（34.78分钟），峰值allocated 1.847GiB。训练源码提交`eb8d494bb1373462fcd07da46abeb5e7bb2533f7`。

第1轮训练loss 0.419614 → last 0.008465（rec 0.005351 + MC KL 0.003113）；best验证posterior loss 0.015872 → last 0.050282。best固定权重完整训练K1 ADE 0.4559 → last训练 0.1156m；验证 0.2726 → 0.3290m。

| 末20轮指标 | 81–90均值 | 91–100均值 | 后段−前段 |
| --- | --- | --- | --- |
| K1_prior_ADE_m | 0.327556 | 0.329114 | 0.001558 |
| K1_prior_FDE_m | 0.740411 | 0.739679 | -0.000732 |
| minADE20_m | 0.231111 | 0.248875 | 0.017764 |
| minFDE20_m | 0.500443 | 0.539924 | 0.039481 |
| posterior_loss | 0.036882 | 0.047482 | 0.010600 |
| train_kl | 0.004378 | 0.003444 | -0.000933 |
| train_loss | 0.010296 | 0.008985 | -0.001311 |
| train_rec | 0.005918 | 0.005541 | -0.000377 |

### ZARA2 NoSocial

参数名义2144002；best第55轮，last第100轮。实际墙钟1814.60s（30.24分钟），峰值allocated 0.113GiB。训练源码提交`52de9edc344b1603ba7cb2a7944239b678e1c0ee`。

第1轮训练loss 0.448721 → last 0.018663（rec 0.006908 + MC KL 0.011755）；best验证posterior loss 0.013876 → last 0.013821。best固定权重完整训练K1 ADE 0.4636 → last训练 0.4115m；验证 0.2789 → 0.2955m。

| 末20轮指标 | 81–90均值 | 91–100均值 | 后段−前段 |
| --- | --- | --- | --- |
| K1_prior_ADE_m | 0.291415 | 0.298882 | 0.007467 |
| K1_prior_FDE_m | 0.647247 | 0.664837 | 0.017590 |
| minADE20_m | 0.152621 | 0.154714 | 0.002093 |
| minFDE20_m | 0.290458 | 0.298021 | 0.007563 |
| posterior_loss | 0.013316 | 0.013484 | 0.000168 |
| train_kl | 0.012278 | 0.011933 | -0.000345 |
| train_loss | 0.019422 | 0.018990 | -0.000432 |
| train_rec | 0.007143 | 0.007056 | -0.000087 |


以上耗时为实际执行，NoSocial额外逐步梯度审计而Full没有该仪器开销，不能据两者墙钟差直接宣称架构效率优势。训练loss是更新中batch加权平均，验证posterior是固定权重、固定目标噪声诊断；不可称精确边缘似然比较。

100轮并不自动证明收敛。最优点、末20轮和同一checkpoint训练/验证差距必须一起看；任何末段改善都不触发额外训练。完整每轮/每batch曲线、逐目标误差、定期模型/optimizer/RNG严格恢复、checkpoint SHA和执行日志已保存。所有NoSocial模型在`experiments/stage5p3/checkpoints/`；best/last由`selection.json`给出，不覆盖旧权重。
