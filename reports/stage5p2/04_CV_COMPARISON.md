# 完全配对CV比较

所有三种CV均重新运行；两个录像合计1410目标窗口、63个录像-行人ID单位。目标ID、历史坐标SHA和未来GT SHA与Stage5.1逐项一致，三种CV既有逐目标误差也逐项复核。未借用旧聚合数值代替重算。

| 录像 | 模型/口径 | 目标窗口 | 行人ID | 窗口ADE | 窗口FDE | 行人等权ADE | 行人等权FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HOTEL | CV-Last2 | 318 | 28 | 0.3295 | 0.6110 | 0.4580 | 0.8553 |
| HOTEL | CV-OLS4 | 318 | 28 | 0.2378 | 0.4389 | 0.3546 | 0.6914 |
| HOTEL | CV-OLS8 | 318 | 28 | 0.2201 | 0.4088 | 0.3816 | 0.7696 |
| HOTEL | SocialVAE-best K1-prior | 318 | 28 | 0.3872 | 0.7515 | 0.7361 | 1.4786 |
| HOTEL | SocialVAE-best K1-first-sample | 318 | 28 | 0.5303 | 1.0754 | 0.9754 | 2.0391 |
| HOTEL | SocialVAE-best K20-independent-minima | 318 | 28 | 0.2025 | 0.3253 | 0.4212 | 0.8036 |
| ZARA2 | CV-Last2 | 1092 | 35 | 0.2963 | 0.6675 | 0.4191 | 0.8453 |
| ZARA2 | CV-OLS4 | 1092 | 35 | 0.3319 | 0.7157 | 0.4728 | 0.9087 |
| ZARA2 | CV-OLS8 | 1092 | 35 | 0.3832 | 0.7872 | 0.5242 | 0.9607 |
| ZARA2 | SocialVAE-best K1-prior | 1092 | 35 | 0.2726 | 0.6075 | 0.3598 | 0.7009 |
| ZARA2 | SocialVAE-best K1-first-sample | 1092 | 35 | 0.4099 | 0.9473 | 0.5506 | 1.1923 |
| ZARA2 | SocialVAE-best K20-independent-minima | 1092 | 35 | 0.1575 | 0.3048 | 0.1871 | 0.3097 |


主对照固定CV-Last2，不能按录像更换主基线。OLS4/8通过历史位置与物理秒OLS估计速度，预测均锚定最后历史位置。各录像最低解析误差：

| 录像 | ADE最优CV | 最低ADE | FDE最优CV | 最低FDE |
| --- | --- | --- | --- | --- |
| HOTEL | CV-OLS8 | 0.2201 | CV-OLS8 | 0.4088 |
| ZARA2 | CV-Last2 | 0.2963 | CV-Last2 | 0.6675 |


上表只是**解析基线上界参考**（事后逐录像选择最强解析基线），不是一个可部署的新模型。更低误差在指标意义上是误差下界；“上界参考”指该有限CV集合可达到的性能参考，不能作统计理论界限。

## 两录像宏平均（各录像等权）

| 模型 | 评价口径 | 录像内权重 | 宏ADE | 宏FDE |
| --- | --- | --- | --- | --- |
| CV-Last2 | deterministic | target-window | 0.3129 | 0.6393 |
| CV-Last2 | deterministic | equal-person-ID | 0.4386 | 0.8503 |
| CV-OLS4 | deterministic | target-window | 0.2848 | 0.5773 |
| CV-OLS4 | deterministic | equal-person-ID | 0.4137 | 0.8001 |
| CV-OLS8 | deterministic | target-window | 0.3017 | 0.5980 |
| CV-OLS8 | deterministic | equal-person-ID | 0.4529 | 0.8652 |
| SocialVAE | K1-first-sample | target-window | 0.4701 | 1.0114 |
| SocialVAE | K1-first-sample | equal-person-ID | 0.7630 | 1.6157 |
| SocialVAE | K1-prior | target-window | 0.3299 | 0.6795 |
| SocialVAE | K1-prior | equal-person-ID | 0.5479 | 1.0897 |
| SocialVAE | K20-independent-minima | target-window | 0.1800 | 0.3151 |
| SocialVAE | K20-independent-minima | equal-person-ID | 0.3041 | 0.5567 |


窗口加权先在录像内按目标窗口平均；行人等权先在每个录像-行人ID内部平均，再平均该录像的行人。宏平均最后对两个录像各赋1/2权重。不能把按所有1410窗口直接合并的数值称为录像宏平均。

K1 latent-prior是预先指定的确定性评价；K1首样本是固定seed的第一条随机轨迹；K20利用20条候选和未来GT选择最优，是不同预测口径。K20优于CV不能推出单轨迹预测优于CV，也不能证明社会消息的独立贡献。

完整配对文件：`paired_model_errors.csv.gz`；完整汇总 `model_metrics.csv`、`person_aggregated_errors.csv`、`recording_macro_metrics.csv`；强CV参考 `analytic_oracle_reference.json`。

## 最终验收补充（真实数据）

窗口加权：HOTEL K1相对Last2的ADE退化0.0577m，相对OLS8退化0.1670m；ZARA2 K1相对Last2改善0.0237m。K20窗口加权HOTEL相对OLS8改善0.0177m，ZARA2相对Last2改善0.1388m。

行人等权：HOTEL K1为0.7361/1.4786m，差于全部三个CV；HOTEL K20为0.4212/0.8036m，优于Last2但弱于OLS4、OLS8。说明该录像较长轨迹的窗口权重显著影响总体结论。ZARA2 K1行人等权为0.3598/0.7009m，也优于Last2的0.4191/0.8453m；K20为0.1871/0.3097m。

两个录像窗口口径宏K1为0.3299/0.6795m，弱于Last2宏0.3129/0.6393m；宏K20为0.1800/0.3151m。宏平均不抹去HOTEL K1退化和行人等权强CV反例。
