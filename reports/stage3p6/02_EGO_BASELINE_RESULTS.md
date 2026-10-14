# 个体运动基线与训练稳定性

全部为协议A；两个历史曝光录像、seed17；best按整体ADE、FDE配对同checkpoint，单位m。各神经run独立从头30epoch，不删后期退化。

| 录像 | 模型 | schedule | best轮 | best ADE/FDE | last30 ADE/FDE | 参数 | 训练秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crowds_zara02 | CV | 解析 | — | 0.296306/0.667523 | 不训练 | 0 | — |
| crowds_zara02 | PedBase | constant | 29 | 0.282905/0.630561 | 0.298853/0.659355 | 35352 | 193.31 |
| crowds_zara02 | PedBase | step10_20 | 22 | 0.276696/0.619606 | 0.284282/0.631653 | 35352 | 194.15 |
| crowds_zara02 | CV-Residual | constant | 24 | 0.276902/0.617161 | 0.285986/0.639910 | 35352 | 210.81 |
| crowds_zara02 | CV-Residual | step10_20 | 8 | 0.277524/0.622087 | 0.278843/0.622135 | 35352 | 208.31 |
| biwi_hotel | CV | 解析 | — | 0.329516/0.611038 | 不训练 | 0 | — |
| biwi_hotel | PedBase | constant | 24 | 0.503319/1.005511 | 0.526877/1.052827 | 35352 | 242.11 |
| biwi_hotel | PedBase | step10_20 | 27 | 0.516671/1.017872 | 0.521228/1.026156 | 35352 | 238.72 |
| biwi_hotel | CV-Residual | constant | 1 | 0.401317/0.792782 | 0.742461/1.547134 | 35352 | 265.10 |
| biwi_hotel | CV-Residual | step10_20 | 1 | 0.401317/0.792782 | 0.594714/1.236998 | 35352 | 263.12 |

## 固定LR对照与曲线

| 录像/模型/策略 | best轮≤5 | 首/末loss | 尾5loss下降 | last−best ADE | 尾5val ADE std |
| --- | --- | --- | --- | --- | --- |
| crowds_zara02/PedBase/constant | False | 0.733953/0.469732 | 0.005990 | 0.015948 | 0.012520 |
| crowds_zara02/PedBase/step10_20 | False | 0.733953/0.466854 | 0.000624 | 0.007586 | 0.003342 |
| crowds_zara02/CV-Residual/constant | False | 0.492932/0.435336 | 0.019419 | 0.009084 | 0.001254 |
| crowds_zara02/CV-Residual/step10_20 | False | 0.492932/0.456611 | 0.001935 | 0.001319 | 0.000299 |
| biwi_hotel/PedBase/constant | False | 0.681032/0.444407 | 0.009748 | 0.023558 | 0.016723 |
| biwi_hotel/PedBase/step10_20 | False | 0.681032/0.442807 | 0.000727 | 0.004558 | 0.005846 |
| biwi_hotel/CV-Residual/constant | True | 0.467518/0.408774 | 0.021791 | 0.341144 | 0.022088 |
| biwi_hotel/CV-Residual/step10_20 | True | 0.467518/0.432548 | 0.001748 | 0.193397 | 0.010266 |

8个ego神经run中2个best落在1–5轮，不能把旧students003早期最优推广到所有录像。训练loss下降与val变化是不同分布下的事实；旧的1024窗口结论不能直接替代本轮更多窗口结果。fixed_schedule_comparison.csv保存预先规定衰减相对constant的best/last/尾部波动差。不同方向如实保留，单seed不能断言唯一原因是LR过高；30epoch末尾loss趋缓也不等于已全局收敛。没有延长训练。

## 历史运动分布

| 留出录像 | split | 历史量 | mean | median | P95 |
| --- | --- | --- | --- | --- | --- |
| crowds_zara02 | train | cutoff_speed_mps | 0.566785 | 0.548945 | 1.280893 |
| crowds_zara02 | train | cutoff_velocity_x_mps | -0.064164 | -0.014206 | 1.059587 |
| crowds_zara02 | train | cutoff_velocity_y_mps | 0.033265 | 0.010142 | 0.625233 |
| crowds_zara02 | train | history_velocity_change_mps | 0.068486 | 0.058022 | 0.180366 |
| crowds_zara02 | train | mean_neighbor_degree | 6.804751 | 8.000000 | 8.000000 |
| crowds_zara02 | train | relative_velocity_change_mps | 0.118319 | 0.107531 | 0.236591 |
| crowds_zara02 | validation | cutoff_speed_mps | 0.440481 | 0.021479 | 1.360520 |
| crowds_zara02 | validation | cutoff_velocity_x_mps | -0.006866 | -0.001053 | 1.222512 |
| crowds_zara02 | validation | cutoff_velocity_y_mps | 0.002097 | 0.000000 | 0.325323 |
| crowds_zara02 | validation | history_velocity_change_mps | 0.035866 | 0.000795 | 0.142136 |
| crowds_zara02 | validation | mean_neighbor_degree | 4.434982 | 5.000000 | 8.000000 |
| crowds_zara02 | validation | relative_velocity_change_mps | 0.067647 | 0.051339 | 0.193718 |
| biwi_hotel | train | cutoff_speed_mps | 0.568428 | 0.549843 | 1.301809 |
| biwi_hotel | train | cutoff_velocity_x_mps | -0.053932 | -0.007893 | 1.123357 |
| biwi_hotel | train | cutoff_velocity_y_mps | 0.026840 | 0.004773 | 0.558941 |
| biwi_hotel | train | history_velocity_change_mps | 0.062187 | 0.052989 | 0.167790 |
| biwi_hotel | train | mean_neighbor_degree | 6.621172 | 8.000000 | 8.000000 |
| biwi_hotel | train | relative_velocity_change_mps | 0.107998 | 0.100171 | 0.218179 |
| biwi_hotel | validation | cutoff_speed_mps | 0.407574 | 0.079057 | 1.519363 |
| biwi_hotel | validation | cutoff_velocity_x_mps | -0.021934 | 0.000000 | 0.175000 |
| biwi_hotel | validation | cutoff_velocity_y_mps | -0.019261 | 0.000000 | 1.253750 |
| biwi_hotel | validation | history_velocity_change_mps | 0.144853 | 0.128040 | 0.378070 |
| biwi_hotel | validation | mean_neighbor_degree | 2.923630 | 2.857143 | 7.714286 |
| biwi_hotel | validation | relative_velocity_change_mps | 0.249477 | 0.252802 | 0.468700 |

速度/方向/变化量及密度全部来自历史。训练与验证的中位速度、方向轴与密度不一致，可能解释泛化/偏差（待验证），不是凭这些相关量认定因果。逐目标同样本CV/PedBase/CVR比较在paired_target_errors.csv，两个ego参数均35352/同初始化，排除了总参数增加这一因素；CV主路径同时改变优化残差分布，不能说仅靠参数量得益。

## 端点训练/验证泛化诊断

补充训练结束后的同checkpoint全训练/验证推理，排除在线minibatch loss与epoch末模型不一致的比较方式；仅诊断、没有优化或模型选择，CPU验证与主GPU指标差异<1e-6。

| 录像/模型/策略/endpoint | split | 目标 | ADE/FDE(m) | ADE−同split CV |
| --- | --- | --- | --- | --- |
| crowds_zara02/CV/analytical/analytical | train | 23545 | 0.508975/1.128326 | — |
| crowds_zara02/CV/analytical/analytical | validation | 1092 | 0.296306/0.667523 | — |
| crowds_zara02/PedBase/constant/best | train | 23545 | 0.464847/1.008460 | -0.044128 |
| crowds_zara02/PedBase/constant/best | validation | 1092 | 0.282905/0.630561 | -0.013401 |
| crowds_zara02/PedBase/constant/last | train | 23545 | 0.467905/1.012134 | -0.041070 |
| crowds_zara02/PedBase/constant/last | validation | 1092 | 0.298853/0.659355 | 0.002547 |
| crowds_zara02/PedBase/step10_20/best | train | 23545 | 0.466689/1.014583 | -0.042286 |
| crowds_zara02/PedBase/step10_20/best | validation | 1092 | 0.276696/0.619606 | -0.019610 |
| crowds_zara02/PedBase/step10_20/last | train | 23545 | 0.468117/1.016656 | -0.040858 |
| crowds_zara02/PedBase/step10_20/last | validation | 1092 | 0.284282/0.631653 | -0.012024 |
| crowds_zara02/CV-Residual/constant/best | train | 23545 | 0.444248/0.965826 | -0.064727 |
| crowds_zara02/CV-Residual/constant/best | validation | 1092 | 0.276902/0.617161 | -0.019404 |
| crowds_zara02/CV-Residual/constant/last | train | 23545 | 0.431426/0.938371 | -0.077550 |
| crowds_zara02/CV-Residual/constant/last | validation | 1092 | 0.285986/0.639910 | -0.010320 |
| crowds_zara02/CV-Residual/step10_20/best | train | 23545 | 0.471223/1.025773 | -0.037752 |
| crowds_zara02/CV-Residual/step10_20/best | validation | 1092 | 0.277524/0.622087 | -0.018782 |
| crowds_zara02/CV-Residual/step10_20/last | train | 23545 | 0.456457/0.994252 | -0.052519 |
| crowds_zara02/CV-Residual/step10_20/last | validation | 1092 | 0.278843/0.622135 | -0.017463 |
| biwi_hotel/CV/analytical/analytical | train | 27145 | 0.485746/1.080470 | — |
| biwi_hotel/CV/analytical/analytical | validation | 318 | 0.329516/0.611038 | — |
| biwi_hotel/PedBase/constant/best | train | 27145 | 0.447608/0.968497 | -0.038137 |
| biwi_hotel/PedBase/constant/best | validation | 318 | 0.503319/1.005511 | 0.173803 |
| biwi_hotel/PedBase/constant/last | train | 27145 | 0.443112/0.959742 | -0.042634 |
| biwi_hotel/PedBase/constant/last | validation | 318 | 0.526877/1.052827 | 0.197361 |
| biwi_hotel/PedBase/step10_20/best | train | 27145 | 0.442612/0.966054 | -0.043133 |
| biwi_hotel/PedBase/step10_20/best | validation | 318 | 0.516671/1.017872 | 0.187155 |
| biwi_hotel/PedBase/step10_20/last | train | 27145 | 0.442521/0.966446 | -0.043225 |
| biwi_hotel/PedBase/step10_20/last | validation | 318 | 0.521228/1.026156 | 0.191712 |
| biwi_hotel/CV-Residual/constant/best | train | 27145 | 0.456808/0.997905 | -0.028938 |
| biwi_hotel/CV-Residual/constant/best | validation | 318 | 0.401317/0.792782 | 0.071802 |
| biwi_hotel/CV-Residual/constant/last | train | 27145 | 0.401987/0.878906 | -0.083759 |
| biwi_hotel/CV-Residual/constant/last | validation | 318 | 0.742461/1.547134 | 0.412945 |
| biwi_hotel/CV-Residual/step10_20/best | train | 27145 | 0.456808/0.997905 | -0.028938 |
| biwi_hotel/CV-Residual/step10_20/best | validation | 318 | 0.401317/0.792782 | 0.071802 |
| biwi_hotel/CV-Residual/step10_20/last | train | 27145 | 0.432063/0.946032 | -0.053683 |
| biwi_hotel/CV-Residual/step10_20/last | validation | 318 | 0.594714/1.236998 | 0.265198 |

训练与验证改善方向相反只能支持分布/泛化风险，不能确定某一个速度轴、LR或数据问题为唯一原因。

## 冻结选择

| 录像 | 输出/编码器 | 神经schedule | anchor ADE/FDE | 源checkpoint SHA256 |
| --- | --- | --- | --- | --- |
| crowds_zara02 | PedBase | step10_20 | 0.276696/0.619606 | e7f936cac808e4d69559fd63473fc8752a7fcd443c74f0b9f3ad3d1b05221258 |
| biwi_hotel | CV输出；CV-Residual编码 | constant | 0.329516/0.611038 | 8a7ad9bf6986191b8c96846f4b99153a2b83c34c3772d14abe9cc07cf2c2d74e |

若选择CV输出，最佳神经encoder/head仍原样冻结，decoder不参与最终基础CV输出；总冻结参数35352、有效输出非学习CV。所有分支共享同一源文件/数值权重/输出规则。多候选anchor使用开发集选择，非独立验证，未评价ETH。

## 实际泛化现象

HOTEL的CV训练/验证ADE为0.485746/0.329516m。CV-Residual constant的best1训练/验证为0.456808/0.401317，last30为0.401987/0.742461：训练继续改善，验证显著恶化，而且模型没有社会模块。ZARA2同模型best24训练/验证为0.444248/0.276902，均优于该split CV0.508975/0.296306；因此不能把CV残差优劣推广至所有录像。

衰减降低四个对照的尾5轮验证波动，但只改善ZARA2 PedBase的best ADE；HOTEL PedBase best反而更差，CVR best不变。不能把所有泛化问题归因于高LR。8个ego run只有2个best在前5轮，两者都是HOTEL CVR；其余最优在8–30轮。
