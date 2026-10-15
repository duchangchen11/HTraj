# Stage3.7：泛化、坐标敏感性与完整历史运动分组

## 本轮主要事实与边界

- **方向表示是HOTEL中的可操作因素，不是统一修复。** M2相对M0的最佳ADE减少0.167667m（约33.31%），ZARA2则增加0.005021m。M3相对M0在HOTEL减少0.165218m、在ZARA2增加0.000138m。后一差异很小，不能声称统计显著退化，但两个版本均未满足双录像改善门槛。
- **最佳与固定预算终点必须区分。** ZARA2的M2/M3在last30的ADE分别比M0低0.003081/0.007614m，但各自best并未超过M0。HOTEL的M2/M3均best1，随后训练集同endpoint ADE继续下降，验证ADE却分别升至0.416973/0.491875m。这不能用单一“没有收敛”解释。
- **低速与高速不是同向收益。** ZARA2低速648目标的M2/M3分别改善0.008282/0.008782m，高速309目标分别退化0.036842/0.028696m。HOTEL高速79目标虽大幅改善M0，M2/M3仍分别比CV差0.137419/0.042015m；中速只有15目标且均比M0差。HOTEL速度变化明显组只有3个目标窗口，不能据此建立普遍结论。低速组并未应用旋转，其变化来自共享权重、训练过程及M3先验，不能归为该组直接旋转收益。
- **固定权重也体现了轴方向敏感性。** HOTEL的M0在预设90°同步输入/指标旋转下，ADE从0.503319变为0.327654m；其可靠朝向组平均预测等变偏差0.763283m。没有训练更新，因此不只是比较两次训练的优化快慢。但这是几何干预，不能据此把90°当作新配置、以干预后的数值覆盖原M0，或宣称唯一因果已被识别。
- **低速回退具有实测限制。** M2/M3可靠朝向组在三个旋转下均得到零数值等变偏差；HOTEL的M2低速组90°平均偏差仍为0.129773m。因此不能宣称整个网络严格旋转等变。固定权重阈值0.1/0.4m/s只改变HOTEL的5/6个目标、ZARA2的19/25个目标可靠性分类；主结论方向没有被这些干预推翻。它们不能替代阈值重训练，也没有用于选择主结果。

这些结果支持“轴方向依赖是HOTEL退化的一部分”，但速度噪声、运动分布、表示与优化影响仍混合；不支持方向归一化已构成跨录像稳定的新主干。

训练前历史统计表明两个录像的运动分布并不一致。可靠朝向目标中，训练集的纵向运动比例在ZARA2留出划分约29.06%、HOTEL留出划分约24.92%；验证分别约0.68%与89.36%。此处纵向指|vy|>|vx|、平均历史速度≥0.2m/s，只分析历史；方向与录像相关但不是唯一可能混杂。原始八扇区计数及所有连续特征见history_distribution.csv。

预设速度组使用当前连续历史段时间加权平均速度向量的模长：低速<0.2，中速[0.2,1)，高速≥1m/s，无可用速度另列。往返运动也可能具有较小净平均速度，因此低速不等于真实静止。HOTEL低速224/318，ZARA2低速648/1092。只评价A完整历史目标，因此部分历史组为空，仍显式保留。所有组会重叠，不能相加当作独立样本；窗口复用同一行人，N是目标窗口数而非独立行人数。

## 坐标变化与优化的区别

| 录像 | 对照 | Δbest ADE(m) | Δ配对FDE(m) |
| --- | --- | --- | --- |
| crowds_zara02 | M2 − M0 | 0.005021 | 0.006293 |
| crowds_zara02 | M3 − M0 | 0.000138 | 0.005519 |
| crowds_zara02 | M3 − M2 | -0.004883 | -0.000774 |
| crowds_zara02 | M3 − CVR-Unnormalized-reference | 0.006141 | 0.018919 |
| crowds_zara02 | M2 − CV | -0.008380 | -0.030669 |
| crowds_zara02 | M3 − CV | -0.013263 | -0.031442 |
| biwi_hotel | M2 − M0 | -0.167667 | -0.443657 |
| biwi_hotel | M3 − M0 | -0.165218 | -0.381661 |
| biwi_hotel | M3 − M2 | 0.002449 | 0.061996 |
| biwi_hotel | M3 − CVR-Unnormalized-reference | -0.063216 | -0.168932 |
| biwi_hotel | M2 − CV | 0.006136 | -0.049184 |
| biwi_hotel | M3 − CV | 0.008585 | 0.012812 |

M2−M0保持参数、初始化、数据、loss及训练策略相同，改变局部坐标的计算归纳偏置；M3−原坐标CV残差保持CV先验。M3−M2则主要检验CV主路径。这里坐标变换也会改变梯度方向及优化过程，不能从单seed、各自best checkpoint的误差完全排除“优化更容易”。需要同时看固定last30和同endpoint训练误差，不能把best提升直接等同独立方向因果证明。fixed_epoch_comparison.csv保存全部30个相同预算epoch的配对差值，作为学习过程描述，不再挑选新checkpoint。没有进行学习率搜索或添加更多训练。

| 录像 | 模型 | best训练ADE | best验证ADE | last训练ADE | last验证ADE |
| --- | --- | --- | --- | --- | --- |
| crowds_zara02 | CV | 0.508975 | 0.296306 | 0.508975 | 0.296306 |
| crowds_zara02 | M0 | 0.464847 | 0.282905 | 0.467905 | 0.298853 |
| crowds_zara02 | M2 | 0.479336 | 0.287926 | 0.475113 | 0.295772 |
| crowds_zara02 | M3 | 0.475875 | 0.283043 | 0.448202 | 0.291239 |
| crowds_zara02 | CVR-Unnormalized-reference | 0.444248 | 0.276902 | 0.431426 | 0.285986 |
| biwi_hotel | CV | 0.485746 | 0.329516 | 0.485746 | 0.329516 |
| biwi_hotel | M0 | 0.447608 | 0.503319 | 0.443112 | 0.526877 |
| biwi_hotel | M2 | 0.482089 | 0.335652 | 0.456052 | 0.416973 |
| biwi_hotel | M3 | 0.473534 | 0.338101 | 0.425460 | 0.491875 |
| biwi_hotel | CVR-Unnormalized-reference | 0.456808 | 0.401317 | 0.401987 | 0.742461 |

## 合法全局旋转检查

| 录像 | 模型 | 历史组 | N | 90°预测等变平均误差(m) | 最大误差(m) |
| --- | --- | --- | --- | --- | --- |
| crowds_zara02 | M0 | all | 1092 | 0.204154 | 2.162992 |
| crowds_zara02 | M0 | heading_reliable | 444 | 0.459326 | 2.162992 |
| crowds_zara02 | M0 | heading_fallback | 648 | 0.029314 | 1.394086 |
| crowds_zara02 | M2 | all | 1092 | 0.011607 | 1.551556 |
| crowds_zara02 | M2 | heading_reliable | 444 | 0.000000 | 0.000000 |
| crowds_zara02 | M2 | heading_fallback | 648 | 0.019560 | 1.551556 |
| crowds_zara02 | M3 | all | 1092 | 0.006198 | 0.608836 |
| crowds_zara02 | M3 | heading_reliable | 444 | 0.000000 | 0.000000 |
| crowds_zara02 | M3 | heading_fallback | 648 | 0.010445 | 0.608836 |
| biwi_hotel | M0 | all | 318 | 0.284516 | 6.097806 |
| biwi_hotel | M0 | heading_reliable | 94 | 0.763283 | 6.097806 |
| biwi_hotel | M0 | heading_fallback | 224 | 0.083605 | 1.094473 |
| biwi_hotel | M2 | all | 318 | 0.091413 | 1.869538 |
| biwi_hotel | M2 | heading_reliable | 94 | 0.000000 | 0.000000 |
| biwi_hotel | M2 | heading_fallback | 224 | 0.129773 | 1.869538 |
| biwi_hotel | M3 | all | 318 | 0.011663 | 0.331060 |
| biwi_hotel | M3 | heading_reliable | 94 | 0.000000 | 0.000000 |
| biwi_hotel | M3 | heading_fallback | 224 | 0.016558 | 0.331060 |

±90°、180°的全部数据见rotation_checks.csv。只对完成训练后的固定best checkpoint同步旋转历史输入与指标GT，检查预测应如何变换；GT仅用于误差计算。可靠方向组的M2/M3应在浮点误差范围等变，恒等低速回退不具有该保证。M0通常依赖全局轴方向。旋转指标不用于再选模型、阈值或checkpoint，也不是新增独立训练任务。

## 阈值敏感性

| 录像 | 模型 | 推理阈值(m/s) | 切换目标数 | N | ADE | FDE |
| --- | --- | --- | --- | --- | --- | --- |
| crowds_zara02 | M2 | 0.1 | 19 | 1092 | 0.287594 | 0.636266 |
| crowds_zara02 | M2 | 0.4 | 25 | 1092 | 0.289154 | 0.639394 |
| crowds_zara02 | M3 | 0.1 | 19 | 1092 | 0.283099 | 0.636261 |
| crowds_zara02 | M3 | 0.4 | 25 | 1092 | 0.283331 | 0.636582 |
| biwi_hotel | M2 | 0.1 | 5 | 318 | 0.339059 | 0.568572 |
| biwi_hotel | M2 | 0.4 | 6 | 318 | 0.333226 | 0.555919 |
| biwi_hotel | M3 | 0.1 | 5 | 318 | 0.337651 | 0.622781 |
| biwi_hotel | M3 | 0.4 | 6 | 318 | 0.338516 | 0.624919 |

训练阈值固定0.2m/s；0.1和0.4只在同一best权重推理时切换。分组始终沿用原0.2定义，记录切换目标数。完整逐组结果见threshold_sensitivity.csv；这是一项接口干预，不能替代预注册的独立阈值重训练或用最低干预误差覆盖主结果。

## 全部历史组主结果

| 录像 | 历史组 | N | 模型 | ADE | FDE | ΔADE vs M0 | ΔADE vs CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crowds_zara02 | all | 1092 | CV | 0.296306 | 0.667523 | 0.013401 | 0.000000 |
| crowds_zara02 | no_neighbors | 33 | CV | 0.163215 | 0.338940 | 0.001831 | 0.000000 |
| crowds_zara02 | low_density | 145 | CV | 0.156175 | 0.343435 | 0.006110 | 0.000000 |
| crowds_zara02 | medium_density | 181 | CV | 0.266820 | 0.612859 | 0.016356 | 0.000000 |
| crowds_zara02 | high_density | 733 | CV | 0.337299 | 0.759924 | 0.014635 | 0.000000 |
| crowds_zara02 | history_full8 | 1092 | CV | 0.296306 | 0.667523 | 0.013401 | 0.000000 |
| crowds_zara02 | history_partial | 0 | CV | — | — | — | — |
| crowds_zara02 | history1 | 0 | CV | — | — | — | — |
| crowds_zara02 | history2to4 | 0 | CV | — | — | — | — |
| crowds_zara02 | history5to7 | 0 | CV | — | — | — | — |
| crowds_zara02 | opposing | 214 | CV | 0.532410 | 1.075528 | 0.055135 | 0.000000 |
| crowds_zara02 | crossing | 28 | CV | 0.978659 | 2.031301 | 0.009324 | 0.000000 |
| crowds_zara02 | co_moving | 386 | CV | 0.535359 | 1.099644 | 0.053786 | 0.000000 |
| crowds_zara02 | motion_unclassified | 679 | CV | 0.142912 | 0.387054 | -0.011178 | 0.000000 |
| crowds_zara02 | relationship_changing | 0 | CV | — | — | — | — |
| crowds_zara02 | relationship_stable | 1052 | CV | 0.300266 | 0.676745 | 0.013553 | 0.000000 |
| crowds_zara02 | relationship_insufficient | 40 | CV | 0.192148 | 0.424967 | 0.009411 | 0.000000 |
| crowds_zara02 | no_legal_social_messages | 33 | CV | 0.163215 | 0.338940 | 0.001831 | 0.000000 |
| crowds_zara02 | history_CV_like | 1079 | CV | 0.292291 | 0.661295 | 0.014277 | 0.000000 |
| crowds_zara02 | history_motion_moderate | 13 | CV | 0.629514 | 1.184430 | -0.059328 | 0.000000 |
| crowds_zara02 | history_velocity_changing | 0 | CV | — | — | — | — |
| crowds_zara02 | history_motion_insufficient | 0 | CV | — | — | — | — |
| crowds_zara02 | heading_reliable | 444 | CV | 0.514697 | 1.070581 | 0.050539 | 0.000000 |
| crowds_zara02 | heading_fallback | 648 | CV | 0.146667 | 0.391353 | -0.012045 | 0.000000 |
| crowds_zara02 | history_speed_low | 648 | CV | 0.146667 | 0.391353 | -0.012045 | 0.000000 |
| crowds_zara02 | history_speed_medium | 135 | CV | 0.672595 | 1.510983 | 0.041281 | 0.000000 |
| crowds_zara02 | history_speed_high | 309 | CV | 0.445713 | 0.878172 | 0.054584 | 0.000000 |
| crowds_zara02 | history_speed_unknown | 0 | CV | — | — | — | — |
| crowds_zara02 | all | 1092 | M0 | 0.282905 | 0.630561 | 0.000000 | -0.013401 |
| crowds_zara02 | no_neighbors | 33 | M0 | 0.161385 | 0.333593 | 0.000000 | -0.001831 |
| crowds_zara02 | low_density | 145 | M0 | 0.150065 | 0.319574 | 0.000000 | -0.006110 |
| crowds_zara02 | medium_density | 181 | M0 | 0.250464 | 0.564587 | 0.000000 | -0.016356 |
| crowds_zara02 | high_density | 733 | M0 | 0.322664 | 0.721740 | 0.000000 | -0.014635 |
| crowds_zara02 | history_full8 | 1092 | M0 | 0.282905 | 0.630561 | 0.000000 | -0.013401 |
| crowds_zara02 | history_partial | 0 | M0 | — | — | — | — |
| crowds_zara02 | history1 | 0 | M0 | — | — | — | — |
| crowds_zara02 | history2to4 | 0 | M0 | — | — | — | — |
| crowds_zara02 | history5to7 | 0 | M0 | — | — | — | — |
| crowds_zara02 | opposing | 214 | M0 | 0.477275 | 0.940473 | 0.000000 | -0.055135 |
| crowds_zara02 | crossing | 28 | M0 | 0.969336 | 1.993729 | 0.000000 | -0.009324 |
| crowds_zara02 | co_moving | 386 | M0 | 0.481573 | 0.968565 | 0.000000 | -0.053786 |
| crowds_zara02 | motion_unclassified | 679 | M0 | 0.154090 | 0.407411 | 0.000000 | 0.011178 |
| crowds_zara02 | relationship_changing | 0 | M0 | — | — | — | — |
| crowds_zara02 | relationship_stable | 1052 | M0 | 0.286713 | 0.639443 | 0.000000 | -0.013553 |
| crowds_zara02 | relationship_insufficient | 40 | M0 | 0.182737 | 0.396969 | 0.000000 | -0.009411 |
| crowds_zara02 | no_legal_social_messages | 33 | M0 | 0.161385 | 0.333593 | 0.000000 | -0.001831 |
| crowds_zara02 | history_CV_like | 1079 | M0 | 0.278014 | 0.621578 | 0.000000 | -0.014277 |
| crowds_zara02 | history_motion_moderate | 13 | M0 | 0.688842 | 1.376154 | 0.000000 | 0.059328 |
| crowds_zara02 | history_velocity_changing | 0 | M0 | — | — | — | — |
| crowds_zara02 | history_motion_insufficient | 0 | M0 | — | — | — | — |
| crowds_zara02 | heading_reliable | 444 | M0 | 0.464158 | 0.944495 | 0.000000 | -0.050539 |
| crowds_zara02 | heading_fallback | 648 | M0 | 0.158713 | 0.415458 | 0.000000 | 0.012045 |
| crowds_zara02 | history_speed_low | 648 | M0 | 0.158713 | 0.415458 | 0.000000 | 0.012045 |
| crowds_zara02 | history_speed_medium | 135 | M0 | 0.631314 | 1.396226 | 0.000000 | -0.041281 |
| crowds_zara02 | history_speed_high | 309 | M0 | 0.391129 | 0.747137 | 0.000000 | -0.054584 |
| crowds_zara02 | history_speed_unknown | 0 | M0 | — | — | — | — |
| crowds_zara02 | all | 1092 | M2 | 0.287926 | 0.636854 | 0.005021 | -0.008380 |
| crowds_zara02 | no_neighbors | 33 | M2 | 0.141575 | 0.283340 | -0.019810 | -0.021640 |
| crowds_zara02 | low_density | 145 | M2 | 0.148818 | 0.317691 | -0.001247 | -0.007357 |
| crowds_zara02 | medium_density | 181 | M2 | 0.257352 | 0.568934 | 0.006888 | -0.009468 |
| crowds_zara02 | high_density | 733 | M2 | 0.329582 | 0.732677 | 0.006918 | -0.007717 |
| crowds_zara02 | history_full8 | 1092 | M2 | 0.287926 | 0.636854 | 0.005021 | -0.008380 |
| crowds_zara02 | history_partial | 0 | M2 | — | — | — | — |
| crowds_zara02 | history1 | 0 | M2 | — | — | — | — |
| crowds_zara02 | history2to4 | 0 | M2 | — | — | — | — |
| crowds_zara02 | history5to7 | 0 | M2 | — | — | — | — |
| crowds_zara02 | opposing | 214 | M2 | 0.528881 | 1.057370 | 0.051606 | -0.003529 |
| crowds_zara02 | crossing | 28 | M2 | 0.938077 | 1.932587 | -0.031259 | -0.040583 |
| crowds_zara02 | co_moving | 386 | M2 | 0.511692 | 1.023991 | 0.030119 | -0.023667 |
| crowds_zara02 | motion_unclassified | 679 | M2 | 0.144896 | 0.387113 | -0.009195 | 0.001984 |
| crowds_zara02 | relationship_changing | 0 | M2 | — | — | — | — |
| crowds_zara02 | relationship_stable | 1052 | M2 | 0.292425 | 0.647072 | 0.005712 | -0.007841 |
| crowds_zara02 | relationship_insufficient | 40 | M2 | 0.169588 | 0.368116 | -0.013149 | -0.022560 |
| crowds_zara02 | no_legal_social_messages | 33 | M2 | 0.141575 | 0.283340 | -0.019810 | -0.021640 |
| crowds_zara02 | history_CV_like | 1079 | M2 | 0.281990 | 0.625720 | 0.003976 | -0.010301 |
| crowds_zara02 | history_motion_moderate | 13 | M2 | 0.780553 | 1.560995 | 0.091712 | 0.151040 |
| crowds_zara02 | history_velocity_changing | 0 | M2 | — | — | — | — |
| crowds_zara02 | history_motion_insufficient | 0 | M2 | — | — | — | — |
| crowds_zara02 | heading_reliable | 444 | M2 | 0.488593 | 0.982358 | 0.024435 | -0.026104 |
| crowds_zara02 | heading_fallback | 648 | M2 | 0.150431 | 0.400120 | -0.008282 | 0.003764 |
| crowds_zara02 | history_speed_low | 648 | M2 | 0.150431 | 0.400120 | -0.008282 | 0.003764 |
| crowds_zara02 | history_speed_medium | 135 | M2 | 0.627351 | 1.370402 | -0.003962 | -0.045244 |
| crowds_zara02 | history_speed_high | 309 | M2 | 0.427971 | 0.812825 | 0.036842 | -0.017741 |
| crowds_zara02 | history_speed_unknown | 0 | M2 | — | — | — | — |
| crowds_zara02 | all | 1092 | M3 | 0.283043 | 0.636080 | 0.000138 | -0.013263 |
| crowds_zara02 | no_neighbors | 33 | M3 | 0.154779 | 0.309426 | -0.006605 | -0.008436 |
| crowds_zara02 | low_density | 145 | M3 | 0.146529 | 0.315895 | -0.003535 | -0.009646 |
| crowds_zara02 | medium_density | 181 | M3 | 0.248311 | 0.563294 | -0.002153 | -0.018509 |
| crowds_zara02 | high_density | 733 | M3 | 0.324399 | 0.732098 | 0.001734 | -0.012900 |
| crowds_zara02 | history_full8 | 1092 | M3 | 0.283043 | 0.636080 | 0.000138 | -0.013263 |
| crowds_zara02 | history_partial | 0 | M3 | — | — | — | — |
| crowds_zara02 | history1 | 0 | M3 | — | — | — | — |
| crowds_zara02 | history2to4 | 0 | M3 | — | — | — | — |
| crowds_zara02 | history5to7 | 0 | M3 | — | — | — | — |
| crowds_zara02 | opposing | 214 | M3 | 0.520698 | 1.054413 | 0.043423 | -0.011712 |
| crowds_zara02 | crossing | 28 | M3 | 0.950714 | 1.964656 | -0.018621 | -0.027945 |
| crowds_zara02 | co_moving | 386 | M3 | 0.500849 | 1.021699 | 0.019276 | -0.034510 |
| crowds_zara02 | motion_unclassified | 679 | M3 | 0.144474 | 0.388383 | -0.009616 | 0.001562 |
| crowds_zara02 | relationship_changing | 0 | M3 | — | — | — | — |
| crowds_zara02 | relationship_stable | 1052 | M3 | 0.286946 | 0.645439 | 0.000233 | -0.013320 |
| crowds_zara02 | relationship_insufficient | 40 | M3 | 0.180396 | 0.389951 | -0.002341 | -0.011751 |
| crowds_zara02 | no_legal_social_messages | 33 | M3 | 0.154779 | 0.309426 | -0.006605 | -0.008436 |
| crowds_zara02 | history_CV_like | 1079 | M3 | 0.278685 | 0.627943 | 0.000671 | -0.013606 |
| crowds_zara02 | history_motion_moderate | 13 | M3 | 0.644776 | 1.311528 | -0.044066 | 0.015263 |
| crowds_zara02 | history_velocity_changing | 0 | M3 | — | — | — | — |
| crowds_zara02 | history_motion_insufficient | 0 | M3 | — | — | — | — |
| crowds_zara02 | heading_reliable | 444 | M3 | 0.477315 | 0.983009 | 0.013157 | -0.037382 |
| crowds_zara02 | heading_fallback | 648 | M3 | 0.149931 | 0.398370 | -0.008782 | 0.003263 |
| crowds_zara02 | history_speed_low | 648 | M3 | 0.149931 | 0.398370 | -0.008782 | 0.003263 |
| crowds_zara02 | history_speed_medium | 135 | M3 | 0.608904 | 1.342474 | -0.022409 | -0.063691 |
| crowds_zara02 | history_speed_high | 309 | M3 | 0.419825 | 0.825962 | 0.028696 | -0.025888 |
| crowds_zara02 | history_speed_unknown | 0 | M3 | — | — | — | — |
| crowds_zara02 | all | 1092 | CVR-Unnormalized-reference | 0.276902 | 0.617161 | -0.006003 | -0.019404 |
| crowds_zara02 | no_neighbors | 33 | CVR-Unnormalized-reference | 0.148078 | 0.308899 | -0.013307 | -0.015137 |
| crowds_zara02 | low_density | 145 | CVR-Unnormalized-reference | 0.144949 | 0.308217 | -0.005116 | -0.011226 |
| crowds_zara02 | medium_density | 181 | CVR-Unnormalized-reference | 0.247766 | 0.549897 | -0.002698 | -0.019054 |
| crowds_zara02 | high_density | 733 | CVR-Unnormalized-reference | 0.315998 | 0.708763 | -0.006666 | -0.021301 |
| crowds_zara02 | history_full8 | 1092 | CVR-Unnormalized-reference | 0.276902 | 0.617161 | -0.006003 | -0.019404 |
| crowds_zara02 | history_partial | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | history1 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | history2to4 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | history5to7 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | opposing | 214 | CVR-Unnormalized-reference | 0.490684 | 0.978519 | 0.013409 | -0.041726 |
| crowds_zara02 | crossing | 28 | CVR-Unnormalized-reference | 0.948286 | 1.944421 | -0.021049 | -0.030373 |
| crowds_zara02 | co_moving | 386 | CVR-Unnormalized-reference | 0.483727 | 0.975809 | 0.002154 | -0.051632 |
| crowds_zara02 | motion_unclassified | 679 | CVR-Unnormalized-reference | 0.143495 | 0.382147 | -0.010595 | 0.000583 |
| crowds_zara02 | relationship_changing | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | relationship_stable | 1052 | CVR-Unnormalized-reference | 0.280938 | 0.626245 | -0.005775 | -0.019328 |
| crowds_zara02 | relationship_insufficient | 40 | CVR-Unnormalized-reference | 0.170740 | 0.378251 | -0.011996 | -0.021407 |
| crowds_zara02 | no_legal_social_messages | 33 | CVR-Unnormalized-reference | 0.148078 | 0.308899 | -0.013307 | -0.015137 |
| crowds_zara02 | history_CV_like | 1079 | CVR-Unnormalized-reference | 0.272596 | 0.609093 | -0.005418 | -0.019695 |
| crowds_zara02 | history_motion_moderate | 13 | CVR-Unnormalized-reference | 0.634278 | 1.286846 | -0.054564 | 0.004764 |
| crowds_zara02 | history_velocity_changing | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | history_motion_insufficient | 0 | CVR-Unnormalized-reference | — | — | — | — |
| crowds_zara02 | heading_reliable | 444 | CVR-Unnormalized-reference | 0.463455 | 0.942970 | -0.000703 | -0.051242 |
| crowds_zara02 | heading_fallback | 648 | CVR-Unnormalized-reference | 0.149078 | 0.393922 | -0.009635 | 0.002411 |
| crowds_zara02 | history_speed_low | 648 | CVR-Unnormalized-reference | 0.149078 | 0.393922 | -0.009635 | 0.002411 |
| crowds_zara02 | history_speed_medium | 135 | CVR-Unnormalized-reference | 0.609239 | 1.344635 | -0.022074 | -0.063356 |
| crowds_zara02 | history_speed_high | 309 | CVR-Unnormalized-reference | 0.399763 | 0.767485 | 0.008634 | -0.045949 |
| crowds_zara02 | history_speed_unknown | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | all | 318 | CV | 0.329516 | 0.611038 | -0.173803 | 0.000000 |
| biwi_hotel | no_neighbors | 14 | CV | 0.299495 | 0.523674 | -0.301331 | 0.000000 |
| biwi_hotel | low_density | 37 | CV | 0.284401 | 0.537619 | -0.136563 | 0.000000 |
| biwi_hotel | medium_density | 134 | CV | 0.299342 | 0.540103 | -0.118034 | 0.000000 |
| biwi_hotel | high_density | 133 | CV | 0.375628 | 0.712129 | -0.226927 | 0.000000 |
| biwi_hotel | history_full8 | 318 | CV | 0.329516 | 0.611038 | -0.173803 | 0.000000 |
| biwi_hotel | history_partial | 0 | CV | — | — | — | — |
| biwi_hotel | history1 | 0 | CV | — | — | — | — |
| biwi_hotel | history2to4 | 0 | CV | — | — | — | — |
| biwi_hotel | history5to7 | 0 | CV | — | — | — | — |
| biwi_hotel | opposing | 69 | CV | 0.532671 | 1.001747 | -0.487322 | 0.000000 |
| biwi_hotel | crossing | 32 | CV | 0.745763 | 1.376372 | -0.205135 | 0.000000 |
| biwi_hotel | co_moving | 70 | CV | 0.566768 | 1.079397 | -0.541035 | 0.000000 |
| biwi_hotel | motion_unclassified | 217 | CV | 0.198951 | 0.363654 | -0.050120 | 0.000000 |
| biwi_hotel | relationship_changing | 11 | CV | 0.858008 | 1.570967 | 0.130467 | 0.000000 |
| biwi_hotel | relationship_stable | 287 | CV | 0.314063 | 0.583567 | -0.177335 | 0.000000 |
| biwi_hotel | relationship_insufficient | 20 | CV | 0.260590 | 0.477295 | -0.290475 | 0.000000 |
| biwi_hotel | no_legal_social_messages | 14 | CV | 0.299495 | 0.523674 | -0.301331 | 0.000000 |
| biwi_hotel | history_CV_like | 223 | CV | 0.226155 | 0.424691 | -0.152285 | 0.000000 |
| biwi_hotel | history_motion_moderate | 92 | CV | 0.558096 | 1.018124 | -0.237738 | 0.000000 |
| biwi_hotel | history_velocity_changing | 3 | CV | 1.002878 | 1.978889 | 0.187333 | 0.000000 |
| biwi_hotel | history_motion_insufficient | 0 | CV | — | — | — | — |
| biwi_hotel | heading_reliable | 94 | CV | 0.521187 | 0.992030 | -0.596592 | 0.000000 |
| biwi_hotel | heading_fallback | 224 | CV | 0.249082 | 0.451158 | 0.003617 | 0.000000 |
| biwi_hotel | history_speed_low | 224 | CV | 0.249082 | 0.451158 | 0.003617 | 0.000000 |
| biwi_hotel | history_speed_medium | 15 | CV | 0.762591 | 1.450818 | 0.166709 | 0.000000 |
| biwi_hotel | history_speed_high | 79 | CV | 0.475351 | 0.904918 | -0.741522 | 0.000000 |
| biwi_hotel | history_speed_unknown | 0 | CV | — | — | — | — |
| biwi_hotel | all | 318 | M0 | 0.503319 | 1.005511 | 0.000000 | 0.173803 |
| biwi_hotel | no_neighbors | 14 | M0 | 0.600825 | 1.226024 | 0.000000 | 0.301331 |
| biwi_hotel | low_density | 37 | M0 | 0.420964 | 0.839015 | 0.000000 | 0.136563 |
| biwi_hotel | medium_density | 134 | M0 | 0.417376 | 0.816587 | 0.000000 | 0.118034 |
| biwi_hotel | high_density | 133 | M0 | 0.602555 | 1.218964 | 0.000000 | 0.226927 |
| biwi_hotel | history_full8 | 318 | M0 | 0.503319 | 1.005511 | 0.000000 | 0.173803 |
| biwi_hotel | history_partial | 0 | M0 | — | — | — | — |
| biwi_hotel | history1 | 0 | M0 | — | — | — | — |
| biwi_hotel | history2to4 | 0 | M0 | — | — | — | — |
| biwi_hotel | history5to7 | 0 | M0 | — | — | — | — |
| biwi_hotel | opposing | 69 | M0 | 1.019994 | 2.154473 | 0.000000 | 0.487322 |
| biwi_hotel | crossing | 32 | M0 | 0.950898 | 1.818969 | 0.000000 | 0.205135 |
| biwi_hotel | co_moving | 70 | M0 | 1.107803 | 2.348656 | 0.000000 | 0.541035 |
| biwi_hotel | motion_unclassified | 217 | M0 | 0.249071 | 0.469059 | 0.000000 | 0.050120 |
| biwi_hotel | relationship_changing | 11 | M0 | 0.727540 | 1.280749 | 0.000000 | -0.130467 |
| biwi_hotel | relationship_stable | 287 | M0 | 0.491398 | 0.985865 | 0.000000 | 0.177335 |
| biwi_hotel | relationship_insufficient | 20 | M0 | 0.551064 | 1.136052 | 0.000000 | 0.290475 |
| biwi_hotel | no_legal_social_messages | 14 | M0 | 0.600825 | 1.226024 | 0.000000 | 0.301331 |
| biwi_hotel | history_CV_like | 223 | M0 | 0.378440 | 0.772692 | 0.000000 | 0.152285 |
| biwi_hotel | history_motion_moderate | 92 | M0 | 0.795834 | 1.553743 | 0.000000 | 0.237738 |
| biwi_hotel | history_velocity_changing | 3 | M0 | 0.815545 | 1.499353 | 0.000000 | -0.187333 |
| biwi_hotel | history_motion_insufficient | 0 | M0 | — | — | — | — |
| biwi_hotel | heading_reliable | 94 | M0 | 1.117779 | 2.383933 | 0.000000 | 0.596592 |
| biwi_hotel | heading_fallback | 224 | M0 | 0.245465 | 0.427067 | 0.000000 | -0.003617 |
| biwi_hotel | history_speed_low | 224 | M0 | 0.245465 | 0.427067 | 0.000000 | -0.003617 |
| biwi_hotel | history_speed_medium | 15 | M0 | 0.595882 | 1.076442 | 0.000000 | -0.166709 |
| biwi_hotel | history_speed_high | 79 | M0 | 1.216874 | 2.632191 | 0.000000 | 0.741522 |
| biwi_hotel | history_speed_unknown | 0 | M0 | — | — | — | — |
| biwi_hotel | all | 318 | M2 | 0.335652 | 0.561855 | -0.167667 | 0.006136 |
| biwi_hotel | no_neighbors | 14 | M2 | 0.378109 | 0.625562 | -0.222716 | 0.078614 |
| biwi_hotel | low_density | 37 | M2 | 0.275587 | 0.460557 | -0.145377 | -0.008814 |
| biwi_hotel | medium_density | 134 | M2 | 0.299672 | 0.469182 | -0.117704 | 0.000331 |
| biwi_hotel | high_density | 133 | M2 | 0.384143 | 0.676698 | -0.218412 | 0.008516 |
| biwi_hotel | history_full8 | 318 | M2 | 0.335652 | 0.561855 | -0.167667 | 0.006136 |
| biwi_hotel | history_partial | 0 | M2 | — | — | — | — |
| biwi_hotel | history1 | 0 | M2 | — | — | — | — |
| biwi_hotel | history2to4 | 0 | M2 | — | — | — | — |
| biwi_hotel | history5to7 | 0 | M2 | — | — | — | — |
| biwi_hotel | opposing | 69 | M2 | 0.586683 | 0.977632 | -0.433310 | 0.054012 |
| biwi_hotel | crossing | 32 | M2 | 0.660570 | 1.071520 | -0.290328 | -0.085193 |
| biwi_hotel | co_moving | 70 | M2 | 0.610708 | 1.040223 | -0.497095 | 0.043940 |
| biwi_hotel | motion_unclassified | 217 | M2 | 0.199366 | 0.333499 | -0.049705 | 0.000415 |
| biwi_hotel | relationship_changing | 11 | M2 | 0.792889 | 1.273130 | 0.065349 | -0.065118 |
| biwi_hotel | relationship_stable | 287 | M2 | 0.318397 | 0.535277 | -0.173001 | 0.004334 |
| biwi_hotel | relationship_insufficient | 20 | M2 | 0.331787 | 0.552038 | -0.219278 | 0.071197 |
| biwi_hotel | no_legal_social_messages | 14 | M2 | 0.378109 | 0.625562 | -0.222716 | 0.078614 |
| biwi_hotel | history_CV_like | 223 | M2 | 0.233249 | 0.394811 | -0.145190 | 0.007094 |
| biwi_hotel | history_motion_moderate | 92 | M2 | 0.567264 | 0.935913 | -0.228570 | 0.009168 |
| biwi_hotel | history_velocity_changing | 3 | M2 | 0.844824 | 1.507651 | 0.029279 | -0.158054 |
| biwi_hotel | history_motion_insufficient | 0 | M2 | — | — | — | — |
| biwi_hotel | heading_reliable | 94 | M2 | 0.618586 | 1.088791 | -0.499193 | 0.097399 |
| biwi_hotel | heading_fallback | 224 | M2 | 0.216921 | 0.340729 | -0.028544 | -0.032161 |
| biwi_hotel | history_speed_low | 224 | M2 | 0.216921 | 0.340729 | -0.028544 | -0.032161 |
| biwi_hotel | history_speed_medium | 15 | M2 | 0.649218 | 1.105620 | 0.053336 | -0.113373 |
| biwi_hotel | history_speed_high | 79 | M2 | 0.612770 | 1.085595 | -0.604103 | 0.137419 |
| biwi_hotel | history_speed_unknown | 0 | M2 | — | — | — | — |
| biwi_hotel | all | 318 | M3 | 0.338101 | 0.623851 | -0.165218 | 0.008585 |
| biwi_hotel | no_neighbors | 14 | M3 | 0.333308 | 0.614696 | -0.267517 | 0.033813 |
| biwi_hotel | low_density | 37 | M3 | 0.283286 | 0.520431 | -0.137679 | -0.001116 |
| biwi_hotel | medium_density | 134 | M3 | 0.302711 | 0.535660 | -0.114666 | 0.003369 |
| biwi_hotel | high_density | 133 | M3 | 0.389512 | 0.742439 | -0.213043 | 0.013885 |
| biwi_hotel | history_full8 | 318 | M3 | 0.338101 | 0.623851 | -0.165218 | 0.008585 |
| biwi_hotel | history_partial | 0 | M3 | — | — | — | — |
| biwi_hotel | history1 | 0 | M3 | — | — | — | — |
| biwi_hotel | history2to4 | 0 | M3 | — | — | — | — |
| biwi_hotel | history5to7 | 0 | M3 | — | — | — | — |
| biwi_hotel | opposing | 69 | M3 | 0.561404 | 1.053586 | -0.458590 | 0.028733 |
| biwi_hotel | crossing | 32 | M3 | 0.762771 | 1.396793 | -0.188128 | 0.017007 |
| biwi_hotel | co_moving | 70 | M3 | 0.590286 | 1.121180 | -0.517516 | 0.023519 |
| biwi_hotel | motion_unclassified | 217 | M3 | 0.201162 | 0.365561 | -0.047909 | 0.002211 |
| biwi_hotel | relationship_changing | 11 | M3 | 0.851021 | 1.506292 | 0.123481 | -0.006987 |
| biwi_hotel | relationship_stable | 287 | M3 | 0.321817 | 0.595292 | -0.169581 | 0.007753 |
| biwi_hotel | relationship_insufficient | 20 | M3 | 0.289682 | 0.548320 | -0.261382 | 0.029093 |
| biwi_hotel | no_legal_social_messages | 14 | M3 | 0.333308 | 0.614696 | -0.267517 | 0.033813 |
| biwi_hotel | history_CV_like | 223 | M3 | 0.235244 | 0.443847 | -0.143196 | 0.009089 |
| biwi_hotel | history_motion_moderate | 92 | M3 | 0.567524 | 1.022728 | -0.228311 | 0.009427 |
| biwi_hotel | history_velocity_changing | 3 | M3 | 0.948218 | 1.771875 | 0.132673 | -0.054660 |
| biwi_hotel | history_motion_insufficient | 0 | M3 | — | — | — | — |
| biwi_hotel | heading_reliable | 94 | M3 | 0.552137 | 1.057546 | -0.565642 | 0.030949 |
| biwi_hotel | heading_fallback | 224 | M3 | 0.248283 | 0.441853 | 0.002818 | -0.000799 |
| biwi_hotel | history_speed_low | 224 | M3 | 0.248283 | 0.441853 | 0.002818 | -0.000799 |
| biwi_hotel | history_speed_medium | 15 | M3 | 0.735262 | 1.347940 | 0.139380 | -0.027329 |
| biwi_hotel | history_speed_high | 79 | M3 | 0.517366 | 1.002408 | -0.699507 | 0.042015 |
| biwi_hotel | history_speed_unknown | 0 | M3 | — | — | — | — |
| biwi_hotel | all | 318 | CVR-Unnormalized-reference | 0.401317 | 0.792782 | -0.102002 | 0.071802 |
| biwi_hotel | no_neighbors | 14 | CVR-Unnormalized-reference | 0.393325 | 0.787989 | -0.207501 | 0.093830 |
| biwi_hotel | low_density | 37 | CVR-Unnormalized-reference | 0.328693 | 0.663053 | -0.092271 | 0.044291 |
| biwi_hotel | medium_density | 134 | CVR-Unnormalized-reference | 0.377091 | 0.727402 | -0.040285 | 0.077749 |
| biwi_hotel | high_density | 133 | CVR-Unnormalized-reference | 0.446771 | 0.895249 | -0.155784 | 0.071144 |
| biwi_hotel | history_full8 | 318 | CVR-Unnormalized-reference | 0.401317 | 0.792782 | -0.102002 | 0.071802 |
| biwi_hotel | history_partial | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | history1 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | history2to4 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | history5to7 | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | opposing | 69 | CVR-Unnormalized-reference | 0.796950 | 1.664113 | -0.223044 | 0.264279 |
| biwi_hotel | crossing | 32 | CVR-Unnormalized-reference | 0.832188 | 1.561223 | -0.118710 | 0.086425 |
| biwi_hotel | co_moving | 70 | CVR-Unnormalized-reference | 0.801953 | 1.692799 | -0.305849 | 0.235186 |
| biwi_hotel | motion_unclassified | 217 | CVR-Unnormalized-reference | 0.212237 | 0.396471 | -0.036834 | 0.013287 |
| biwi_hotel | relationship_changing | 11 | CVR-Unnormalized-reference | 0.911585 | 1.645162 | 0.184045 | 0.053577 |
| biwi_hotel | relationship_stable | 287 | CVR-Unnormalized-reference | 0.385713 | 0.766375 | -0.105685 | 0.071649 |
| biwi_hotel | relationship_insufficient | 20 | CVR-Unnormalized-reference | 0.344597 | 0.702913 | -0.206467 | 0.084007 |
| biwi_hotel | no_legal_social_messages | 14 | CVR-Unnormalized-reference | 0.393325 | 0.787989 | -0.207501 | 0.093830 |
| biwi_hotel | history_CV_like | 223 | CVR-Unnormalized-reference | 0.287647 | 0.586274 | -0.090793 | 0.061492 |
| biwi_hotel | history_motion_moderate | 92 | CVR-Unnormalized-reference | 0.658879 | 1.260255 | -0.136955 | 0.100783 |
| biwi_hotel | history_velocity_changing | 3 | CVR-Unnormalized-reference | 0.952257 | 1.807396 | 0.136712 | -0.050621 |
| biwi_hotel | history_motion_insufficient | 0 | CVR-Unnormalized-reference | — | — | — | — |
| biwi_hotel | heading_reliable | 94 | CVR-Unnormalized-reference | 0.780907 | 1.665562 | -0.336872 | 0.259720 |
| biwi_hotel | heading_fallback | 224 | CVR-Unnormalized-reference | 0.242025 | 0.426526 | -0.003440 | -0.007057 |
| biwi_hotel | history_speed_low | 224 | CVR-Unnormalized-reference | 0.242025 | 0.426526 | -0.003440 | -0.007057 |
| biwi_hotel | history_speed_medium | 15 | CVR-Unnormalized-reference | 0.745794 | 1.358381 | 0.149912 | -0.016797 |
| biwi_hotel | history_speed_high | 79 | CVR-Unnormalized-reference | 0.787574 | 1.723887 | -0.429299 | 0.312223 |
| biwi_hotel | history_speed_unknown | 0 | CVR-Unnormalized-reference | — | — | — | — |

全部best/last分组及空组见group_comparison.csv，共560条；逐目标配对误差见paired_target_errors.csv，共14,100条，目标集合逐项核对完全一致。不能将只在某组改善推广为整体稳定收益，也不能仅凭单seed证明跨随机性稳定。
