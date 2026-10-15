# Stage3.8：跨录像与历史可靠性分析

三个平滑估计器全部只改善HOTEL、退化其他4/5录像。所有5录像都已报告，没有把HOTEL重新选为新方法目标场景。

| 录像 | N目标窗口 | CV-Last2 ADE/FDE | CV-OLS4 ADE/FDE | CV-OLS8 ADE/FDE | CV-Median ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | 318 | 0.329516 / 0.611038 | 0.237814 / 0.438866 | 0.220144 / 0.408760 | 0.218247 / 0.402738 |
| students001 | 1169 | 0.380756 / 0.833056 | 0.427223 / 0.891753 | 0.507109 / 1.005962 | 0.514984 / 1.020309 |
| students003 | 706 | 0.727884 / 1.627788 | 0.811086 / 1.737995 | 0.931942 / 1.901824 | 0.949848 / 1.937206 |
| crowds_zara01 | 297 | 0.430096 / 0.953000 | 0.465470 / 0.994360 | 0.520416 / 1.076965 | 0.532430 / 1.094044 |
| crowds_zara02 | 1092 | 0.296306 / 0.667523 | 0.331875 / 0.715683 | 0.383175 / 0.787222 | 0.387353 / 0.796291 |

## 重叠窗口与行人权重

| 录像 | 模型 | 行人单位N | 均ADE改善人数 | 按行人均ΔADE |
| --- | --- | --- | --- | --- |
| biwi_hotel | CV-OLS4 | 28 | 22 | -0.103483 |
| biwi_hotel | CV-OLS8 | 28 | 19 | -0.076471 |
| biwi_hotel | CV-Median | 28 | 17 | -0.055096 |
| students001 | CV-OLS4 | 45 | 6 | 0.046682 |
| students001 | CV-OLS8 | 45 | 3 | 0.156724 |
| students001 | CV-Median | 45 | 4 | 0.168450 |
| students003 | CV-OLS4 | 48 | 2 | 0.083293 |
| students003 | CV-OLS8 | 48 | 3 | 0.200636 |
| students003 | CV-Median | 48 | 3 | 0.224441 |
| crowds_zara01 | CV-OLS4 | 21 | 2 | 0.041082 |
| crowds_zara01 | CV-OLS8 | 21 | 5 | 0.131764 |
| crowds_zara01 | CV-Median | 21 | 3 | 0.128247 |
| crowds_zara02 | CV-OLS4 | 35 | 5 | 0.053629 |
| crowds_zara02 | CV-OLS8 | 35 | 3 | 0.105071 |
| crowds_zara02 | CV-Median | 35 | 6 | 0.102933 |

HOTEL的OLS4/OLS8/Median分别改善22/28、19/28、17/28个行人的窗口均ADE；其他录像受益人数少，按行人检查仍未反转四录像退化。行人可共同经历场景事件，录像只有5个且已曝光，不能据此做普遍显著性或新确认集声明。

## 十项历史特征及补充可靠性量

| 录像 | 历史量（单位见01） | N窗口 | 均值 | 中位数 |
| --- | --- | --- | --- | --- |
| biwi_hotel | last_speed_mps | 318 | 0.407574 | 0.079057 |
| biwi_hotel | mean_velocity_speed_mps | 318 | 0.363508 | 0.039676 |
| biwi_hotel | mean_scalar_speed_mps | 318 | 0.401624 | 0.109172 |
| biwi_hotel | velocity_variance_m2ps2 | 318 | 0.023099 | 0.014273 |
| biwi_hotel | adjacent_velocity_change_mps | 318 | 0.144853 | 0.128040 |
| biwi_hotel | net_displacement_m | 318 | 1.017823 | 0.111092 |
| biwi_hotel | total_path_m | 318 | 1.124546 | 0.305681 |
| biwi_hotel | net_path_ratio | 318 | 0.728738 | 0.989507 |
| biwi_hotel | linear_fit_RMS_m | 318 | 0.037433 | 0.038295 |
| biwi_hotel | terminal_mean_velocity_difference_mps | 318 | 0.107481 | 0.068184 |
| students001 | last_speed_mps | 1169 | 0.415420 | 0.366098 |
| students001 | mean_velocity_speed_mps | 1169 | 0.417409 | 0.371640 |
| students001 | mean_scalar_speed_mps | 1169 | 0.427219 | 0.395774 |
| students001 | velocity_variance_m2ps2 | 1169 | 0.018464 | 0.009062 |
| students001 | adjacent_velocity_change_mps | 1169 | 0.055771 | 0.050809 |
| students001 | net_displacement_m | 1169 | 1.168746 | 1.040593 |
| students001 | total_path_m | 1169 | 1.196212 | 1.108167 |
| students001 | net_path_ratio | 1169 | 0.968963 | 0.993278 |
| students001 | linear_fit_RMS_m | 1169 | 0.043711 | 0.038031 |
| students001 | terminal_mean_velocity_difference_mps | 1169 | 0.124603 | 0.095344 |
| students003 | last_speed_mps | 706 | 0.838139 | 0.869997 |
| students003 | mean_velocity_speed_mps | 706 | 0.810387 | 0.837671 |
| students003 | mean_scalar_speed_mps | 706 | 0.822527 | 0.847773 |
| students003 | velocity_variance_m2ps2 | 706 | 0.037871 | 0.017017 |
| students003 | adjacent_velocity_change_mps | 706 | 0.082247 | 0.075708 |
| students003 | net_displacement_m | 706 | 2.269084 | 2.345478 |
| students003 | total_path_m | 706 | 2.303076 | 2.373764 |
| students003 | net_path_ratio | 706 | 0.982076 | 0.994243 |
| students003 | linear_fit_RMS_m | 706 | 0.065171 | 0.054415 |
| students003 | terminal_mean_velocity_difference_mps | 706 | 0.195465 | 0.160565 |
| crowds_zara01 | last_speed_mps | 297 | 1.010967 | 1.052997 |
| crowds_zara01 | mean_velocity_speed_mps | 297 | 1.042626 | 1.063469 |
| crowds_zara01 | mean_scalar_speed_mps | 297 | 1.046603 | 1.066020 |
| crowds_zara01 | velocity_variance_m2ps2 | 297 | 0.020395 | 0.007072 |
| crowds_zara01 | adjacent_velocity_change_mps | 297 | 0.054691 | 0.040138 |
| crowds_zara01 | net_displacement_m | 297 | 2.919352 | 2.977712 |
| crowds_zara01 | total_path_m | 297 | 2.930489 | 2.984855 |
| crowds_zara01 | net_path_ratio | 297 | 0.993806 | 0.999269 |
| crowds_zara01 | linear_fit_RMS_m | 297 | 0.044203 | 0.035003 |
| crowds_zara01 | terminal_mean_velocity_difference_mps | 297 | 0.120307 | 0.093648 |
| crowds_zara02 | last_speed_mps | 1092 | 0.440481 | 0.021479 |
| crowds_zara02 | mean_velocity_speed_mps | 1092 | 0.437001 | 0.029852 |
| crowds_zara02 | mean_scalar_speed_mps | 1092 | 0.441910 | 0.032265 |
| crowds_zara02 | velocity_variance_m2ps2 | 1092 | 0.013258 | 0.000000 |
| crowds_zara02 | adjacent_velocity_change_mps | 1092 | 0.035866 | 0.000795 |
| crowds_zara02 | net_displacement_m | 1092 | 1.223602 | 0.083584 |
| crowds_zara02 | total_path_m | 1092 | 1.237349 | 0.090341 |
| crowds_zara02 | net_path_ratio | 1092 | 0.976262 | 0.998892 |
| crowds_zara02 | linear_fit_RMS_m | 1092 | 0.028426 | 0.000096 |
| crowds_zara02 | terminal_mean_velocity_difference_mps | 1092 | 0.082568 | 0.000517 |

完整逐目标vx/vy、夹角及有效标记、历史长度见history_features.csv。均速向量模长和平均速率分开保存；往返小净位移不是静止证据。全部A目标历史均为8，历史长度的相关系数缺失是无变异，不填成0。

## 特征与误差的描述相关

| 录像 | 历史量 | 窗口N | 行人N | ρ:Last2 ADE窗口 | ρ:Last2 ADE行人均值 | ρ:OLS8−Last2窗口 | ρ:OLS8−Last2行人均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | velocity_variance_m2ps2 | 318 | 28 | 0.632 | 0.697 | -0.180 | -0.172 |
| biwi_hotel | net_path_ratio | 249 | 28 | 0.129 | -0.302 | 0.191 | 0.303 |
| biwi_hotel | linear_fit_RMS_m | 318 | 28 | 0.617 | 0.624 | -0.118 | 0.091 |
| biwi_hotel | terminal_mean_velocity_difference_mps | 318 | 28 | 0.829 | 0.712 | -0.393 | -0.056 |
| biwi_hotel | last_mean_angle_deg | 91 | 24 | 0.364 | 0.339 | -0.327 | -0.170 |
| students001 | velocity_variance_m2ps2 | 1169 | 45 | 0.505 | 0.609 | 0.317 | 0.462 |
| students001 | net_path_ratio | 1169 | 45 | -0.215 | 0.084 | -0.201 | 0.235 |
| students001 | linear_fit_RMS_m | 1169 | 45 | 0.495 | 0.636 | 0.307 | 0.470 |
| students001 | terminal_mean_velocity_difference_mps | 1169 | 45 | 0.474 | 0.632 | 0.403 | 0.489 |
| students001 | last_mean_angle_deg | 727 | 42 | 0.006 | 0.224 | 0.217 | -0.043 |
| students003 | velocity_variance_m2ps2 | 706 | 48 | 0.247 | 0.658 | 0.327 | 0.474 |
| students003 | net_path_ratio | 706 | 48 | -0.177 | -0.372 | -0.234 | -0.472 |
| students003 | linear_fit_RMS_m | 706 | 48 | 0.250 | 0.639 | 0.295 | 0.439 |
| students003 | terminal_mean_velocity_difference_mps | 706 | 48 | 0.196 | 0.668 | 0.446 | 0.504 |
| students003 | last_mean_angle_deg | 631 | 47 | 0.098 | 0.504 | 0.273 | 0.313 |
| crowds_zara01 | velocity_variance_m2ps2 | 297 | 21 | 0.248 | 0.855 | 0.187 | 0.057 |
| crowds_zara01 | net_path_ratio | 297 | 21 | -0.228 | -0.519 | -0.174 | -0.253 |
| crowds_zara01 | linear_fit_RMS_m | 297 | 21 | 0.264 | 0.910 | 0.148 | 0.019 |
| crowds_zara01 | terminal_mean_velocity_difference_mps | 297 | 21 | 0.277 | 0.940 | 0.317 | 0.103 |
| crowds_zara01 | last_mean_angle_deg | 278 | 21 | 0.228 | 0.752 | 0.142 | 0.106 |
| crowds_zara02 | velocity_variance_m2ps2 | 1092 | 35 | 0.738 | 0.734 | 0.276 | 0.378 |
| crowds_zara02 | net_path_ratio | 1092 | 35 | -0.463 | -0.081 | -0.190 | -0.149 |
| crowds_zara02 | linear_fit_RMS_m | 1092 | 35 | 0.740 | 0.755 | 0.276 | 0.248 |
| crowds_zara02 | terminal_mean_velocity_difference_mps | 1092 | 35 | 0.740 | 0.766 | 0.288 | 0.420 |
| crowds_zara02 | last_mean_angle_deg | 425 | 34 | 0.204 | 0.549 | 0.158 | 0.213 |

ρ为Spearman描述相关，没有p值、iid窗口置信区间或数据驱动分组阈值。完整Pearson/Spearman、ADE/FDE及delta指标见feature_error_associations.csv。夹角仅在双方速度足够时纳入，路径比剔除未定义的完全静止路径。相关可能受速度、静止比例、行人重复与真实运动变化混杂，不能等同噪声强度或预测时gate的可用收益。

关键反例：末端/均速差与OLS8−Last2 ADE的窗口ρ在HOTEL为−0.393，其他四录像为+0.403/+0.446/+0.317/+0.288。相同“不稳定”特征对应相反收益方向；只凭该指标不能设计一个已被证实跨录像有效的切换器。方差及拟合残差与Last2未来ADE普遍正相关，说明历史确有难度信息，但并不推出长历史速度能解决它。
