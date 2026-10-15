# Stage3.8：解析运动学与强基线结果

20项固定解析模型评价全部完成，另外8次兼容神经checkpoint推理；没有训练新网络、估计器或门控。所有误差均米制K1、完整8/12目标，四模型同一3582目标窗口，分属177个录像×行人单位。

| 录像 | N目标窗口 | CV-Last2 ADE/FDE | CV-OLS4 ADE/FDE | CV-OLS8 ADE/FDE | CV-Median ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | 318 | 0.329516 / 0.611038 | 0.237814 / 0.438866 | 0.220144 / 0.408760 | 0.218247 / 0.402738 |
| students001 | 1169 | 0.380756 / 0.833056 | 0.427223 / 0.891753 | 0.507109 / 1.005962 | 0.514984 / 1.020309 |
| students003 | 706 | 0.727884 / 1.627788 | 0.811086 / 1.737995 | 0.931942 / 1.901824 | 0.949848 / 1.937206 |
| crowds_zara01 | 297 | 0.430096 / 0.953000 | 0.465470 / 0.994360 | 0.520416 / 1.076965 | 0.532430 / 1.094044 |
| crowds_zara02 | 1092 | 0.296306 / 0.667523 | 0.331875 / 0.715683 | 0.383175 / 0.787222 | 0.387353 / 0.796291 |

## 不混用的三种平均

| 口径 | 模型 | 录像数 | N窗口 | 录像×行人单位 | ADE | FDE |
| --- | --- | --- | --- | --- | --- | --- |
| recording_macro | CV-Last2 | 5 | 3582 | 177 | 0.432911 | 0.938481 |
| target_window_weighted | CV-Last2 | 5 | 3582 | 177 | 0.422970 | 0.929466 |
| person_macro_within_recording_then_recording_macro | CV-Last2 | 5 | 3582 | 177 | 0.529462 | 1.117984 |
| recording_macro | CV-OLS4 | 5 | 3582 | 177 | 0.454694 | 0.955731 |
| target_window_weighted | CV-OLS4 | 5 | 3582 | 177 | 0.460170 | 0.973170 |
| person_macro_within_recording_then_recording_macro | CV-OLS4 | 5 | 3582 | 177 | 0.553703 | 1.141019 |
| recording_macro | CV-OLS8 | 5 | 3582 | 177 | 0.512557 | 1.036147 |
| target_window_weighted | CV-OLS8 | 5 | 3582 | 177 | 0.528687 | 1.068718 |
| person_macro_within_recording_then_recording_macro | CV-OLS8 | 5 | 3582 | 177 | 0.633007 | 1.264180 |
| recording_macro | CV-Median | 5 | 3582 | 177 | 0.520572 | 1.050118 |
| target_window_weighted | CV-Median | 5 | 3582 | 177 | 0.536888 | 1.084020 |
| person_macro_within_recording_then_recording_macro | CV-Median | 5 | 3582 | 177 | 0.643257 | 1.283186 |

录像宏平均：先在每个录像按目标窗口平均，再等权平均5个录像。窗口加权：3582个目标窗口等权。行人口径：先同一录像内同一ID跨窗口平均，再行人等权、录像等权；它用于检查长轨迹重复权重，不构成行人完全独立的统计假设。三种口径排序都未支持替代Last2，不能只用某个子集的宏FDE改善覆盖总体退化。

## 与历史最佳兼容神经对照比较

| 录像 | 冻结强对照 | N | ADE | FDE | 状态 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | PedBase-M0 | 318 | 0.503319 | 1.005511 | frozen_checkpoint_evaluated |
| biwi_hotel | PedBase-BestCompatible | 318 | 0.503319 | 1.005511 | frozen_checkpoint_evaluated |
| biwi_hotel | CVResidual-BestCompatible | 318 | 0.338101 | 0.623851 | frozen_checkpoint_evaluated |
| students001 | PedBase-M0 | — | — | — | not_available |
| students001 | PedBase-BestCompatible | — | — | — | not_available |
| students001 | CVResidual-BestCompatible | — | — | — | not_available |
| students003 | PedBase-M0 | 706 | 0.707297 | 1.524885 | frozen_checkpoint_evaluated |
| students003 | PedBase-BestCompatible | 706 | 0.707297 | 1.524885 | frozen_checkpoint_evaluated |
| students003 | CVResidual-BestCompatible | — | — | — | not_available |
| crowds_zara01 | PedBase-M0 | — | — | — | not_available |
| crowds_zara01 | PedBase-BestCompatible | — | — | — | not_available |
| crowds_zara01 | CVResidual-BestCompatible | — | — | — | not_available |
| crowds_zara02 | PedBase-M0 | 1092 | 0.282905 | 0.630561 | frozen_checkpoint_evaluated |
| crowds_zara02 | PedBase-BestCompatible | 1092 | 0.276696 | 0.619606 | frozen_checkpoint_evaluated |
| crowds_zara02 | CVResidual-BestCompatible | 1092 | 0.276902 | 0.617161 | frozen_checkpoint_evaluated |

M0为Stage3.7原PedBase常学习率对照；students003为Stage3.5同结构A协议PedBase。额外保留历史最佳兼容PedBase（ZARA2为Stage3.6固定衰减版本），避免仅比较弱M0。历史最佳兼容CV残差：HOTEL为Stage3.7归一化M3，ZARA2为Stage3.6原坐标CV-Residual constant；候选列表和选择在本轮评价前冻结，见protocol.json。所有已有模型经源文件、manifest、checkpoint SHA与重评价数值核对；此类历史开发选择仍有选择偏差。

三个平滑模型在HOTEL都胜过Last2、M0及最佳兼容CV残差，但在ZARA2均输给这些强对照，在students003均输给M0。students001/zara01没有合法对应神经checkpoint，students003没有CV残差；表中空缺不是0，没有借用训练见过该录像的模型。严格共享覆盖子集的宏/窗口/行人结果单列reference_matched_aggregates.csv，不能把两录像参考与五录像解析宏平均横比。

## 真实运行耗时

| 录像 | 模型 | 整折forward中位数(s) | 每目标窗口(μs) | 重复次数 |
| --- | --- | --- | --- | --- |
| biwi_hotel | CV-Last2 | 0.034870 | 109.65 | 20 |
| biwi_hotel | CV-OLS4 | 0.039941 | 125.60 | 20 |
| biwi_hotel | CV-OLS8 | 0.039908 | 125.50 | 20 |
| biwi_hotel | CV-Median | 0.037749 | 118.71 | 20 |
| students001 | CV-Last2 | 0.018719 | 16.01 | 20 |
| students001 | CV-OLS4 | 0.021455 | 18.35 | 20 |
| students001 | CV-OLS8 | 0.021458 | 18.36 | 20 |
| students001 | CV-Median | 0.020476 | 17.52 | 20 |
| students003 | CV-Last2 | 0.022795 | 32.29 | 20 |
| students003 | CV-OLS4 | 0.025999 | 36.83 | 20 |
| students003 | CV-OLS8 | 0.026018 | 36.85 | 20 |
| students003 | CV-Median | 0.024789 | 35.11 | 20 |
| crowds_zara01 | CV-Last2 | 0.029851 | 100.51 | 20 |
| crowds_zara01 | CV-OLS4 | 0.034429 | 115.92 | 20 |
| crowds_zara01 | CV-OLS8 | 0.034357 | 115.68 | 20 |
| crowds_zara01 | CV-Median | 0.032282 | 108.69 | 20 |
| crowds_zara02 | CV-Last2 | 0.051282 | 46.96 | 20 |
| crowds_zara02 | CV-OLS4 | 0.058041 | 53.15 | 20 |
| crowds_zara02 | CV-OLS8 | 0.057984 | 53.10 | 20 |
| crowds_zara02 | CV-Median | 0.055693 | 51.00 | 20 |

CPU单线程、batch8、相同缓存观测批次，warmup3遍+20遍forward计时；包含各估计器现有运动预处理，不含IO、分组、GT误差计算。每目标窗口耗时还受同批历史上下文人数及padding影响，不是脱离数据构成的纯速度估计微基准。本阶段解析float64、神经float32，旧CV微小差异仅为舍入；新数据与时间协议未变。
