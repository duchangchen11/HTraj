# 泛化、训练预算与输入分布核查

单位米；Delta=Full−Placebo，负值表示Full好。各模型按最低窗口加权验证K1 latent-prior ADE选checkpoint，FDE及所有辅助指标均配对同一checkpoint。行人权重分析仍使用该checkpoint，不另选最低行人ADE。K1 prior不是预测分布数学期望，训练目标为原重建+MC KL估计，不是精确边缘NLL。

## 强CV与Full同时报告

| 录像 | 权重 | seed/参考 | Full或CV ADE/FDE |
| --- | --- | --- | --- |
| biwi_hotel | target-window | CV-Last2 (解析参考) | 0.329516 / 0.611038 |
| biwi_hotel | equal-person-ID | CV-Last2 (解析参考) | 0.458041 / 0.855327 |
| biwi_hotel | target-window | CV-OLS4 (解析参考) | 0.237814 / 0.438866 |
| biwi_hotel | equal-person-ID | CV-OLS4 (解析参考) | 0.354559 / 0.691444 |
| biwi_hotel | target-window | CV-OLS8 (解析参考) | 0.220144 / 0.408760 |
| biwi_hotel | equal-person-ID | CV-OLS8 (解析参考) | 0.381570 / 0.769618 |
| crowds_zara02 | target-window | CV-Last2 (解析参考) | 0.296306 / 0.667523 |
| crowds_zara02 | equal-person-ID | CV-Last2 (解析参考) | 0.419127 / 0.845329 |
| crowds_zara02 | target-window | CV-OLS4 (解析参考) | 0.331875 / 0.715683 |
| crowds_zara02 | equal-person-ID | CV-OLS4 (解析参考) | 0.472756 / 0.908660 |
| crowds_zara02 | target-window | CV-OLS8 (解析参考) | 0.383175 / 0.787222 |
| crowds_zara02 | equal-person-ID | CV-OLS8 (解析参考) | 0.524198 / 0.960695 |

| 录像 | 权重 | seed | Full ADE/FDE |
| --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.387184 / 0.751521 |
| biwi_hotel | target-window | 42 | 0.435954 / 0.883544 |
| biwi_hotel | target-window | 2026 | 0.417311 / 0.825002 |
| biwi_hotel | equal-person-ID | 17 | 0.736104 / 1.478561 |
| biwi_hotel | equal-person-ID | 42 | 0.882612 / 1.870898 |
| biwi_hotel | equal-person-ID | 2026 | 0.946751 / 1.997171 |
| crowds_zara02 | target-window | 17 | 0.272563 / 0.607513 |
| crowds_zara02 | target-window | 42 | 0.269965 / 0.599033 |
| crowds_zara02 | target-window | 2026 | 0.271458 / 0.605856 |
| crowds_zara02 | equal-person-ID | 17 | 0.359766 / 0.700907 |
| crowds_zara02 | equal-person-ID | 42 | 0.347003 / 0.680388 |
| crowds_zara02 | equal-person-ID | 2026 | 0.349934 / 0.690656 |

## 正常行走目标

| 录像 | 权重 | seed | Full正常行走ADE/FDE | Placebo正常行走ADE/FDE |
| --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.799220 / 1.617945 | 0.866852 / 1.819489 |
| biwi_hotel | target-window | 42 | 0.978133 / 2.070627 | 0.929941 / 1.967218 |
| biwi_hotel | target-window | 2026 | 1.038434 / 2.214783 | 0.920461 / 1.986569 |
| biwi_hotel | equal-person-ID | 17 | 0.830000 / 1.674611 | 0.914616 / 1.909141 |
| biwi_hotel | equal-person-ID | 42 | 1.002005 / 2.134550 | 0.964506 / 2.048838 |
| biwi_hotel | equal-person-ID | 2026 | 1.081797 / 2.295446 | 0.964893 / 2.077840 |
| crowds_zara02 | target-window | 17 | 0.420717 / 0.851958 | 0.441189 / 0.889906 |
| crowds_zara02 | target-window | 42 | 0.415474 / 0.843516 | 0.445081 / 0.907126 |
| crowds_zara02 | target-window | 2026 | 0.410856 / 0.835535 | 0.449203 / 0.906952 |
| crowds_zara02 | equal-person-ID | 17 | 0.445547 / 0.835947 | 0.485625 / 0.920519 |
| crowds_zara02 | equal-person-ID | 42 | 0.433292 / 0.816799 | 0.477957 / 0.910616 |
| crowds_zara02 | equal-person-ID | 2026 | 0.440437 / 0.837042 | 0.493601 / 0.944509 |

| 录像 | 权重 | 参考模型 | 正常行走ADE/FDE | 窗口/ID |
| --- | --- | --- | --- | --- |
| biwi_hotel | target-window | CV-Last2 | 0.533348 / 1.015078 | 91/24 |
| biwi_hotel | equal-person-ID | CV-Last2 | 0.504479 / 0.947615 | 91/24 |
| biwi_hotel | target-window | CV-OLS4 | 0.412659 / 0.803163 | 91/24 |
| biwi_hotel | equal-person-ID | CV-OLS4 | 0.393993 / 0.777200 | 91/24 |
| biwi_hotel | target-window | CV-OLS8 | 0.443563 / 0.872085 | 91/24 |
| biwi_hotel | equal-person-ID | CV-OLS8 | 0.425763 / 0.870408 | 91/24 |
| crowds_zara02 | target-window | CV-Last2 | 0.504078 / 1.049855 | 422/33 |
| crowds_zara02 | equal-person-ID | CV-Last2 | 0.543033 / 1.061006 | 422/33 |
| crowds_zara02 | target-window | CV-OLS4 | 0.570361 / 1.129509 | 422/33 |
| crowds_zara02 | equal-person-ID | CV-OLS4 | 0.633714 / 1.185681 | 422/33 |
| crowds_zara02 | target-window | CV-OLS8 | 0.660416 / 1.241451 | 422/33 |
| crowds_zara02 | equal-person-ID | CV-OLS8 | 0.795931 / 1.431489 | 422/33 |

HOTEL/target-window正常行走：Full在3/3 seed的ADE高于同目标CV-Last2（0.533348m）；其它强CV逐项列出。
HOTEL/equal-person-ID正常行走：Full在3/3 seed的ADE高于同目标CV-Last2（0.504479m）；其它强CV逐项列出。

## 历史静止目标的绝对预测能力

| 录像 | 权重 | seed | Full历史静止ADE/FDE | Placebo历史静止ADE/FDE |
| --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.182059 / 0.364103 | 0.145800 / 0.277568 |
| biwi_hotel | target-window | 42 | 0.146458 / 0.297664 | 0.175033 / 0.375174 |
| biwi_hotel | target-window | 2026 | 0.114239 / 0.205010 | 0.151113 / 0.292868 |
| biwi_hotel | equal-person-ID | 17 | 0.180139 / 0.349868 | 0.147092 / 0.266879 |
| biwi_hotel | equal-person-ID | 42 | 0.160507 / 0.317173 | 0.188950 / 0.393164 |
| biwi_hotel | equal-person-ID | 2026 | 0.114548 / 0.189874 | 0.150012 / 0.277286 |
| crowds_zara02 | target-window | 17 | 0.120318 / 0.329922 | 0.113873 / 0.316446 |
| crowds_zara02 | target-window | 42 | 0.121351 / 0.324519 | 0.116657 / 0.320679 |
| crowds_zara02 | target-window | 2026 | 0.126064 / 0.338053 | 0.118294 / 0.324034 |
| crowds_zara02 | equal-person-ID | 17 | 0.201803 / 0.567512 | 0.194784 / 0.553910 |
| crowds_zara02 | equal-person-ID | 42 | 0.202480 / 0.566738 | 0.196376 / 0.556067 |
| crowds_zara02 | equal-person-ID | 2026 | 0.212836 / 0.591600 | 0.202237 / 0.568241 |

| 录像 | 权重 | 参考模型 | 历史静止ADE/FDE | 窗口/ID |
| --- | --- | --- | --- | --- |
| biwi_hotel | target-window | CV-Last2 | 0.101309 / 0.189901 | 106/4 |
| biwi_hotel | equal-person-ID | CV-Last2 | 0.120290 / 0.219163 | 106/4 |
| biwi_hotel | target-window | CV-OLS4 | 0.085026 / 0.156712 | 106/4 |
| biwi_hotel | equal-person-ID | CV-OLS4 | 0.096447 / 0.169944 | 106/4 |
| biwi_hotel | target-window | CV-OLS8 | 0.077664 / 0.142256 | 106/4 |
| biwi_hotel | equal-person-ID | CV-OLS8 | 0.090487 / 0.156530 | 106/4 |
| crowds_zara02 | target-window | CV-Last2 | 0.104459 / 0.298659 | 579/9 |
| crowds_zara02 | equal-person-ID | CV-Last2 | 0.186302 / 0.539123 | 579/9 |
| crowds_zara02 | target-window | CV-OLS4 | 0.105407 / 0.300145 | 579/9 |
| crowds_zara02 | equal-person-ID | CV-OLS4 | 0.187559 / 0.540912 | 579/9 |
| crowds_zara02 | target-window | CV-OLS8 | 0.107625 / 0.303936 | 579/9 |
| crowds_zara02 | equal-person-ID | CV-OLS8 | 0.191603 / 0.547614 | 579/9 |

biwi_hotel/target-window历史静止：Full在3/3 seed的ADE高于固定CV-Last2（0.101309m）；少样本=True。这是相对CV的退化，与04报告相对Placebo的差值分别解释。
biwi_hotel/equal-person-ID历史静止：Full在2/3 seed的ADE高于固定CV-Last2（0.120290m）；少样本=True。这是相对CV的退化，与04报告相对Placebo的差值分别解释。
crowds_zara02/target-window历史静止：Full在3/3 seed的ADE高于固定CV-Last2（0.104459m）；少样本=True。这是相对CV的退化，与04报告相对Placebo的差值分别解释。
crowds_zara02/equal-person-ID历史静止：Full在3/3 seed的ADE高于固定CV-Last2（0.186302m）；少样本=True。这是相对CV的退化，与04报告相对Placebo的差值分别解释。

## 训练稳定性与后20轮

- biwi_hotel/Full：seed17/42/2026的best分别为3 / 2 / 2轮；前5轮最优3/3，最后10轮最优0/3。
- biwi_hotel/PlaceboSocial：seed17/42/2026的best分别为3 / 2 / 2轮；前5轮最优3/3，最后10轮最优0/3。
- crowds_zara02/Full：seed17/42/2026的best分别为30 / 29 / 28轮；前5轮最优0/3，最后10轮最优0/3。
- crowds_zara02/PlaceboSocial：seed17/42/2026的best分别为76 / 57 / 42轮；前5轮最优0/3，最后10轮最优0/3。

| 录像 | seed | 模型 | best≤5 | best≥91 | 训练K1(best/last) | 验证K1(best/last) | 后20轮训练loss均值变化 | 后20轮验证ADE均值变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | 17 | Full | True | False | 0.513493 / 0.114646 | 0.387184 / 0.646965 | -0.001385 | -0.016050 |
| biwi_hotel | 17 | PlaceboSocial | True | False | 0.497393 / 0.378626 | 0.392742 / 0.562979 | -0.000511 | -0.042908 |
| crowds_zara02 | 17 | Full | False | False | 0.455903 / 0.115562 | 0.272563 / 0.329019 | -0.001311 | 0.001558 |
| crowds_zara02 | 17 | PlaceboSocial | False | False | 0.443863 / 0.397122 | 0.279180 / 0.306436 | -0.000560 | 0.010052 |
| biwi_hotel | 42 | Full | True | False | 0.514705 / 0.174431 | 0.435954 / 0.690960 | -0.001029 | -0.013215 |
| biwi_hotel | 42 | PlaceboSocial | True | False | 0.529050 / 0.378081 | 0.444768 / 0.545751 | -0.000527 | -0.005683 |
| crowds_zara02 | 42 | Full | False | False | 0.446182 / 0.117216 | 0.269965 / 0.346779 | -0.001213 | -0.000285 |
| crowds_zara02 | 42 | PlaceboSocial | False | False | 0.457921 / 0.397103 | 0.280692 / 0.295310 | -0.000595 | 0.000453 |
| biwi_hotel | 2026 | Full | True | False | 0.507115 / 0.118682 | 0.417311 / 0.691426 | -0.001126 | -0.002292 |
| biwi_hotel | 2026 | PlaceboSocial | True | False | 0.507001 / 0.367409 | 0.406706 / 0.781096 | -0.000564 | 0.015524 |
| crowds_zara02 | 2026 | Full | False | False | 0.451415 / 0.115364 | 0.271458 / 0.351435 | -0.001392 | 0.004840 |
| crowds_zara02 | 2026 | PlaceboSocial | False | False | 0.473541 / 0.391680 | 0.283085 / 0.301555 | -0.000555 | 0.001221 |

## 冻结合成分布差异

| 录像 | split | 上下文 | 统计量 | 均值 | P95 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | train | real | count | 4.094322 | 9.000000 |
| biwi_hotel | train | real | distance | 1.221782 | 1.930484 |
| biwi_hotel | train | real | speed | 0.542571 | 1.240102 |
| biwi_hotel | train | synthetic | count | 4.000000 | 4.000000 |
| biwi_hotel | train | synthetic | distance | 0.802829 | 0.983062 |
| biwi_hotel | train | synthetic | speed | 0.573771 | 1.339323 |
| biwi_hotel | validation | real | count | 1.023585 | 3.000000 |
| biwi_hotel | validation | real | distance | 1.073755 | 1.948795 |
| biwi_hotel | validation | real | speed | 0.987908 | 1.985101 |
| biwi_hotel | validation | synthetic | count | 4.000000 | 4.000000 |
| biwi_hotel | validation | synthetic | distance | 0.800516 | 0.920511 |
| biwi_hotel | validation | synthetic | speed | 0.401686 | 1.515062 |
| crowds_zara02 | train | real | count | 4.296225 | 10.000000 |
| crowds_zara02 | train | real | distance | 1.236134 | 1.932167 |
| crowds_zara02 | train | real | speed | 0.545289 | 1.205088 |
| crowds_zara02 | train | synthetic | count | 4.000000 | 4.000000 |
| crowds_zara02 | train | synthetic | distance | 0.802485 | 0.981522 |
| crowds_zara02 | train | synthetic | speed | 0.570537 | 1.309668 |
| crowds_zara02 | validation | real | count | 2.459592 | 5.000000 |
| crowds_zara02 | validation | real | distance | 1.091435 | 1.899037 |
| crowds_zara02 | validation | real | speed | 0.380305 | 1.309494 |
| crowds_zara02 | validation | synthetic | count | 4.000000 | 4.000000 |
| crowds_zara02 | validation | synthetic | distance | 0.801606 | 0.977408 |
| crowds_zara02 | validation | synthetic | speed | 0.442013 | 1.388092 |

后20轮变化指91–100均值减81–90均值。best很早且后期训练loss下降、验证ADE上升是经验泛化退化迹象，不等于证明具体因果原因。训练与验证目标运动构成不同，训练K1与验证K1差距不能脱离录像分布解释；原始VAE目标也不保证确定性K1单调改善。best接近100需记预算截断，绝不因此延长。

三种CV均在同一冻结目标集合上重新核对过GT和误差，主解析参考固定CV-Last2。事后逐录像/权重挑选最低CV仅是“解析基线上界参考”，不是可部署新模型。社会归因与绝对预测能力是两个问题：即使Full优于Placebo，HOTEL如果仍弱于CV必须保留该事实。K20与确定性CV预测预算不同，不混为K1比较。

上下文分布为Stage5.4全冻结训练/验证目标的历史观测统计，SHA逐值复用；输入本身不随训练seed改变，不是新增分布匹配实验。Placebo始终4个近邻、完整8帧、恒定初始偏移；真实邻居数量、距离、进入和速度可观测性不同。context_mask_summary.csv记录实际历史mask，synthetic_future_slots_all_padding列只针对合成上下文（real行为不由该列False推断）。不能因三seed一致便消除这些结构/输入分布混杂。

上下文表count为每历史帧有效邻居数，distance单位米，speed单位米/秒（官方逐步位移除以0.4秒）；count、distance、speed按各自有效观测集合汇总，不能把这些观测条目当作独立行人。

ETH未评价；两个录像为历史多轮曝光的开发数据。录像隔离保护每个run的训练/验证支持，不意味着新的独立科研确认。
