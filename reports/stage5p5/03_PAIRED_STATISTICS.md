# 配对统计与不确定性

单位米；Delta=Full−Placebo，负值表示Full好。各模型按最低窗口加权验证K1 latent-prior ADE选checkpoint，FDE及所有辅助指标均配对同一checkpoint。行人权重分析仍使用该checkpoint，不另选最低行人ADE。K1 prior不是预测分布数学期望，训练目标为原重建+MC KL估计，不是精确边缘NLL。


## K1-prior：全部seed原始指标

| 录像 | 权重 | seed | Full ADE/FDE | Placebo ADE/FDE | Delta ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.387184 / 0.751521 | 0.392742 / 0.764808 | -0.005557 / -0.013287 |
| biwi_hotel | target-window | 42 | 0.435954 / 0.883544 | 0.444768 / 0.916580 | -0.008814 / -0.033036 |
| biwi_hotel | target-window | 2026 | 0.417311 / 0.825002 | 0.406706 / 0.814658 | 0.010605 / 0.010344 |
| biwi_hotel | equal-person-ID | 17 | 0.736104 / 1.478561 | 0.804978 / 1.667192 | -0.068874 / -0.188631 |
| biwi_hotel | equal-person-ID | 42 | 0.882612 / 1.870898 | 0.855735 / 1.812639 | 0.026877 / 0.058258 |
| biwi_hotel | equal-person-ID | 2026 | 0.946751 / 1.997171 | 0.850443 / 1.819159 | 0.096308 / 0.178011 |
| crowds_zara02 | target-window | 17 | 0.272563 / 0.607513 | 0.279180 / 0.618906 | -0.006617 / -0.011393 |
| crowds_zara02 | target-window | 42 | 0.269965 / 0.599033 | 0.280692 / 0.625838 | -0.010728 / -0.026805 |
| crowds_zara02 | target-window | 2026 | 0.271458 / 0.605856 | 0.283085 / 0.626648 | -0.011627 / -0.020792 |
| crowds_zara02 | equal-person-ID | 17 | 0.359766 / 0.700907 | 0.383404 / 0.756621 | -0.023637 / -0.055714 |
| crowds_zara02 | equal-person-ID | 42 | 0.347003 / 0.680388 | 0.380880 / 0.751217 | -0.033877 / -0.070829 |
| crowds_zara02 | equal-person-ID | 2026 | 0.349934 / 0.690656 | 0.383506 / 0.753974 | -0.033571 / -0.063318 |

## K1-first-sample：全部seed原始指标

| 录像 | 权重 | seed | Full ADE/FDE | Placebo ADE/FDE | Delta ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.530319 / 1.075447 | 0.544893 / 1.112961 | -0.014575 / -0.037515 |
| biwi_hotel | target-window | 42 | 0.603631 / 1.221764 | 0.576458 / 1.161059 | 0.027173 / 0.060704 |
| biwi_hotel | target-window | 2026 | 0.556816 / 1.136155 | 0.528707 / 1.081503 | 0.028109 / 0.054652 |
| biwi_hotel | equal-person-ID | 17 | 0.975382 / 2.039130 | 0.994675 / 2.093751 | -0.019294 / -0.054622 |
| biwi_hotel | equal-person-ID | 42 | 1.043457 / 2.261190 | 0.984085 / 2.127658 | 0.059372 / 0.133532 |
| biwi_hotel | equal-person-ID | 2026 | 1.027643 / 2.198855 | 0.916330 / 1.985010 | 0.111313 / 0.213845 |
| crowds_zara02 | target-window | 17 | 0.409855 / 0.947254 | 0.410256 / 0.918173 | -0.000400 / 0.029081 |
| crowds_zara02 | target-window | 42 | 0.412511 / 0.934268 | 0.427352 / 0.959602 | -0.014841 / -0.025334 |
| crowds_zara02 | target-window | 2026 | 0.414610 / 0.928649 | 0.427949 / 0.952999 | -0.013339 / -0.024349 |
| crowds_zara02 | equal-person-ID | 17 | 0.550585 / 1.192296 | 0.567618 / 1.203511 | -0.017033 / -0.011215 |
| crowds_zara02 | equal-person-ID | 42 | 0.552187 / 1.205486 | 0.623339 / 1.296306 | -0.071153 / -0.090821 |
| crowds_zara02 | equal-person-ID | 2026 | 0.552858 / 1.149176 | 0.605051 / 1.253323 | -0.052194 / -0.104147 |

## K20-independent-minima：全部seed原始指标

| 录像 | 权重 | seed | Full ADE/FDE | Placebo ADE/FDE | Delta ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 17 | 0.202466 / 0.325331 | 0.223376 / 0.374198 | -0.020910 / -0.048866 |
| biwi_hotel | target-window | 42 | 0.269654 / 0.468072 | 0.240765 / 0.411456 | 0.028889 / 0.056616 |
| biwi_hotel | target-window | 2026 | 0.257420 / 0.455440 | 0.228329 / 0.387747 | 0.029091 / 0.067694 |
| biwi_hotel | equal-person-ID | 17 | 0.421215 / 0.803645 | 0.479347 / 0.956199 | -0.058133 / -0.152554 |
| biwi_hotel | equal-person-ID | 42 | 0.577417 / 1.167315 | 0.498592 / 0.995097 | 0.078825 / 0.172218 |
| biwi_hotel | equal-person-ID | 2026 | 0.559038 / 1.156086 | 0.470975 / 0.956628 | 0.088063 / 0.199459 |
| crowds_zara02 | target-window | 17 | 0.157530 / 0.304836 | 0.146936 / 0.280722 | 0.010594 / 0.024113 |
| crowds_zara02 | target-window | 42 | 0.149610 / 0.285548 | 0.151430 / 0.282442 | -0.001820 / 0.003106 |
| crowds_zara02 | target-window | 2026 | 0.156347 / 0.295992 | 0.154351 / 0.300276 | 0.001996 / -0.004284 |
| crowds_zara02 | equal-person-ID | 17 | 0.187058 / 0.309672 | 0.193721 / 0.307688 | -0.006663 / 0.001984 |
| crowds_zara02 | equal-person-ID | 42 | 0.184002 / 0.296369 | 0.198344 / 0.310070 | -0.014342 / -0.013701 |
| crowds_zara02 | equal-person-ID | 2026 | 0.190112 / 0.292819 | 0.196807 / 0.314410 | -0.006695 / -0.021592 |

## K1-prior：跨seed与两录像宏平均

| 录像/宏平均 | 权重 | ΔADE均值±样本SD | ΔFDE均值±样本SD | Full/Placebo ADE的跨seed SD | ADE/FDE改善seed数 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | -0.001256 ± 0.010400 | -0.011993 ± 0.021719 | 0.024609 / 0.026927 | 2/3 ; 2/3 |
| biwi_hotel | equal-person-ID | 0.018104 ± 0.082939 | 0.015880 ± 0.186959 | 0.107974 / 0.027903 | 1/3 ; 1/3 |
| crowds_zara02 | target-window | -0.009657 ± 0.002671 | -0.019663 ± 0.007768 | 0.001304 / 0.001969 | 3/3 ; 3/3 |
| crowds_zara02 | equal-person-ID | -0.030362 ± 0.005826 | -0.063287 ± 0.007558 | 0.006685 / 0.001487 | 3/3 ; 3/3 |
| two-recording-macro | target-window | -0.005457 ± 0.004662 | -0.015828 ± 0.012712 | 0.011669 / 0.013629 | 3/3 ; 3/3 |
| two-recording-macro | equal-person-ID | -0.006129 ± 0.038879 | -0.023704 ± 0.091018 | 0.051118 / 0.013555 | 2/3 ; 2/3 |

## K1-first-sample：跨seed与两录像宏平均

| 录像/宏平均 | 权重 | ΔADE均值±样本SD | ΔFDE均值±样本SD | Full/Placebo ADE的跨seed SD | ADE/FDE改善seed数 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 0.013569 ± 0.024378 | 0.025947 ± 0.055043 | 0.037123 / 0.024285 | 1/3 ; 1/3 |
| biwi_hotel | equal-person-ID | 0.050464 ± 0.065757 | 0.097585 ± 0.137796 | 0.035627 / 0.042507 | 1/3 ; 1/3 |
| crowds_zara02 | target-window | -0.009527 ± 0.007939 | -0.006867 ± 0.031136 | 0.002383 / 0.010047 | 3/3 ; 2/3 |
| crowds_zara02 | equal-person-ID | -0.046793 ± 0.027461 | -0.068728 ± 0.050251 | 0.001168 / 0.028403 | 3/3 ; 3/3 |
| two-recording-macro | target-window | 0.002021 ± 0.008257 | 0.009540 ± 0.011981 | 0.019091 / 0.013835 | 1/3 ; 1/3 |
| two-recording-macro | equal-person-ID | 0.001835 ± 0.024782 | 0.014429 ± 0.044292 | 0.018324 / 0.021519 | 2/3 ; 1/3 |

## K20-independent-minima：跨seed与两录像宏平均

| 录像/宏平均 | 权重 | ΔADE均值±样本SD | ΔFDE均值±样本SD | Full/Placebo ADE的跨seed SD | ADE/FDE改善seed数 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | 0.012357 ± 0.028810 | 0.025148 ± 0.064337 | 0.035786 / 0.008959 | 1/3 ; 1/3 |
| biwi_hotel | equal-person-ID | 0.036252 ± 0.081870 | 0.073041 ± 0.195845 | 0.085374 / 0.014161 | 1/3 ; 1/3 |
| crowds_zara02 | target-window | 0.003590 ± 0.006358 | 0.007645 ± 0.014733 | 0.004272 / 0.003735 | 1/3 ; 1/3 |
| crowds_zara02 | equal-person-ID | -0.009233 ± 0.004424 | -0.011103 ± 0.012001 | 0.003055 / 0.002354 | 3/3 ; 2/3 |
| two-recording-macro | target-window | 0.007973 ± 0.011416 | 0.016396 ± 0.024935 | 0.016374 / 0.005486 | 1/3 ; 1/3 |
| two-recording-macro | equal-person-ID | 0.013509 ± 0.039980 | 0.030969 ± 0.092146 | 0.042549 / 0.007766 | 1/3 ; 1/3 |

## K1簇bootstrap

| 录像 | seed | 权重 | 指标 | 点差值 | 95%簇bootstrap区间 | 含0 |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | 17 | target-window | ADE | -0.005557 | -0.051602 ~ 0.013267 | True |
| biwi_hotel | 17 | target-window | FDE | -0.013287 | -0.137654 ~ 0.031710 | True |
| biwi_hotel | 17 | equal-person-ID | ADE | -0.068874 | -0.116386 ~ -0.019229 | False |
| biwi_hotel | 17 | equal-person-ID | FDE | -0.188631 | -0.280476 ~ -0.097603 | False |
| crowds_zara02 | 17 | target-window | ADE | -0.006617 | -0.017677 ~ 0.002037 | True |
| crowds_zara02 | 17 | target-window | FDE | -0.011393 | -0.036800 ~ 0.011962 | True |
| crowds_zara02 | 17 | equal-person-ID | ADE | -0.023637 | -0.049666 ~ -0.002628 | False |
| crowds_zara02 | 17 | equal-person-ID | FDE | -0.055714 | -0.111816 ~ -0.009547 | False |
| biwi_hotel | 42 | target-window | ADE | -0.008814 | -0.030907 ~ 0.041758 | True |
| biwi_hotel | 42 | target-window | FDE | -0.033036 | -0.079301 ~ 0.073375 | True |
| biwi_hotel | 42 | equal-person-ID | ADE | 0.026877 | -0.030109 ~ 0.086234 | True |
| biwi_hotel | 42 | equal-person-ID | FDE | 0.058258 | -0.046877 ~ 0.165471 | True |
| crowds_zara02 | 42 | target-window | ADE | -0.010728 | -0.023393 ~ -0.001676 | False |
| crowds_zara02 | 42 | target-window | FDE | -0.026805 | -0.053125 ~ -0.007069 | False |
| crowds_zara02 | 42 | equal-person-ID | ADE | -0.033877 | -0.070535 ~ -0.007326 | False |
| crowds_zara02 | 42 | equal-person-ID | FDE | -0.070829 | -0.146416 ~ -0.011974 | False |
| biwi_hotel | 2026 | target-window | ADE | 0.010605 | -0.015295 ~ 0.077383 | True |
| biwi_hotel | 2026 | target-window | FDE | 0.010344 | -0.043194 ~ 0.147102 | True |
| biwi_hotel | 2026 | equal-person-ID | ADE | 0.096308 | 0.046107 ~ 0.150740 | False |
| biwi_hotel | 2026 | equal-person-ID | FDE | 0.178011 | 0.067012 ~ 0.299339 | False |
| crowds_zara02 | 2026 | target-window | ADE | -0.011627 | -0.027544 ~ -0.000957 | False |
| crowds_zara02 | 2026 | target-window | FDE | -0.020792 | -0.052792 ~ 0.001301 | True |
| crowds_zara02 | 2026 | equal-person-ID | ADE | -0.033571 | -0.049344 ~ -0.016940 | False |
| crowds_zara02 | 2026 | equal-person-ID | FDE | -0.063318 | -0.103707 ~ -0.024438 | False |

每个固定seed/录像真实执行2000次ID整簇有放回抽样，保留每个ID所有重叠窗口及重复次数。窗口加权按抽样窗口总数加权，行人权重先在ID内部平均；同一抽样矩阵用于两指标、两权重和三个指标族。原始抽样ID、全部分布、稳定种子、72个区间记录见bootstrap/。CI为percentile 2.5%/97.5%，不是p值或确认性显著性检验。

HOTEL28 ID、ZARA2 35 ID，同行人窗口强相关，不同行人亦可能互动相关；该cluster方法只部分处理相关性。三次训练seed不是三个新数据集，两个录像不是大量独立场景。区间条件于已选择best的模型与固定评价噪声，不重做训练/checkpoint选择，不能修正验证选模或历史开发曝光偏差。

样本SD用ddof1，n=3估计本身不稳定。独立模型误差跨seed SD与配对差值SD同时展示：前者含共同初始化难易波动，后者针对配对效应。训练seed同时决定初始化、预注册样本顺序及训练潜变量噪声；这是整体随机训练稳定性验证，不能归因为只改变初始权重的单因素实验。主K1确定性无需评价采样；K1首样本/K20跨seed还包含预注册评价噪声变化，不能将其SD全解释为初始化影响。

两录像宏平均先分别计算录像内权重指标，再各占1/2；不是把1410窗口混池。原始三seed分别列出，不以均值隐藏方向反转。NoSocial/EgoCapacity仅seed17历史参考，未纳入这些均值、SD或bootstrap。
