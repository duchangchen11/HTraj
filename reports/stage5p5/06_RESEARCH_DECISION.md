# 研究判断

seed17毫米级优势未在所有seed、录像和权重下重复。当前社会内容收益不稳定或不普遍，停止继续基于该收益开发复杂社会交互模块；不能用三个seed均值或某个改善分组挽救失败的稳定性结论。

- biwi_hotel/target-window：K1 ADE改善2/3 seed，FDE改善2/3 seed。该录像与权重内未达到ADE/FDE三seed全部改善。
- biwi_hotel/equal-person-ID：K1 ADE改善1/3 seed，FDE改善1/3 seed。该录像与权重内未达到ADE/FDE三seed全部改善。
- crowds_zara02/target-window：K1 ADE改善3/3 seed，FDE改善3/3 seed。该录像与权重内方向重复，但不能代替两个录像同时满足门槛。
- crowds_zara02/equal-person-ID：K1 ADE改善3/3 seed，FDE改善3/3 seed。该录像与权重内方向重复，但不能代替两个录像同时满足门槛。

单位米；Delta=Full−Placebo，负值表示Full好。各模型按最低窗口加权验证K1 latent-prior ADE选checkpoint，FDE及所有辅助指标均配对同一checkpoint。行人权重分析仍使用该checkpoint，不另选最低行人ADE。K1 prior不是预测分布数学期望，训练目标为原重建+MC KL估计，不是精确边缘NLL。

| 录像 | 权重 | seed | Full K1 ADE/FDE | Placebo K1 ADE/FDE | Delta ADE/FDE |
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

| 录像 | 权重 | 3seed ADE差值 | 3seed FDE差值 | ADE反转 | FDE反转 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | -0.005557 / -0.008814 / 0.010605 | -0.013287 / -0.033036 / 0.010344 | True | True |
| biwi_hotel | equal-person-ID | -0.068874 / 0.026877 / 0.096308 | -0.188631 / 0.058258 / 0.178011 | True | True |
| crowds_zara02 | target-window | -0.006617 / -0.010728 / -0.011627 | -0.011393 / -0.026805 / -0.020792 | False | False |
| crowds_zara02 | equal-person-ID | -0.023637 / -0.033877 / -0.033571 | -0.055714 / -0.070829 / -0.063318 | False | False |

| 录像 | 权重 | 平均ΔADE±样本SD | Full ADE跨seed SD | Placebo ADE跨seed SD | 平均ΔFDE±样本SD |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | -0.001256 ± 0.010400 | 0.024609 | 0.026927 | -0.011993 ± 0.021719 |
| biwi_hotel | equal-person-ID | 0.018104 ± 0.082939 | 0.107974 | 0.027903 | 0.015880 ± 0.186959 |
| crowds_zara02 | target-window | -0.009657 ± 0.002671 | 0.001304 | 0.001969 | -0.019663 ± 0.007768 |
| crowds_zara02 | equal-person-ID | -0.030362 ± 0.005826 | 0.006685 | 0.001487 | -0.063287 ± 0.007558 |
| two-recording-macro | target-window | -0.005457 ± 0.004662 | 0.011669 | 0.013629 | -0.015828 ± 0.012712 |
| two-recording-macro | equal-person-ID | -0.006129 ± 0.038879 | 0.051118 | 0.013555 | -0.023704 ± 0.091018 |


## 收益幅度与自然波动

- biwi_hotel/target-window：|平均ΔADE|=0.001256m，配对差值样本SD=0.010400m，Full/Placebo原始ADE跨seed SD=0.024609/0.026927m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。
- biwi_hotel/equal-person-ID：|平均ΔADE|=0.018104m，配对差值样本SD=0.082939m，Full/Placebo原始ADE跨seed SD=0.107974/0.027903m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。
- crowds_zara02/target-window：|平均ΔADE|=0.009657m，配对差值样本SD=0.002671m，Full/Placebo原始ADE跨seed SD=0.001304/0.001969m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。
- crowds_zara02/equal-person-ID：|平均ΔADE|=0.030362m，配对差值样本SD=0.005826m，Full/Placebo原始ADE跨seed SD=0.006685/0.001487m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。
- two-recording-macro/target-window：|平均ΔADE|=0.005457m，配对差值样本SD=0.004662m，Full/Placebo原始ADE跨seed SD=0.011669/0.013629m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。
- two-recording-macro/equal-person-ID：|平均ΔADE|=0.006129m，配对差值样本SD=0.038879m，Full/Placebo原始ADE跨seed SD=0.051118/0.013555m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。

K1条件bootstrap的24个录像/seed/权重/指标区间中，11个包含0。该计数只作不确定性描述，不是预设稳定性门槛或多重显著性结论。

## 辅助采样指标单独判断

| 指标族 | 录像 | 权重 | 平均ΔADE/ΔFDE | ADE/FDE改善seed数 |
| --- | --- | --- | --- | --- |
| K1-first-sample | biwi_hotel | target-window | 0.013569 / 0.025947 | 1/3 ; 1/3 |
| K1-first-sample | biwi_hotel | equal-person-ID | 0.050464 / 0.097585 | 1/3 ; 1/3 |
| K1-first-sample | crowds_zara02 | target-window | -0.009527 / -0.006867 | 3/3 ; 2/3 |
| K1-first-sample | crowds_zara02 | equal-person-ID | -0.046793 / -0.068728 | 3/3 ; 3/3 |
| K1-first-sample | two-recording-macro | target-window | 0.002021 / 0.009540 | 1/3 ; 1/3 |
| K1-first-sample | two-recording-macro | equal-person-ID | 0.001835 / 0.014429 | 2/3 ; 1/3 |
| K20-independent-minima | biwi_hotel | target-window | 0.012357 / 0.025148 | 1/3 ; 1/3 |
| K20-independent-minima | biwi_hotel | equal-person-ID | 0.036252 / 0.073041 | 1/3 ; 1/3 |
| K20-independent-minima | crowds_zara02 | target-window | 0.003590 / 0.007645 | 1/3 ; 1/3 |
| K20-independent-minima | crowds_zara02 | equal-person-ID | -0.009233 / -0.011103 | 3/3 ; 2/3 |
| K20-independent-minima | two-recording-macro | target-window | 0.007973 / 0.016396 | 1/3 ; 1/3 |
| K20-independent-minima | two-recording-macro | equal-person-ID | 0.013509 / 0.030969 | 1/3 ; 1/3 |

仅当同一指标族各seed方向均一致时才记录该辅助指标重复；不将主K1结论自动推广到K20。

## 四项历史条件问题

- biwi_hotel/target-window/speed_normal：ADE改善1/3，FDE改善1/3；窗口91，ID24，少样本标记False。
- biwi_hotel/target-window/speed_stationary：ADE改善2/3，FDE改善2/3；窗口106，ID4，少样本标记True。
- biwi_hotel/target-window/last_neighbors_6plus：ADE改善0/0，FDE改善0/0；窗口0，ID0，少样本标记True。
- biwi_hotel/target-window/density_3plus：ADE改善0/3，FDE改善1/3；窗口6，ID2，少样本标记True。
- biwi_hotel/equal-person-ID/speed_normal：ADE改善1/3，FDE改善1/3；窗口91，ID24，少样本标记False。
- biwi_hotel/equal-person-ID/speed_stationary：ADE改善2/3，FDE改善2/3；窗口106，ID4，少样本标记True。
- biwi_hotel/equal-person-ID/last_neighbors_6plus：ADE改善0/0，FDE改善0/0；窗口0，ID0，少样本标记True。
- biwi_hotel/equal-person-ID/density_3plus：ADE改善0/3，FDE改善1/3；窗口6，ID2，少样本标记True。
- crowds_zara02/target-window/speed_normal：ADE改善3/3，FDE改善3/3；窗口422，ID33，少样本标记False。
- crowds_zara02/target-window/speed_stationary：ADE改善0/3，FDE改善0/3；窗口579，ID9，少样本标记True。
- crowds_zara02/target-window/last_neighbors_6plus：ADE改善3/3，FDE改善2/3；窗口50，ID9，少样本标记True。
- crowds_zara02/target-window/density_3plus：ADE改善3/3，FDE改善3/3；窗口336，ID14，少样本标记False。
- crowds_zara02/equal-person-ID/speed_normal：ADE改善3/3，FDE改善3/3；窗口422，ID33，少样本标记False。
- crowds_zara02/equal-person-ID/speed_stationary：ADE改善0/3，FDE改善0/3；窗口579，ID9，少样本标记True。
- crowds_zara02/equal-person-ID/last_neighbors_6plus：ADE改善2/3，FDE改善2/3；窗口50，ID9，少样本标记True。
- crowds_zara02/equal-person-ID/density_3plus：ADE改善3/3，FDE改善3/3；窗口336，ID14，少样本标记False。

本阶段只允许并实际执行8个新增100epoch run，未训练NoSocial/EgoCapacity，未改变Placebo、attention、结构、loss、数据协议或评价batch。seed42成绩未改变seed2026配置，未加seed或200epoch延长。4个seed17来源冻结而非重复训练，全部12组合单独列出。

Stage5.4原始单seed结果和报告保持不变。本阶段新增证据用于更新其跨随机训练可推广性解释，不回写旧指标，也不把旧阶段的初步迹象升级为未经验证的独立因果结论。

平均配对差值必须与配对SD、各模型跨seed SD及每个seed原始差值一同解释。共同难易波动可能被配对抵消，因此不能只凭独立模型SD较大否定一致的配对方向；也不能凭n=3的平均小差值宣布统计确认。K1首样本/K20的评价噪声随训练seed按预注册键变化，其波动不是纯初始化波动。

bootstrap固定最佳checkpoint和评价噪声，只重采样人ID，未重做训练或checkpoint选择；不修复选模偏差、历史开发曝光、行人间相互依赖或独立录像不足。HOTEL28、ZARA2 35 ID，不能把1410重叠窗口或三次训练当大量独立样本。

HOTEL与三种强CV、正常行走绝对误差、后20轮泛化退化在05报告完整展示。K1与K20分开解释，不以K20替代社会K1稳定性。普通VAE、伪邻居和多seed统计本身不是论文新方法。

本阶段结束，暂不设计第四章新网络，不自行开展更多训练。若稳定性未获支持，明确停止由这项毫米差驱动的复杂社会模块路线；成熟SocialVAE可保留为冻结公开基线，后续研究问题需新的独立指令。即使稳定性通过，也只支持进一步研究设计，不等于已获得方法创新。
