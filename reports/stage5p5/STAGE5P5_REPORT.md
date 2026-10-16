# Stage5.5：社会内容收益的跨初始化核查

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

| 录像 | 权重 | 平均ΔADE±样本SD | Full ADE跨seed SD | Placebo ADE跨seed SD | 平均ΔFDE±样本SD |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | -0.001256 ± 0.010400 | 0.024609 | 0.026927 | -0.011993 ± 0.021719 |
| biwi_hotel | equal-person-ID | 0.018104 ± 0.082939 | 0.107974 | 0.027903 | 0.015880 ± 0.186959 |
| crowds_zara02 | target-window | -0.009657 ± 0.002671 | 0.001304 | 0.001969 | -0.019663 ± 0.007768 |
| crowds_zara02 | equal-person-ID | -0.030362 ± 0.005826 | 0.006685 | 0.001487 | -0.063287 ± 0.007558 |
| two-recording-macro | target-window | -0.005457 ± 0.004662 | 0.011669 | 0.013629 | -0.015828 ± 0.012712 |
| two-recording-macro | equal-person-ID | -0.006129 ± 0.038879 | 0.051118 | 0.013555 | -0.023704 ± 0.091018 |

| 录像 | 权重 | 3seed ADE差值 | 3seed FDE差值 | ADE反转 | FDE反转 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | target-window | -0.005557 / -0.008814 / 0.010605 | -0.013287 / -0.033036 / 0.010344 | True | True |
| biwi_hotel | equal-person-ID | -0.068874 / 0.026877 / 0.096308 | -0.188631 / 0.058258 / 0.178011 | True | True |
| crowds_zara02 | target-window | -0.006617 / -0.010728 / -0.011627 | -0.011393 / -0.026805 / -0.020792 | False | False |
| crowds_zara02 | equal-person-ID | -0.023637 / -0.033877 / -0.033571 | -0.055714 / -0.070829 / -0.063318 | False | False |

K1条件bootstrap的24个录像/seed/权重/指标区间中，11个包含0。该计数只作不确定性描述，不是预设稳定性门槛或多重显著性结论。

| 录像 | seed | 模型 | best轮 | best K1 ADE/FDE | last K1 ADE/FDE | 实耗min | 进程内PyTorch峰值分配GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | 17 | Full | 3 | 0.387184 / 0.751521 | 0.646965 / 1.442235 | 38.977063 | 1.847579 |
| biwi_hotel | 17 | PlaceboSocial | 3 | 0.392742 / 0.764808 | 0.562979 / 1.142229 | 39.900667 | 0.191485 |
| crowds_zara02 | 17 | Full | 30 | 0.272563 / 0.607513 | 0.329019 / 0.739805 | 34.784877 | 1.847157 |
| crowds_zara02 | 17 | PlaceboSocial | 76 | 0.279180 / 0.618906 | 0.306436 / 0.679855 | 36.310610 | 0.192216 |
| biwi_hotel | 42 | Full | 2 | 0.435954 / 0.883544 | 0.690960 / 1.543189 | 66.250956 | 1.849527 |
| biwi_hotel | 42 | PlaceboSocial | 2 | 0.444768 / 0.916580 | 0.545751 / 1.113987 | 64.671642 | 0.191257 |
| crowds_zara02 | 42 | Full | 29 | 0.269965 / 0.599033 | 0.346779 / 0.784363 | 63.605324 | 1.849199 |
| crowds_zara02 | 42 | PlaceboSocial | 57 | 0.280692 / 0.625838 | 0.295310 / 0.655219 | 63.001233 | 0.192216 |
| biwi_hotel | 2026 | Full | 2 | 0.417311 / 0.825002 | 0.691426 / 1.495265 | 64.257206 | 1.849771 |
| biwi_hotel | 2026 | PlaceboSocial | 2 | 0.406706 / 0.814658 | 0.781096 / 1.623472 | 63.628949 | 0.191314 |
| crowds_zara02 | 2026 | Full | 28 | 0.271458 / 0.605856 | 0.351435 / 0.807351 | 63.250382 | 1.849199 |
| crowds_zara02 | 2026 | PlaceboSocial | 42 | 0.283085 / 0.626648 | 0.301555 / 0.669227 | 62.914432 | 0.191314 |


## 辅助采样指标

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


本阶段只允许并实际执行8个新增100epoch run，未训练NoSocial/EgoCapacity，未改变Placebo、attention、结构、loss、数据协议或评价batch。seed42成绩未改变seed2026配置，未加seed或200epoch延长。4个seed17来源冻结而非重复训练，全部12组合单独列出。

Stage5.4原始单seed结果和报告保持不变。本阶段新增证据用于更新其跨随机训练可推广性解释，不回写旧指标，也不把旧阶段的初步迹象升级为未经验证的独立因果结论。

平均配对差值必须与配对SD、各模型跨seed SD及每个seed原始差值一同解释。共同难易波动可能被配对抵消，因此不能只凭独立模型SD较大否定一致的配对方向；也不能凭n=3的平均小差值宣布统计确认。K1首样本/K20的评价噪声随训练seed按预注册键变化，其波动不是纯初始化波动。

bootstrap固定最佳checkpoint和评价噪声，只重采样人ID，未重做训练或checkpoint选择；不修复选模偏差、历史开发曝光、行人间相互依赖或独立录像不足。HOTEL28、ZARA2 35 ID，不能把1410重叠窗口或三次训练当大量独立样本。

HOTEL与三种强CV、正常行走绝对误差、后20轮泛化退化在05报告完整展示。K1与K20分开解释，不以K20替代社会K1稳定性。普通VAE、伪邻居和多seed统计本身不是论文新方法。

本阶段结束，暂不设计第四章新网络，不自行开展更多训练。若稳定性未获支持，明确停止由这项毫米差驱动的复杂社会模块路线；成熟SocialVAE可保留为冻结公开基线，后续研究问题需新的独立指令。即使稳定性通过，也只支持进一步研究设计，不等于已获得方法创新。

验收：8组新增正式run均100完整epoch；12组合完整索引；349项完整单元测试通过；6346个历史文件SHA未变；48个新增checkpoint核验；ETH未评价，官方源代码和Stage5.4合成规则未改。

训练前冻结协议/代码提交：`cdeb5993a2c1f22b5153ccba81aa20257b929545`，全部8组新增run采用该代码版本。最终交付提交由本报告所在Git历史核查；不虚构包含自身报告的循环SHA。

结果入口：results/tables/stage5p5/analysis/；bootstrap原始分布、抽样ID和区间在bootstrap/；每轮原始日志在runs/。代码/数据/配置/模型SHA见frozen_source_sha.json、各run/provenance.json、analysis/all_12_run_index.csv及checkpoint_inventory.csv。NoSocial/EgoCapacity历史结果只在historical_reference_only.csv，不参与多seed汇总。

详细报告01冻结协议、02训练、03配对与bootstrap、04全部历史组、05泛化、06科研决策均已保存。本地commit后停止，不自动push。用户自行：git push -u origin feat/stage5p5-social-stability。
