# PlaceboSocial真实执行结果

单位米，ADE/FDE配对来自同一按最低验证K1 ADE选择的checkpoint。K1 latent-prior不是预测分布数学期望；K20两项各自取最优候选。全部结果来自seed17，重复曝光开发录像，不是独立确认或跨seed证据。

| 录像 | 模型 | K1 ADE/FDE | K1首样本 ADE/FDE | K20 minADE/minFDE |
| --- | --- | --- | --- | --- |
| biwi_hotel | CV-Last2 | 0.329516 / 0.611038 | — | — |
| biwi_hotel | CV-OLS4 | 0.237814 / 0.438866 | — | — |
| biwi_hotel | CV-OLS8 | 0.220144 / 0.408760 | — | — |
| biwi_hotel | Full | 0.387184 / 0.751521 | 0.530319 / 1.075447 | 0.202466 / 0.325331 |
| biwi_hotel | NoSocial | 0.395194 / 0.777809 | 0.573431 / 1.153491 | 0.246114 / 0.416576 |
| biwi_hotel | EgoCapacity | 0.440967 / 0.867500 | 0.549841 / 1.111585 | 0.220368 / 0.369487 |
| biwi_hotel | PlaceboSocial | 0.392742 / 0.764808 | 0.544893 / 1.112961 | 0.223376 / 0.374198 |
| crowds_zara02 | CV-Last2 | 0.296306 / 0.667523 | — | — |
| crowds_zara02 | CV-OLS4 | 0.331875 / 0.715683 | — | — |
| crowds_zara02 | CV-OLS8 | 0.383175 / 0.787222 | — | — |
| crowds_zara02 | Full | 0.272563 / 0.607513 | 0.409855 / 0.947254 | 0.157530 / 0.304836 |
| crowds_zara02 | NoSocial | 0.278869 / 0.613396 | 0.401261 / 0.901552 | 0.155090 / 0.294163 |
| crowds_zara02 | EgoCapacity | 0.276378 / 0.620761 | 0.392280 / 0.900171 | 0.152978 / 0.293431 |
| crowds_zara02 | PlaceboSocial | 0.279180 / 0.618906 | 0.410256 / 0.918173 | 0.146936 / 0.280722 |

| 录像 | split | 上下文 | 统计量 | 均值 | P95 | 零值比例 |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | train | real | count | 4.094322 | 9.000000 | 0.058648 |
| biwi_hotel | train | real | distance | 1.221782 | 1.930484 | 0.000000 |
| biwi_hotel | train | real | speed | 0.542571 | 1.240102 | 0.000000 |
| biwi_hotel | train | real | relative_speed | 0.609569 | 1.683201 | 0.000954 |
| biwi_hotel | train | synthetic | count | 4.000000 | 4.000000 | 0.000000 |
| biwi_hotel | train | synthetic | distance | 0.802829 | 0.983062 | 0.000000 |
| biwi_hotel | train | synthetic | speed | 0.573771 | 1.339323 | 0.000000 |
| biwi_hotel | train | synthetic | relative_speed | 0.059377 | 0.165918 | 0.000000 |
| biwi_hotel | validation | real | count | 1.023585 | 3.000000 | 0.314072 |
| biwi_hotel | validation | real | distance | 1.073755 | 1.948795 | 0.000000 |
| biwi_hotel | validation | real | speed | 0.987908 | 1.985101 | 0.184073 |
| biwi_hotel | validation | real | relative_speed | 0.955067 | 2.700463 | 0.120540 |
| biwi_hotel | validation | synthetic | count | 4.000000 | 4.000000 | 0.000000 |
| biwi_hotel | validation | synthetic | distance | 0.800516 | 0.920511 | 0.000000 |
| biwi_hotel | validation | synthetic | speed | 0.401686 | 1.515062 | 0.374214 |
| biwi_hotel | validation | synthetic | relative_speed | 0.034694 | 0.157803 | 0.374214 |
| crowds_zara02 | train | real | count | 4.296225 | 10.000000 | 0.057008 |
| crowds_zara02 | train | real | distance | 1.236134 | 1.932167 | 0.000000 |
| crowds_zara02 | train | real | speed | 0.545289 | 1.205088 | 0.002277 |
| crowds_zara02 | train | real | relative_speed | 0.618911 | 1.663665 | 0.001236 |
| crowds_zara02 | train | synthetic | count | 4.000000 | 4.000000 | 0.000000 |
| crowds_zara02 | train | synthetic | distance | 0.802485 | 0.981522 | 0.000000 |
| crowds_zara02 | train | synthetic | speed | 0.570537 | 1.309668 | 0.013057 |
| crowds_zara02 | train | synthetic | relative_speed | 0.061251 | 0.163123 | 0.013057 |
| crowds_zara02 | validation | real | count | 2.459592 | 5.000000 | 0.141827 |
| crowds_zara02 | validation | real | distance | 1.091435 | 1.899037 | 0.000000 |
| crowds_zara02 | validation | real | speed | 0.380305 | 1.309494 | 0.000000 |
| crowds_zara02 | validation | real | relative_speed | 0.422618 | 1.567629 | 0.070136 |
| crowds_zara02 | validation | synthetic | count | 4.000000 | 4.000000 | 0.000000 |
| crowds_zara02 | validation | synthetic | distance | 0.801606 | 0.977408 | 0.000000 |
| crowds_zara02 | validation | synthetic | speed | 0.442013 | 1.388092 | 0.000000 |
| crowds_zara02 | validation | synthetic | relative_speed | 0.036248 | 0.169396 | 0.000000 |

实际执行两组100完整epoch；规则仅此一套、没有搜索。公式和边界/未来padding规则见PLACEBO_FROZEN_PROTOCOL.md；训练前10项隔离结构测试和全套337项测试通过。Full原始社会模块均实际执行并获得梯度，见placebo_capacity_by_parameter.csv和每轮梯度记录。名义参数2144002、结构有效上界2045698，与Full一致，但常量初始偏移限制输入多样性，非函数类相同。协议、合成上下文分布与隔离审计见本阶段phase2产物。Placebo改变社会输入分布，不是完美单因素对照；结果不能自动等同于独立社会因果效应。
