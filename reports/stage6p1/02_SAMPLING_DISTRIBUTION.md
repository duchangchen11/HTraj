# 历史速度类别与唯一采样分布

每目标速度=sqrt(dx²+dy²)/0.4的7段均值，float64，只读取8帧目标位置。stationary<0.05、low[0.05,0.4)、normal[0.4,1.8)、fast≥1.8m/s。类别与概率API不接受未来、邻居或录像身份；来源目标池已经按冻结监督资格确定，不因采样改变。

唯一S：P(i)=0.5/N+0.5/(G*N_g)，G是非空速度类别数，因此总和0.5+0.5=1；空类概率0、不造样本。U按完整置换一轮一次，S每轮有放回N次。非空类期望占比=0.5*N_g/N+0.5/G。下表训练占比为训练前期望，验证只描述原有构成，不用来计算P或修改比例。

| 留出录像 | split | 类别 | 类别目标数/N | 原比例 | S期望比例 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | train | stationary | 4644/27145 | 0.171081 | 0.210541 |
| biwi_hotel | train | low | 6130/27145 | 0.225824 | 0.237912 |
| biwi_hotel | train | normal | 16341/27145 | 0.601989 | 0.425995 |
| biwi_hotel | train | fast | 30/27145 | 0.001105 | 0.125553 |
| biwi_hotel | validation | stationary | 106/318 | 0.333333 | — |
| biwi_hotel | validation | low | 121/318 | 0.380503 | — |
| biwi_hotel | validation | normal | 91/318 | 0.286164 | — |
| biwi_hotel | validation | fast | 0/318 | 0.000000 | — |
| crowds_zara02 | train | stationary | 3116/23545 | 0.132342 | 0.191171 |
| crowds_zara02 | train | low | 5924/23545 | 0.251603 | 0.250802 |
| crowds_zara02 | train | normal | 14487/23545 | 0.615290 | 0.432645 |
| crowds_zara02 | train | fast | 18/23545 | 0.000764 | 0.125382 |
| crowds_zara02 | validation | stationary | 579/1092 | 0.530220 | — |
| crowds_zara02 | validation | low | 91/1092 | 0.083333 | — |
| crowds_zara02 | validation | normal | 422/1092 | 0.386447 | — |
| crowds_zara02 | validation | fast | 0/1092 | 0.000000 | — |


实际每轮四类visits、unique_targets、repeated_visits、repeated_target_ID_windows、coverage、sampled_fraction保存各run/epochs/epochXXX_sampling.csv；完整每目标访问计数含0访问、累积计数在epochXXX_target_visits.csv.gz。这里“目标”指recording/start_frame/person-ID窗口键，不把重叠窗口当独立行人。repeated_visits=总访问−独特目标窗口；repeated_target_ID_windows=访问≥2次的独特目标窗口数，两者不同。保存order SHA和固定生成代码以复算顺序，累计覆盖率另报。结果文件最终由真实训练产生，本报告不填未执行指标。
