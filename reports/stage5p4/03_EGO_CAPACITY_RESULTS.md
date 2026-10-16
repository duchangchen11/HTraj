# EgoCapacity正式训练结果

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

| 录像 | 模型 | best轮 | best训练ADE/FDE | last验证ADE/FDE | 100轮实耗min | 进程内PyTorch峰值分配GiB |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | Full | 3 | 0.513493 / 1.108064 | 0.646965 / 1.442235 | 38.98 | 1.848 |
| crowds_zara02 | Full | 30 | 0.455903 / 0.996379 | 0.329019 / 0.739805 | 34.78 | 1.847 |
| biwi_hotel | NoSocial | 2 | 0.509051 / 1.096352 | 0.705353 / 1.508765 | 33.66 | 0.113 |
| crowds_zara02 | NoSocial | 55 | 0.463614 / 1.005194 | 0.295512 / 0.656382 | 30.24 | 0.113 |
| biwi_hotel | EgoCapacity | 3 | 0.518636 / 1.125055 | 0.623997 / 1.313158 | 34.51 | 0.121 |
| crowds_zara02 | EgoCapacity | 47 | 0.466980 / 1.014499 | 0.309304 / 0.682250 | 31.30 | 0.121 |
| biwi_hotel | PlaceboSocial | 3 | 0.497393 / 1.073847 | 0.562979 / 1.142229 | 39.90 | 0.191 |
| crowds_zara02 | PlaceboSocial | 76 | 0.443863 / 0.960917 | 0.306436 / 0.679855 | 36.31 | 0.192 |

| 录像 | 模型 | 选择 | 轮次 | 训练loss | 重建 | MC KL | 验证后验目标 | 训练K1 ADE | 验证K1 ADE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | EgoCapacity | best | 3 | 0.037829 | 0.017807 | 0.020022 | 0.035980 | 0.518636 | 0.440967 |
| biwi_hotel | EgoCapacity | last | 100 | 0.017079 | 0.006473 | 0.010606 | 0.039060 | 0.371787 | 0.623997 |
| crowds_zara02 | EgoCapacity | best | 47 | 0.021354 | 0.008147 | 0.013207 | 0.014451 | 0.466980 | 0.276378 |
| crowds_zara02 | EgoCapacity | last | 100 | 0.018035 | 0.006883 | 0.011152 | 0.014188 | 0.386689 | 0.309304 |
| biwi_hotel | PlaceboSocial | best | 3 | 0.038742 | 0.018594 | 0.020148 | 0.035026 | 0.497393 | 0.392742 |
| biwi_hotel | PlaceboSocial | last | 100 | 0.017464 | 0.006639 | 0.010824 | 0.029718 | 0.378626 | 0.562979 |
| crowds_zara02 | PlaceboSocial | best | 76 | 0.019734 | 0.007245 | 0.012489 | 0.013618 | 0.443863 | 0.279180 |
| crowds_zara02 | PlaceboSocial | last | 100 | 0.018716 | 0.007206 | 0.011510 | 0.014766 | 0.397122 | 0.306436 |

| 录像 | 模型 | 名义 | 结构有效上界 | loss连接模块 | 长期非零梯度标量 | 长期从未非零梯度标量 |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | ego_capacity | 2669570 | 1985474 | 2182082 | 1981885 | 687685 |
| crowds_zara02 | ego_capacity | 2669570 | 1985474 | 2182082 | 1982792 | 686778 |
| biwi_hotel | placebo_social | 2144002 | 2045698 | 2144002 | 1951344 | 192658 |
| crowds_zara02 | placebo_social | 2144002 | 2045698 | 2144002 | 1950566 | 193436 |

两组EgoCapacity都从全新seed17初始化，未导入旧权重。每轮遍历全部训练目标，保留尾batch；完整曲线、逐目标预测误差、per-step loss/rec/KL/梯度、每轮参数审计见results/tables/stage5p4/ego_capacity/。best和last及固定间隔checkpoint均留存在被Git忽略的experiments/stage5p4/checkpoints/；来源和SHA见selection.json、checkpoint_inventory.csv。两组EgoCapacity部分时段在同一GPU并发，模型、优化器和RNG各自独立；execution_schedule.json留档。各run实耗包含资源共享影响。Full/NoSocial实耗为冻结历史运行值，软件路径和资源调度不同，不能只凭实耗直接比较架构效率。

结构上界相近不等于函数类、优化轨迹或实际梯度活跃率完全相同。名义未用参数不计入有效上界；MLP中从未激活的ReLU相关标量由长期审计单列。没有通过任意参数掩码凑数。
