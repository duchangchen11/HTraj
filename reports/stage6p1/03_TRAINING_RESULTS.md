# 四组实际训练结果

| 录像 | 条件 | best/last | 参数/有效上界 | 访问/优化步 | wall秒 | 峰值显存MiB | train loss首/末 | best train ADE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | Uniform | 2/30 | 2144002/1459906 | 814350/6390 | 714.358375 | 115.802734 | 0.425328 / 0.021673 | 0.509051 |
| biwi_hotel | Stratified | 3/30 | 2144002/1459906 | 814350/6390 | 713.013831 | 115.814453 | 0.435386 / 0.020028 | 0.513500 |
| crowds_zara02 | Uniform | 30/30 | 2144002/1459906 | 706350/5520 | 682.446843 | 117.244141 | 0.448721 / 0.023132 | 0.481472 |
| crowds_zara02 | Stratified | 27/30 | 2144002/1459906 | 706350/5520 | 687.152453 | 115.796875 | 0.458405 / 0.020499 | 0.488250 |


四组为独立进程/同seed随机初始化、无热启动；各30完整预算epoch，smoke包含1–5轮，未增加训练。S的“完整”指N次预算访问，不是全体独特目标均被访问。并发导致wall时间包含资源争用，GPU显存是各进程PyTorch峰值而非整卡独占使用。参数与数据SHA、全部30曲线/逐步梯度/loss原始记录、定期权重和best/last索引保留。

| 录像 | 选取 | 评价 | 权重 | N/ID | U ADE/FDE | S ADE/FDE | S−U ADE/FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | best | K1-prior | target-window | 318/28 | 0.395194 / 0.777809 | 0.379526 / 0.747698 | -0.015668 / -0.030112 |
| biwi_hotel | best | K1-first-sample | target-window | 318/28 | 0.573431 / 1.153491 | 0.499462 / 1.010621 | -0.073968 / -0.142869 |
| biwi_hotel | best | K20-independent-minima | target-window | 318/28 | 0.246114 / 0.416576 | 0.218137 / 0.362277 | -0.027976 / -0.054299 |
| biwi_hotel | best | K1-prior | equal-person-ID | 318/28 | 0.816794 / 1.725044 | 0.712476 / 1.483290 | -0.104318 / -0.241753 |
| biwi_hotel | best | K1-first-sample | equal-person-ID | 318/28 | 0.950430 / 1.964655 | 0.827591 / 1.782147 | -0.122839 / -0.182508 |
| biwi_hotel | best | K20-independent-minima | equal-person-ID | 318/28 | 0.509779 / 1.050047 | 0.429511 / 0.865476 | -0.080267 / -0.184571 |
| biwi_hotel | last | K1-prior | target-window | 318/28 | 0.558966 / 1.170339 | 0.578712 / 1.214370 | 0.019746 / 0.044031 |
| biwi_hotel | last | K1-first-sample | target-window | 318/28 | 0.728703 / 1.533088 | 0.782287 / 1.665304 | 0.053584 / 0.132216 |
| biwi_hotel | last | K20-independent-minima | target-window | 318/28 | 0.246668 / 0.401834 | 0.269871 / 0.472063 | 0.023203 / 0.070230 |
| biwi_hotel | last | K1-prior | equal-person-ID | 318/28 | 1.078557 / 2.506483 | 1.185349 / 2.739674 | 0.106792 / 0.233191 |
| biwi_hotel | last | K1-first-sample | equal-person-ID | 318/28 | 1.488053 / 3.216738 | 1.530159 / 3.409375 | 0.042106 / 0.192637 |
| biwi_hotel | last | K20-independent-minima | equal-person-ID | 318/28 | 0.435092 / 0.790411 | 0.590113 / 1.228079 | 0.155021 / 0.437668 |
| crowds_zara02 | best | K1-prior | target-window | 1092/35 | 0.280626 / 0.618619 | 0.287247 / 0.650746 | 0.006621 / 0.032127 |
| crowds_zara02 | best | K1-first-sample | target-window | 1092/35 | 0.434273 / 0.992916 | 0.434248 / 0.981975 | -0.000024 / -0.010941 |
| crowds_zara02 | best | K20-independent-minima | target-window | 1092/35 | 0.158593 / 0.302609 | 0.156119 / 0.293545 | -0.002473 / -0.009064 |
| crowds_zara02 | best | K1-prior | equal-person-ID | 1092/35 | 0.380404 / 0.742902 | 0.409779 / 0.828651 | 0.029375 / 0.085749 |
| crowds_zara02 | best | K1-first-sample | equal-person-ID | 1092/35 | 0.609148 / 1.313812 | 0.568843 / 1.210546 | -0.040305 / -0.103266 |
| crowds_zara02 | best | K20-independent-minima | equal-person-ID | 1092/35 | 0.198453 / 0.311286 | 0.204781 / 0.325599 | 0.006327 / 0.014313 |
| crowds_zara02 | last | K1-prior | target-window | 1092/35 | 0.280626 / 0.618619 | 0.300288 / 0.674956 | 0.019662 / 0.056337 |
| crowds_zara02 | last | K1-first-sample | target-window | 1092/35 | 0.434273 / 0.992916 | 0.444121 / 1.021428 | 0.009848 / 0.028512 |
| crowds_zara02 | last | K20-independent-minima | target-window | 1092/35 | 0.158593 / 0.302609 | 0.158288 / 0.303798 | -0.000305 / 0.001190 |
| crowds_zara02 | last | K1-prior | equal-person-ID | 1092/35 | 0.380404 / 0.742902 | 0.407381 / 0.818155 | 0.026976 / 0.075253 |
| crowds_zara02 | last | K1-first-sample | equal-person-ID | 1092/35 | 0.609148 / 1.313812 | 0.610706 / 1.345671 | 0.001558 / 0.031859 |
| crowds_zara02 | last | K20-independent-minima | equal-person-ID | 1092/35 | 0.198453 / 0.311286 | 0.201119 / 0.328988 | 0.002665 / 0.017702 |
| two-recording-macro | best | K1-prior | target-window | 1410/63 | 0.337910 / 0.698214 | 0.333387 / 0.699222 | -0.004523 / 0.001008 |
| two-recording-macro | best | K1-first-sample | target-window | 1410/63 | 0.503852 / 1.073203 | 0.466855 / 0.996298 | -0.036996 / -0.076905 |
| two-recording-macro | best | K20-independent-minima | target-window | 1410/63 | 0.202353 / 0.359592 | 0.187128 / 0.327911 | -0.015225 / -0.031681 |
| two-recording-macro | best | K1-prior | equal-person-ID | 1410/63 | 0.598599 / 1.233973 | 0.561128 / 1.155971 | -0.037471 / -0.078002 |
| two-recording-macro | best | K1-first-sample | equal-person-ID | 1410/63 | 0.779789 / 1.639233 | 0.698217 / 1.496346 | -0.081572 / -0.142887 |
| two-recording-macro | best | K20-independent-minima | equal-person-ID | 1410/63 | 0.354116 / 0.680667 | 0.317146 / 0.595538 | -0.036970 / -0.085129 |
| two-recording-macro | last | K1-prior | target-window | 1410/63 | 0.419796 / 0.894479 | 0.439500 / 0.944663 | 0.019704 / 0.050184 |
| two-recording-macro | last | K1-first-sample | target-window | 1410/63 | 0.581488 / 1.263002 | 0.613204 / 1.343366 | 0.031716 / 0.080364 |
| two-recording-macro | last | K20-independent-minima | target-window | 1410/63 | 0.202630 / 0.352221 | 0.214080 / 0.387931 | 0.011449 / 0.035710 |
| two-recording-macro | last | K1-prior | equal-person-ID | 1410/63 | 0.729481 / 1.624693 | 0.796365 / 1.778915 | 0.066884 / 0.154222 |
| two-recording-macro | last | K1-first-sample | equal-person-ID | 1410/63 | 1.048600 / 2.265275 | 1.070433 / 2.377523 | 0.021832 / 0.112248 |
| two-recording-macro | last | K20-independent-minima | equal-person-ID | 1410/63 | 0.316773 / 0.550849 | 0.395616 / 0.778533 | 0.078843 / 0.227685 |

所有FDE配对同checkpoint；不拼接独立最优ADE/FDE。best选择只依据窗口K1 ADE，ID权重/K20没有重新选权。原始训练目标不是精确边缘NLL，K1 latent-prior不是分布期望。K20的最小ADE与最小FDE可以来自不同候选。

| 录像 | 条件 | 平均独特覆盖率 | 最小 | 最大 | 30轮重复访问 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | Uniform | 1.000000 | 1.000000 | 1.000000 | 0 |
| biwi_hotel | Stratified | 0.575082 | 0.571597 | 0.578154 | 346032 |
| crowds_zara02 | Uniform | 1.000000 | 1.000000 | 1.000000 | 0 |
| crowds_zara02 | Stratified | 0.571705 | 0.566447 | 0.576216 | 302526 |


best_at_budget_boundary见run_summary.csv：若best=30则预算截断，不宣称充分收敛，不自动延长。训练loss持续下降也不等于验证泛化改善。best_train与验证差距应在相同K1口径下解释，而非把重建loss和ADE直接相减。

## 实际类别曝光与稀有目标

| 留出录像 | 条件 | 类别 | 目标窗口/来源ID | 实际30轮访问 | 每目标平均访问 | 每轮平均覆盖率 | 实际类别占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | Uniform | stationary | 4644/97 | 139320 | 30.000 | 1.000000 | 0.171081 |
| biwi_hotel | Uniform | low | 6130/315 | 183900 | 30.000 | 1.000000 | 0.225824 |
| biwi_hotel | Uniform | normal | 16341/868 | 490230 | 30.000 | 1.000000 | 0.601989 |
| biwi_hotel | Uniform | fast | 30/13 | 900 | 30.000 | 1.000000 | 0.001105 |
| biwi_hotel | Stratified | stationary | 4644/97 | 171374 | 36.902 | 0.707637 | 0.210443 |
| biwi_hotel | Stratified | low | 6130/315 | 193741 | 31.605 | 0.651218 | 0.237909 |
| biwi_hotel | Stratified | normal | 16341/868 | 347072 | 21.239 | 0.508070 | 0.426195 |
| biwi_hotel | Stratified | fast | 30/13 | 102163 | 3405.433 | 1.000000 | 0.125453 |
| crowds_zara02 | Uniform | stationary | 3116/108 | 93480 | 30.000 | 1.000000 | 0.132342 |
| crowds_zara02 | Uniform | low | 5924/311 | 177720 | 30.000 | 1.000000 | 0.251603 |
| crowds_zara02 | Uniform | normal | 14487/789 | 434610 | 30.000 | 1.000000 | 0.615290 |
| crowds_zara02 | Uniform | fast | 18/7 | 540 | 30.000 | 1.000000 | 0.000764 |
| crowds_zara02 | Stratified | stationary | 3116/108 | 134712 | 43.232 | 0.764752 | 0.190716 |
| crowds_zara02 | Stratified | low | 5924/311 | 176843 | 29.852 | 0.629980 | 0.250362 |
| crowds_zara02 | Stratified | normal | 14487/789 | 306019 | 21.124 | 0.505821 | 0.433240 |
| crowds_zara02 | Stratified | fast | 18/7 | 88776 | 4932.000 | 1.000000 | 0.125683 |

两个S累计30轮均覆盖全部原目标窗口，但每轮约57%的独特覆盖仍与U的100%不同；正常类每目标约21次而非U的30次。fast只有30/18窗口，涉及13/7来源ID，在S中每窗口累计约3405/4932次，U只有30次。固定混合并非“增加正常行走样本”：正常类比例下降、稀有高速比例大增，多个训练构成变化共同发生。重复曝光、优化噪声、状态比例和模型选择时间都可能影响结果，未独立证明某一项是原因；不事后另试比例。recording-person ID可跨历史类别出现，不能把各类别ID数相加当独立行人数。

实际类别计数/覆盖/重复见actual_class_sampling_summary.csv，每ID曝光见actual_source_person_exposure.csv.gz，全部逐epoch访问计数含零访问已保存。Uniform前5轮与旧同轮ADE/FDE一致，仅作实现复现检查；本次比较完整使用新30轮U，没有以旧100轮替代。
