# 历史运动分组和速度诊断

| 录像 | 条件 | 组 | 权重 | N/ID | ADE/FDE | 预测/真实均速 | 有符号偏差 | mean|目标偏差| | 沿/横终点误差 | 可靠朝向N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel | Uniform | speed_stationary | target-window | 106/4 | 0.113171 / 0.207669 | 0.030247 / 0.083992 | -0.053744 | 0.066181 | — / — | 0 |
| biwi_hotel | Uniform | speed_stationary | equal-person-ID | 106/4 | 0.118449 / 0.205285 | 0.031548 / 0.098822 | -0.067274 | 0.077411 | — / — | 0 |
| biwi_hotel | Uniform | speed_low | target-window | 121/5 | 0.268267 / 0.434828 | 0.082344 / 0.098097 | -0.015752 | 0.078472 | 0.579365 / 0.037313 | 8 |
| biwi_hotel | Uniform | speed_low | equal-person-ID | 121/5 | 0.241576 / 0.397633 | 0.075496 / 0.081111 | -0.005615 | 0.070624 | 0.597665 / 0.182743 | 8 |
| biwi_hotel | Uniform | speed_normal | target-window | 91/24 | 0.892475 / 1.897980 | 0.945982 / 1.213391 | -0.267408 | 0.323971 | -1.282490 / 0.687691 | 91 |
| biwi_hotel | Uniform | speed_normal | equal-person-ID | 91/24 | 0.931708 / 1.984590 | 0.991938 / 1.297841 | -0.305903 | 0.345354 | -1.465664 / 0.720187 | 91 |
| biwi_hotel | Uniform | speed_fast | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| biwi_hotel | Uniform | speed_fast | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |
| biwi_hotel | Uniform | turning | target-window | 3/2 | 0.705717 / 1.401488 | 0.325035 / 0.238339 | 0.086696 | 0.273671 | 0.543300 / -0.197233 | 3 |
| biwi_hotel | Uniform | turning | equal-person-ID | 3/2 | 0.804253 / 1.439034 | 0.284278 / 0.289371 | -0.005093 | 0.275369 | 0.173066 / -0.457008 | 3 |
| biwi_hotel | Uniform | direction_change | target-window | 30/4 | 0.377586 / 0.598810 | 0.139498 / 0.161044 | -0.021546 | 0.116165 | 0.338423 / -0.440888 | 5 |
| biwi_hotel | Uniform | direction_change | equal-person-ID | 30/4 | 0.446524 / 0.721909 | 0.185567 / 0.189072 | -0.003505 | 0.130954 | 0.272058 / -0.290053 | 5 |
| biwi_hotel | Stratified | speed_stationary | target-window | 106/4 | 0.151463 / 0.295994 | 0.050265 / 0.083992 | -0.033727 | 0.062834 | — / — | 0 |
| biwi_hotel | Stratified | speed_stationary | equal-person-ID | 106/4 | 0.158949 / 0.300185 | 0.050993 / 0.098822 | -0.047830 | 0.072150 | — / — | 0 |
| biwi_hotel | Stratified | speed_low | target-window | 121/5 | 0.249292 / 0.417480 | 0.081930 / 0.098097 | -0.016167 | 0.069183 | 0.273477 / -0.044284 | 8 |
| biwi_hotel | Stratified | speed_low | equal-person-ID | 121/5 | 0.219225 / 0.377414 | 0.075237 / 0.081111 | -0.005874 | 0.062887 | 0.357178 / 0.124446 | 8 |
| biwi_hotel | Stratified | speed_normal | target-window | 91/24 | 0.818352 / 1.712939 | 1.026484 / 1.213391 | -0.186906 | 0.247579 | -0.964854 / -0.451352 | 91 |
| biwi_hotel | Stratified | speed_normal | equal-person-ID | 91/24 | 0.809714 / 1.700224 | 1.090936 / 1.297841 | -0.206905 | 0.252199 | -1.075593 / -0.394594 | 91 |
| biwi_hotel | Stratified | speed_fast | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| biwi_hotel | Stratified | speed_fast | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |
| biwi_hotel | Stratified | turning | target-window | 3/2 | 0.724047 / 1.523175 | 0.350385 / 0.238339 | 0.112047 | 0.297114 | 0.667167 / -0.227057 | 3 |
| biwi_hotel | Stratified | turning | equal-person-ID | 3/2 | 0.818484 / 1.535924 | 0.304006 / 0.289371 | 0.014635 | 0.292236 | 0.266400 / -0.486728 | 3 |
| biwi_hotel | Stratified | direction_change | target-window | 30/4 | 0.386641 / 0.622995 | 0.126577 / 0.161044 | -0.034467 | 0.121330 | -0.110705 / -1.090890 | 5 |
| biwi_hotel | Stratified | direction_change | equal-person-ID | 30/4 | 0.494793 / 0.816631 | 0.164650 / 0.189072 | -0.024423 | 0.140076 | -0.060431 / -0.764670 | 5 |
| crowds_zara02 | Uniform | speed_stationary | target-window | 579/9 | 0.117765 / 0.317147 | 0.009156 / 0.069124 | -0.059968 | 0.066287 | — / — | 0 |
| crowds_zara02 | Uniform | speed_stationary | equal-person-ID | 579/9 | 0.202497 / 0.566131 | 0.011093 / 0.124155 | -0.113062 | 0.118108 | — / — | 0 |
| crowds_zara02 | Uniform | speed_low | target-window | 91/10 | 0.551255 / 1.226244 | 0.190770 / 0.358095 | -0.167325 | 0.230534 | -0.271478 / 0.249213 | 25 |
| crowds_zara02 | Uniform | speed_low | equal-person-ID | 91/10 | 0.728778 / 1.591625 | 0.332088 / 0.598336 | -0.266247 | 0.308066 | -0.708672 / 0.524394 | 25 |
| crowds_zara02 | Uniform | speed_normal | target-window | 422/33 | 0.445718 / 0.901222 | 1.036224 / 1.024661 | 0.011563 | 0.110153 | 0.148794 / 0.065546 | 422 |
| crowds_zara02 | Uniform | speed_normal | equal-person-ID | 422/33 | 0.480602 / 0.909892 | 1.098938 / 1.112267 | -0.013329 | 0.119755 | 0.032043 / 0.078550 | 422 |
| crowds_zara02 | Uniform | speed_fast | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Uniform | speed_fast | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Uniform | turning | target-window | 15/5 | 0.622813 / 1.416444 | 0.854188 / 0.734520 | 0.119668 | 0.197178 | 0.573658 / 0.382575 | 15 |
| crowds_zara02 | Uniform | turning | equal-person-ID | 15/5 | 0.613527 / 1.427164 | 0.853045 / 0.701316 | 0.151729 | 0.209861 | 0.717980 / 0.302689 | 15 |
| crowds_zara02 | Uniform | direction_change | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Uniform | direction_change | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Stratified | speed_stationary | target-window | 579/9 | 0.114743 / 0.328030 | 0.013738 / 0.069124 | -0.055386 | 0.068383 | — / — | 0 |
| crowds_zara02 | Stratified | speed_stationary | equal-person-ID | 579/9 | 0.199226 / 0.574712 | 0.015788 / 0.124155 | -0.108367 | 0.118165 | — / — | 0 |
| crowds_zara02 | Stratified | speed_low | target-window | 91/10 | 0.545841 / 1.221390 | 0.193591 / 0.358095 | -0.164504 | 0.227351 | -0.213180 / 0.215812 | 25 |
| crowds_zara02 | Stratified | speed_low | equal-person-ID | 91/10 | 0.718467 / 1.570596 | 0.336625 / 0.598336 | -0.261711 | 0.302490 | -0.661831 / 0.493081 | 25 |
| crowds_zara02 | Stratified | speed_normal | target-window | 422/33 | 0.468166 / 0.970472 | 1.077414 / 1.024661 | 0.052753 | 0.122976 | 0.343225 / 0.191898 | 422 |
| crowds_zara02 | Stratified | speed_normal | equal-person-ID | 422/33 | 0.513802 / 1.001731 | 1.142806 / 1.112267 | 0.030539 | 0.137014 | 0.239263 / 0.220110 | 422 |
| crowds_zara02 | Stratified | speed_fast | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Stratified | speed_fast | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Stratified | turning | target-window | 15/5 | 0.666434 / 1.513655 | 0.894416 / 0.734520 | 0.159896 | 0.229155 | 0.767533 / 0.544265 | 15 |
| crowds_zara02 | Stratified | turning | equal-person-ID | 15/5 | 0.663076 / 1.536363 | 0.895455 / 0.701316 | 0.194139 | 0.246083 | 0.918712 / 0.457482 | 15 |
| crowds_zara02 | Stratified | direction_change | target-window | 0/0 | — / — | — / — | — | — | — / — | 0 |
| crowds_zara02 | Stratified | direction_change | equal-person-ID | 0/0 | — / — | — / — | — | — | — / — | 0 |

速度由x_last到预测首帧再到后续11段位置除0.4计算，GT用同一起点；目标速度偏差是12段速度均值之差。有符号组偏差、绝对组偏差、逐目标绝对偏差与逐步绝对速度误差是不同统计量，不能互换。

终点沿/横轴由历史净位移定义，净位移≥0.2m且历史均速≥0.2m/s才可靠；否则保留空值，仍计算ADE/FDE/速度。沿轴负值可描述落后，但小组均值可能抵消，不能指定唯一因果。无可靠方向样本没有伪填0。所有33个冻结历史分组包括空组在all_history_group_metrics.csv和all_motion_group_diagnostics.csv，均未删难例。

## HOTEL正常行走：全部12步

| 条件 | 时间s | 权重 | N/ID | 位置误差m | 预测/真实速度m/s | 偏差m/s |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 0.400000 | target-window | 91/24 | 0.127570 | 1.001249 / 1.228603 | -0.227354 |
| Uniform | 0.400000 | equal-person-ID | 91/24 | 0.131691 | 1.063435 / 1.315280 | -0.251845 |
| Uniform | 0.800000 | target-window | 91/24 | 0.231814 | 1.009247 / 1.227433 | -0.218186 |
| Uniform | 0.800000 | equal-person-ID | 91/24 | 0.240959 | 1.070211 / 1.311097 | -0.240886 |
| Uniform | 1.200000 | target-window | 91/24 | 0.333258 | 1.014419 / 1.222478 | -0.208059 |
| Uniform | 1.200000 | equal-person-ID | 91/24 | 0.343989 | 1.072155 / 1.305974 | -0.233819 |
| Uniform | 1.600000 | target-window | 91/24 | 0.452268 | 1.013000 / 1.230509 | -0.217509 |
| Uniform | 1.600000 | equal-person-ID | 91/24 | 0.471460 | 1.067317 / 1.320652 | -0.253336 |
| Uniform | 2.000000 | target-window | 91/24 | 0.589139 | 1.003314 / 1.230733 | -0.227418 |
| Uniform | 2.000000 | equal-person-ID | 91/24 | 0.621175 | 1.054229 / 1.337130 | -0.282901 |
| Uniform | 2.400000 | target-window | 91/24 | 0.733684 | 0.986410 / 1.215584 | -0.229174 |
| Uniform | 2.400000 | equal-person-ID | 91/24 | 0.766588 | 1.033791 / 1.291906 | -0.258115 |
| Uniform | 2.800000 | target-window | 91/24 | 0.895906 | 0.963780 / 1.215220 | -0.251440 |
| Uniform | 2.800000 | equal-person-ID | 91/24 | 0.934792 | 1.007629 / 1.302008 | -0.294379 |
| Uniform | 3.200000 | target-window | 91/24 | 1.069343 | 0.936611 / 1.220750 | -0.284139 |
| Uniform | 3.200000 | equal-person-ID | 91/24 | 1.116889 | 0.977158 / 1.313861 | -0.336703 |
| Uniform | 3.600000 | target-window | 91/24 | 1.257570 | 0.906210 / 1.215515 | -0.309306 |
| Uniform | 3.600000 | equal-person-ID | 91/24 | 1.314404 | 0.943600 / 1.304009 | -0.360410 |
| Uniform | 4.000000 | target-window | 91/24 | 1.454548 | 0.873642 / 1.194690 | -0.321048 |
| Uniform | 4.000000 | equal-person-ID | 91/24 | 1.519625 | 0.908167 / 1.278089 | -0.369922 |
| Uniform | 4.400000 | target-window | 91/24 | 1.666619 | 0.839520 / 1.178450 | -0.338929 |
| Uniform | 4.400000 | equal-person-ID | 91/24 | 1.734334 | 0.871522 / 1.243977 | -0.372455 |
| Uniform | 4.800000 | target-window | 91/24 | 1.897980 | 0.804385 / 1.180723 | -0.376338 |
| Uniform | 4.800000 | equal-person-ID | 91/24 | 1.984590 | 0.834042 / 1.250103 | -0.416062 |
| Stratified | 0.400000 | target-window | 91/24 | 0.121685 | 1.027858 / 1.228603 | -0.200745 |
| Stratified | 0.400000 | equal-person-ID | 91/24 | 0.124674 | 1.088001 / 1.315280 | -0.227279 |
| Stratified | 0.800000 | target-window | 91/24 | 0.216948 | 1.050442 / 1.227433 | -0.176991 |
| Stratified | 0.800000 | equal-person-ID | 91/24 | 0.221463 | 1.112403 / 1.311097 | -0.198694 |
| Stratified | 1.200000 | target-window | 91/24 | 0.311732 | 1.070179 / 1.222478 | -0.152299 |
| Stratified | 1.200000 | equal-person-ID | 91/24 | 0.312176 | 1.134322 / 1.305974 | -0.171652 |
| Stratified | 1.600000 | target-window | 91/24 | 0.419378 | 1.084711 / 1.230509 | -0.145797 |
| Stratified | 1.600000 | equal-person-ID | 91/24 | 0.419459 | 1.149879 / 1.320652 | -0.170774 |
| Stratified | 2.000000 | target-window | 91/24 | 0.545514 | 1.090586 / 1.230733 | -0.140147 |
| Stratified | 2.000000 | equal-person-ID | 91/24 | 0.550740 | 1.156495 / 1.337130 | -0.180635 |
| Stratified | 2.400000 | target-window | 91/24 | 0.682558 | 1.085930 / 1.215584 | -0.129653 |
| Stratified | 2.400000 | equal-person-ID | 91/24 | 0.675182 | 1.152143 / 1.291906 | -0.139763 |
| Stratified | 2.800000 | target-window | 91/24 | 0.831167 | 1.070840 / 1.215220 | -0.144380 |
| Stratified | 2.800000 | equal-person-ID | 91/24 | 0.818146 | 1.137147 / 1.302008 | -0.164861 |
| Stratified | 3.200000 | target-window | 91/24 | 0.990824 | 1.046319 / 1.220750 | -0.174430 |
| Stratified | 3.200000 | equal-person-ID | 91/24 | 0.975127 | 1.112694 / 1.313861 | -0.201167 |
| Stratified | 3.600000 | target-window | 91/24 | 1.153756 | 1.013069 / 1.215515 | -0.202446 |
| Stratified | 3.600000 | equal-person-ID | 91/24 | 1.132783 | 1.079044 / 1.304009 | -0.224966 |
| Stratified | 4.000000 | target-window | 91/24 | 1.323831 | 0.972880 / 1.194690 | -0.221810 |
| Stratified | 4.000000 | equal-person-ID | 91/24 | 1.298040 | 1.037922 / 1.278089 | -0.240167 |
| Stratified | 4.400000 | target-window | 91/24 | 1.509898 | 0.927329 / 1.178450 | -0.251121 |
| Stratified | 4.400000 | equal-person-ID | 91/24 | 1.488550 | 0.991170 / 1.243977 | -0.252807 |
| Stratified | 4.800000 | target-window | 91/24 | 1.712939 | 0.877667 / 1.180723 | -0.303056 |
| Stratified | 4.800000 | equal-person-ID | 91/24 | 1.700224 | 0.940013 / 1.250103 | -0.310091 |


每目标/每步配对原始派生误差在motion_diagnostics.csv.gz与per_step_motion.csv.gz；完整XY只在被忽略cache并记录SHA。未来持续/减速标签是Stage6事后定义，未用于采样、模型输入或在线gate。当前实验只改变整个采样策略，不能独立证明状态构成或速度偏差是唯一因果中介。

## 实际运动改善与失败代价

HOTEL正常行走91窗口/24ID，真实未来均速1.213391m/s；U/S预测0.945982/1.026484m/s。有符号偏差-0.267408→-0.186906m/s，绝对组均值偏差0.267408→0.186906，下降30.10%。mean(|目标偏差|)也从0.323971降至0.247579m/s（下降23.58%），故该组改善不只是组均值正负抵消；这仍不等于每个目标都改善。

同组ADE/FDE从0.892475/1.897980降至0.818352/1.712939m。但HOTEL历史静止106窗口仅4ID，FDE从0.207669升至0.295994m，增加0.088324m，超过max(0.01m,5%)。必须保留这个退化，不能只展示正常行走。

ZARA2窗口与行人权重ADE/FDE门槛均失败；HOTEL静止FDE两权重失败。即使两录像宏平均有某些改善，也不能覆盖单录像失败。结论是固定采样产生条件性改变，未获得预设的跨录像稳健收益；停止当前采样假设，不再改比例/阈值/seed/模型/epoch。ZARA2-U的best=30，结论受固定预算限制；没有证明更长训练必然无效，也没有授权延长。
