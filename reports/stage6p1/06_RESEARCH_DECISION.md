# 冻结规则下的研究决策

**未通过全部预设门槛，停止当前采样假设；不再搜索比例、速度阈值、seed、结构或训练长度。**

| 门槛 | 录像 | 权重 | 通过 | 完整检查值 |
| --- | --- | --- | --- | --- |
| ADE | biwi_hotel | target-window | True | observed_gain=0.01566762504103425; required=0.005 |
| FDE | biwi_hotel | target-window | True | allowed=0.0077780923658232175; observed_regression=-0.03011169620502574 |
| ADE | biwi_hotel | equal-person-ID | True | observed_gain=0.10431761127184236; required=0.008167937877349575 |
| FDE | biwi_hotel | equal-person-ID | True | allowed=0.017250435223965254; observed_regression=-0.2417533483854597 |
| ADE | crowds_zara02 | target-window | False | observed_gain=-0.006621160773447821; required=0.005 |
| FDE | crowds_zara02 | target-window | False | allowed=0.006186194098942602; observed_regression=0.032126933193403984 |
| ADE | crowds_zara02 | equal-person-ID | False | observed_gain=-0.029374681365482325; required=0.005 |
| FDE | crowds_zara02 | equal-person-ID | False | allowed=0.007429024764479968; observed_regression=0.08574874810611899 |
| HOTEL_normal_speed_bias | HOTEL | target-window | True | Stratified_absolute_bias_mps=0.1869062456387646; Uniform_absolute_bias_mps=0.26740836179211; required_ratio=0.9 |
| stationary_FDE | biwi_hotel | target-window | False | allowed=0.010383460475659034; observed_regression=0.08832446011310477 |
| stationary_FDE | biwi_hotel | equal-person-ID | False | allowed=0.010264258121684118; observed_regression=0.09490021945433236 |
| stationary_FDE | crowds_zara02 | target-window | True | allowed=0.01585736569376672; observed_regression=0.010882899137004742 |
| stationary_FDE | crowds_zara02 | equal-person-ID | True | allowed=0.028306536916117044; observed_regression=0.008581632027777841 |
| integrity | all | all | True |  |


预算边界best：crowds_zara02/Uniform。到30轮即停止，不自动延长。


如果速度偏差改善但ADE/FDE门槛不通过，只是运动学诊断结果；若HOTEL改善、ZARA2退化，只是条件性现象。一个子组改善不能覆盖整体/另一录像失败。未来误差不能反过来定义采样类别。即便全部通过，常规类别重采样也不是新论文方法，仍需更多未曝光资料、seed和针对具体命题的文献核查；本轮不自动追加。

下一研究方向交由用户决定，不重启旧社会模块或可靠性gate。当前不训练创新模型、不读取ETH、不上传checkpoint/数据、不自动push。

## 对应实际结果的解释

HOTEL正常行走91窗口/24ID，真实未来均速1.213391m/s；U/S预测0.945982/1.026484m/s。有符号偏差-0.267408→-0.186906m/s，绝对组均值偏差0.267408→0.186906，下降30.10%。mean(|目标偏差|)也从0.323971降至0.247579m/s（下降23.58%），故该组改善不只是组均值正负抵消；这仍不等于每个目标都改善。

同组ADE/FDE从0.892475/1.897980降至0.818352/1.712939m。但HOTEL历史静止106窗口仅4ID，FDE从0.207669升至0.295994m，增加0.088324m，超过max(0.01m,5%)。必须保留这个退化，不能只展示正常行走。

ZARA2窗口与行人权重ADE/FDE门槛均失败；HOTEL静止FDE两权重失败。即使两录像宏平均有某些改善，也不能覆盖单录像失败。结论是固定采样产生条件性改变，未获得预设的跨录像稳健收益；停止当前采样假设，不再改比例/阈值/seed/模型/epoch。ZARA2-U的best=30，结论受固定预算限制；没有证明更长训练必然无效，也没有授权延长。
