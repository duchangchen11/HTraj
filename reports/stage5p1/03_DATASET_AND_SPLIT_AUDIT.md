# 数据、两协议manifest与录像隔离

只读取HTraj本项目五个非ETH原始录像：biwi_hotel、students001、students003、crowds_zara01、crowds_zara02。原始文件逐项SHA与Stage3.5 manifest一致。没有加载ETH轨迹或评价其成绩，也没有导入其他项目的数据缓存或自定义模型。

## 两个预先固定开发划分

沿用Stage3.5已冻结的`val_biwi_hotel.json`和`val_crowds_zara02.json`，未重新按成绩选择录像。训练仅用其他录像的**原train时间区域**；留出的录像任何时段都不进入训练；其他录像旧validation也没有被提升为train。验证仅用留出录像的原validation区域，未扩成其全录像。既有20帧支持和隔离区保持不变。

|留出录像|split|manifest窗口|有监督窗口|完整8/12目标窗口|录像×行人ID单位|
|---|---|---:|---:|---:|---:|
|HOTEL|train|2221|2121|27145|892|
|HOTEL|validation|149|94|318|28|
|ZARA2|train|1959|1684|23545|833|
|ZARA2|validation|172|172|1092|35|

A/B各自从原始数据生成索引后进行集合核对，以上四项的监督目标集合完全相同，only-A/only-B均0。两者输入信息不同，所以相同目标不代表相同协议。A没有训练；B只在ZARA2划分进行5轮工程smoke，HOTEL没有神经训练或评价。ZARA2训练含HOTEL旧train区域，因此不能拿该checkpoint去HOTEL宣称录像隔离泛化。

源码审查和manifests可以包含ETH封存状态元数据；加载器只有显式五录像allowlist，没有ETH预测分支。完整时域支持为s至s+190，B预测支持仅s至s+70。不存在训练/验证跨录像重叠，且没有偷偷搬动旧验证区域。

## 历史上下文与运动信息覆盖（B）

|留出录像/split|平均历史上下文行人|有新进入邻居的目标|近邻运动机会|未知速度屏蔽机会|完全失去历史社会支持的目标|
|---|---:|---:|---:|---:|---:|
|HOTEL train|41.509|24684 / 27145|784813|4798（0.6114%）|10（0.0368%）|
|HOTEL validation|14.041|315 / 318|2344|46（1.9625%）|0|
|ZARA2 train|46.001|22213 / 23545|714314|4715（0.6601%）|9（0.0382%）|
|ZARA2 validation|10.502|688 / 1092|18922|73（0.3858%）|2（0.1832%）|

“历史上下文”统计8帧内出现过的其他ID，尚未套2米半径；“近邻运动机会”是第1–7帧的2米内可见关系事件，不等同于上下文人数。第0帧合法纯位置初始化另外记录。单次关系可跨时间、目标和重叠窗口多次计数，不能解释为独立人数或独立交互事件。

新进入邻居定义为首次历史可见时刻>0；不使用未来GT定义。只有1帧历史的邻居保留原始存在性和缺失标记，但没有可假造的速度；完整8帧监督目标本身不会被删成短历史难例。2个ZARA2目标全部社会消息无效时仍保留并由原网络自身运动路径预测，未从指标中删除。

A保留官方首出现补值行为，其相同原始历史上的“未知速度机会”列仅是对照审计量，**不代表A实际屏蔽了这些消息**；真实屏蔽规则只属于B。A目标索引另外保存未来-only/dummy槽数及完整官方邻居张量hash，不能混淆A的输入信息范围。

## 文件与hash

- 冻结协议/超参数：`configs/stage5p1/protocol.json`。
- 两套窗口manifest：`configs/stage5p1/manifests/{录像}/{A,B}.json`。
- 对应完整目标ID、窗口、原始历史/未来及输入hash：`results/tables/stage5p1/manifests/{录像}/{A,B}_{train,validation}_targets.csv.gz`。
- 原始文件SHA、manifest SHA、索引SHA、官方loader实际扫描耗时：`data_execution.json`。
- 逐窗口逐行人可观测性/速度/加速度bitmask：`observability/*.csv.gz`及`observability_index.json`。覆盖58,454个行人窗口；这不是58,454个独立物理行人。
- 样本总计和两协议目标集合检查：`data_summary.csv`、`protocol_target_set_comparison.csv`；近邻覆盖门槛：`coverage_gate.json`。

压缩目标manifest约16MiB，保存身份、支持范围与hash，不包含原始坐标，不是训练tensor缓存。原始数据、所有checkpoint和临时tensor均忽略Git。窗口JSON和压缩索引保留完整队列，没有为了轻量只保留改善样本。

HOTEL/ZARA2均已在历史实验中曝光；留一录像隔离是数据泄漏控制，不使它们重新成为独立确认集。当前没有独立最终确认或五场景正式训练，ETH继续封存。
