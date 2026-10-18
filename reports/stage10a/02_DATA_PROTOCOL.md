# HTraj-RiskGraph-Causal8x12-v1：冻结数据契约

监督目标集合逐项复用Stage5.1的`HTraj-SocialVAE-Causal8x12-v1`，模型输入接口独立。8历史/12未来，0.4s（2.5Hz）、米制，神经输出K1确定性回归。本轮没有训练，也没有新神经模型验证ADE/FDE。未来若做K20基准须重新立项，不能拿本轮K1框架与公开minADE20直接排名。

唯一manifest为`configs/stage5p1/manifests/{biwi_hotel,crowds_zara02}/B.json`，hash在`results/tables/stage10a/data_audit.json`。它们来自Stage3.5录像隔离及旧时间支持区域：训练排除验证录像全部时段，其他录像只用旧train区域；验证只用被留出录像旧validation区域，未偷加旧validation到train。每窗口完整20帧支持为start..start+190 tick，历史为start..start+70，每步10tick。物理0.4s来自既有数据约定，不由frame数字自行推算。

HOTEL/ZARA2已经多次开发曝光，不是新的独立确认数据。ETH本阶段继续封存；历史官方test文件仅作为保护清单做不解释内容的字节SHA，不计算其指标。Stage9使用AgentFormer HOTEL折的其他来源train/val及OfflineComplete20，数据和目标完全不同，禁止新旧数字混排。

## 固定目标与历史上下文

目标须完整8帧历史和12帧未来，未来只定义监督资格/GT。上下文C只含历史8帧出现过的ID，部分历史保留、未来是否存在不影响入选。历史位置mask为真即允许接收/发送**位置信息**；相对速度只有两端各自连续两帧可见才有效。此规则区别于Stage5.1原SocialVAE固定6维接口屏蔽未知运动packet：本轮显式known通道允许位置-only消息，所以目标协议兼容不等于模型输入完全相同。

缺失速度以数值0占位并附显式标记，绝不解释成真实静止；风险退回当前距离，不能推断未知者正在接近。重新进入者不继承旧GRU或社会carry；没有插值或窗口前速度。`inference_inputs(recording,start)`只查询8个原始frame，函数没有未来参数。标签函数独立，监督mask不筛掉上下文节点。

## 集合核对与统计单位

|留出录像|split|窗口/有监督窗口|目标窗口|source×person ID|每人最多窗口|上下文N最大|
|---|---|---|---|---|---|---|
|biwi_hotel|train|2221/2121|27145|892|432|79|
|biwi_hotel|validation|149/94|318|28|68|22|
|crowds_zara02|train|1959/1684|23545|833|307|79|
|crowds_zara02|validation|172/172|1092|35|154|15|

四份`*_targets.csv.gz`逐行保存`sample_id/source/frame_id/person_id/history_SHA/GT_SHA`，还保留完整上下文位置/mask/ID SHA；与Stage5.1全部历史/GT哈希完全相同。`sample_id=recording:start:person`，frame_id是窗口起点，last_observed_frame另存。哈希算法沿用`array_sha`：dtype+shape+连续字节；上下文ID和mask另散列。训练来源交集只在不同fold之间存在，单fold内train/val录像相交为0；两个fold不能当完全独立重复实验。

同一人的窗口高度重叠。未来必须同时报告窗口加权、source×person先平均后的等权、两录像宏平均，配对误差按同sample/ID/GT计算；不能把50,690训练目标窗口称为50,690独立行人。Stage5.1监督集合是当前冻结基准，不按新图规则删除难例。

阈值审计只用来源训练窗口的历史位置，不用验证误差/未来轨迹。对重复fold窗口按source+start去重，取每时刻最近可见邻居距离的q95，向上取0.5m网格并限制[2,8]m；邻居数为该半径下度数q75向上取整并限制[1,8]；风险距离尺度取最近距离中位数按0.5m取整并限制[.5,2]。结果radius=3m、K=8、sigma=.5m，H=4.8s按任务时域固定，速度数值阈值1e-4m/s。详细分位数/原始5文件SHA及旧manifest SHA见data_audit，不从验证ADE选择参数。

因果支持结论局限于本接口和测试覆盖：原始数据反事实删除/替换所有12帧未来、插入纯未来ID，HOTEL/ZARA2真实窗口的输入与三图逐位相同；人工原始记录反事实还验证随机权重预测逐位不变。它不证明完整实际社会影响，也不把监督资格使用未来标签错误称为输入泄漏。
