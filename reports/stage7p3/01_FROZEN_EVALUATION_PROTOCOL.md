# Stage7.3 固定权重五场景评价协议（评价前冻结）

基础提交：`507ef5ca613663ef4c31ea45a7090a4b28ed3cd0`。开发分支：`feat/stage7p3-agentformer-official-evaluation`。

本阶段零训练、零优化器更新。AgentFormer官方维护版固定为`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`，源码保持不变。使用Stage7.2作者压缩包中的五组CVAE+DLow原权重，全部strict=True恢复；DLow中的CVAE未注册在DLow的state_dict中，因此必须分别核查两份checkpoint及其配置。权重索引不是由本地测试成绩选取。

协议为`HTraj-AgentFormer-OfficialETHUCY8x12-DLow20-v1`：历史8帧、未来12帧、0.4秒、最终米制。按官方完整20帧存在性确定上下文与评价目标，允许未来存在性参与离线上下文选择。与历史严格因果协议分开，不混合排名。ETH封存规则已由用户授权的标准五折规则替代；测试成绩不参与任何模型、超参数或权重选择。

执行次序固定为ETH、HOTEL、UNIV、ZARA1、ZARA2，串行完成全部预测，再统一评价。保留原始test.py三个函数（AST提取原函数体，避免其主入口strict=False和自动评价行为）；仅增加严格加载、有限值、形状、目标、文件检查和计时。原始网络、DLow、输出序列化、损失均不修改。

DLow使用原始mean=True语义：20个可学习latent heads的z=b；不称为20个独立CVAE prior随机样本。只评价`samples`，不使用`recon`。ETH由原serializer乘traj_scale=2恢复，其他折为1。坐标原样保存到三位小数，不加FPC、GT排序或后处理。原配置seed=1；记录实际软件、CUDA/cuDNN及运行标志。

minADE@20与minFDE@20逐目标独立选择最优候选；窗口内及录像内按目标窗口加权，UNIV两序列按目标数汇总，五场景宏平均等权。第一候选K1仅为辅助值，不是数学期望。重叠目标窗口不视为独立统计样本，不做错误的独立样本显著性检验。

同一份保存预测先作严格ID/frame/K=20检查，再运行官方eval.py和独立字典GT对齐、float64计算；全精度容差1e-5米，官方CLI四位小数容差0.000051米。两者共同承受原serializer三位小数影响，不把序列化前预测与后预测混评。

预先定义初步工程复现门槛：五场景及宏平均的ADE、FDE与作者两位小数参照的差值均不超过max(0.01米,作者值×5%)，且所有输入/权重/目标/资源/评价核验通过。另记录是否落入作者两位小数的±0.005米区间。此门槛仅用于初步复现验收，不是性能提升标准，也不要求未知的作者全精度逐位一致。

OOM、非有限值或目标错配则停止该场景，保留现场，不删拥挤场景、不减K、不改权重或配置。未完成的场景不补假成绩。不根据测试成绩修改任何设置。

冻结清单与执行源码SHA见`configs/stage7p3/evaluation_protocol.json`，冻结目标见`configs/stage7p3/test_windows.csv.gz`。正式推理入口会拒绝未提交工作区，并核实源码、协议、配置已真实存在于当前提交。实际冻结commit由执行记录保存；只有确认提交成功和工作区干净后才执行。

来源：[官方版本](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)，该版本的`test.py`、`eval.py`、`model/dlow.py`、`data/ethucy_split.py`及Stage7.2本地SHA核验记录。
