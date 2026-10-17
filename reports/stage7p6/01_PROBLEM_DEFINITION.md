# 问题定义与历史证据复核

结论：问题真实存在于已检查的动态轨迹数据，但“给所有历史邻居补零未来再mask损失”不是合法修复。本阶段不更改Stage7.1 NO-GO，也不否定Stage7.3官方完整20帧协议的权重复现。

固定源码`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；本轮研究脚本不运行网络、不训练、不重算任何测试指标；完整回归测试包含既有人工前向/后向检查。Stage7.2的标准协议取代旧ETH封存规则；本轮仍不访问ETH评价误差。Stage7.3测试集已经被观察，不能称为未来未接触确认集。

## 三种不同问题

|问题|定义|现有证据|不能推出的结论|
|---|---|---|---|
|未来GT数值泄漏|固定历史/资格，推理结果取决于未来坐标|Stage7.1严格历史适配器10项原始数据反事实，K1与固定噪声K20差值全部0|不是证明官方get_valid_id仅用历史，也不是证明所有预处理都严格因果|
|未来存在性资格依赖|未来记录是否存在改变入模节点集合|官方get_valid_id同时检查8帧历史与12帧未来；改变集合还改变scene_orig及attention上下文|完整20帧离线benchmark已声明该条件，不能将其偷偷等同在线历史场景|
|训练后验缺标签|历史可见节点无合法完整future，原后验仍要求12帧token|FutureEncoder未使用fut_mask；mask-only人工补值影响有标签目标的q|不代表训练后验合法使用监督目标GT本身是泄漏；不代表所有CVAE都无法解决|

源码：[资格](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/data/preprocessor.py#L80)、[后验](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/model/agentformer.py#L171)、[损失](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/model/agentformer_loss.py#L1)。本地逐段SHA见`static_source_anchors.json`。

## C、S和未解决范围

C为当前窗口8帧均可见的行人ID交集，仅由历史决定。S⊆C为有完整12帧标签且属于冻结监督目标集合的节点；U=C\S。记录未来是否存在只用于训练标签与事后评分资格，不参与C构建或推理前向。这里故意不填补历史不足节点；不解决未来新行人生成、birth/death、身份切换。离开视野不等于停走或物理消失。

下表仅复用Stage7.1的Stage5.1支持区域，并筛选|S|>0窗口。不能外推为官方74文件发生率：

|fold/split|有监督窗口|C节点窗口|S目标窗口|缺未来节点窗口|混合标签窗口|
|---|---:|---:|---:|---:|---:|
|biwi_hotel/train|2121|37502|27145|10357|1818|
|biwi_hotel/validation|94|713|318|395|94|
|crowds_zara02/train|1684|32920|23545|9375|1440|
|crowds_zara02/validation|172|1486|1092|394|123|


全部支持窗口另见`saved_cohort_summary.csv`：HOTEL train2221/val149，ZARA2 train1959/val172；有监督分别2121/94、1684/172。差额是|S|=0窗口，不能参与监督平均。逐行核验发现这些支持区域中完整future节点恰好都属于S，未将该偶然一致提升为普遍定义。HOTEL验证94个有监督窗口全部含缺未来邻居；ZARA2为123/172。计数单位为窗口及节点窗口，不是独立行人数，不能按数万个独立样本做显著性推断。

最大历史完整上下文C=73、S=53、缺完整future20，位于students001训练支持start30。该73人不是HOTEL验证人数，也不是官方74文件重新扫描最大值。短历史节点仍被排除，数量另表保存。

## 不改变原失败结论

Stage7.1 HOTEL(start14950,target344,neighbor343)将未知邻居补值从末位置改为末位置+100m，监督后验均值最大差0.5597405434，q对未知future梯度L1为0.1061537489/0.0149514638；ZARA2最大差0.4992415309（完整原始精度以JSON为准）。这只是非法补值负探针，绝非预测性能。两种HOTEL补值KL恰被clip为2，sample项相同，不能编造已观察的KL变化。NaN补值导致非有限loss，NaN×0不安全。

原始10个反事实、raw变体SHA及检查字段已复用至`counterfactual_reuse.json`；HOTEL shorten案例原本仅1帧future，与delete后效果相同，不算额外独立证据。7个完整标签真实batch loss/backward通过；3个混合标签case拒绝训练。适配器的prior推理通过，与原训练后验NO-GO同时成立。[原报告](../stage7p1/STAGE7P1_REPORT.md)

本轮没有统计“新增这些邻居会使预测更准”，真实问题的存在性与新增信息的预测价值是两个尚未打通的证据环节。
