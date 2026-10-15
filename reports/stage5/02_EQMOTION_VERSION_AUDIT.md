# EqMotion：版本、checkpoint与协议停止审查

**结论：EqMotion暂不可用作严格官方checkpoint复现基础，本阶段结束调试。** 没有删除额外权重、补随机参数、改类别数、增加GRU或调用非严格加载。模型真实轨迹预测执行次数为0，ADE/FDE未产生。

## 官方可达历史

完整bare clone保存在`experiments/stage5/repos/eqmotion.git`，不改变Stage4冻结源码。检查官方heads/tags与所有28个可达提交。唯一公开分支为main；未发现tag或另一套可达ETH/UCY实现。`eth_ucy/model_t.py`、`eth_ucy/gcl_t.py`、`main_eth_diverse.py`在初始提交尚不存在，自`4c082a7eaa9b797a81d52b57c81b505d05de5e4c`加入后，27个含代码提交的三项blob均完全相同，直到当前`5aec2e0b61c511fa93a24138dd90da59a089084b`。

|文件|所有含代码提交共同的Git blob|
|---|---|
|eth_ucy/model_t.py|`4b73076e7c8b138cec2efdddb510ef2c538683dd`|
|eth_ucy/gcl_t.py|`cac343befcf262851c8354e587f86b68c4989796`|
|main_eth_diverse.py|`4e0a56d8963c960bb05d6292c49ddda885acd852`|

逐提交清单为`eqmotion_history.csv`，远程refs为`eqmotion_remote_refs.txt`。官方issues API限流403；定向网页检索未找到作者提供的匹配版。这个结论限于公开可达版本，不包括作者私人代码、不可达对象或尚未提供的修订，不宣称已穷尽一切可能。

## 真实严格加载结果

调用当前官方类、4层、hidden_nf64，按Stage4已有官方配置分别使用hid_channel64/128；category_num保留源码硬编码4。

|fold|总参数|预期/保存键数|缺失|多余|共有键形状冲突|strict加载|预测|
|---|---:|---:|---:|---:|---:|---|---|
|HOTEL|3,027,268|870 / 1269|0|399|0|失败|未运行|
|ZARA1|5,536,260|870 / 1085|184|399|0|失败|未运行|

HOTEL checkpoint SHA256：`7505fd78ec53755e665ba291245ee2cd12059314b1f5308383f77f1f95c50025`；ZARA1为`e8140e2b0ad30eea9038c0ff1e7e126624bbc94256b6cdbfc6ae31f3a83a274b`。完整错误、键清单见`eqmotion_strict_loading.json`；不要根据本表手动删键。

399个额外键中包括embedding_gru的8项以及23个图/输出层各17项，其中有att_q/att_k等训练模块。即使HOTEL共有参数形状一致，也不能证明丢弃额外模块后的计算图等价于权重对应训练网络。

ZARA1缺少各gcl/head的category_mlp第2/3类分支，合计184项。进一步检查发现：HOTEL这些ModuleList有0–3类，ZARA1仅0–1类；但**两者根部关系分类器的输出矩阵均为[4,64]**。因此“把全局类别数改为2”也没有官方版本依据，且会制造其他形状差异。这里仅从键推断局部类别分支，不把根部Sequential层编号误认成类别。证据`eqmotion_category_metadata.json`。其他非ETH fold权重未加载，不填写其类别结果。

## K20与上下文预算

源码输出`[B,N,20,12,2]`，20条轨迹来自学习的不同head，不是随机高斯重复采样。官方评价取第0个ego，对每个目标分别取最小ADE与最小FDE，获胜head可以不同；固定head0才是可命名的K1，不能把20头最优结果当确定性预测。

`process_eth_data_diverse.py`覆盖total_num：UNIV为2（含ego），ZARA1为3，其余fold为1。因此HOTEL和ZARA2在该脚本下只有ego，不能用其成绩验证社会图价值。邻居原先按最后历史位置距离和5米阈值筛选，但完整轨迹筛选依赖未来存在性，不能称历史-only邻居协议。若增加节点预算，将改变官方数据分布，不能作为不加说明的原版复现。

训练脚本有机器专属数据路径、测试入口落入训练及以test FDE选模的风险。Stage5没有运行该入口，也没有读取ETH数据/指标。本轮没有架构兼容修复、正式训练、真实checkpoint forward，故没有RTX3080吞吐或性能结论。

来源：[官方冻结代码](https://github.com/MediaBrain-SJTU/EqMotion/tree/5aec2e0b61c511fa93a24138dd90da59a089084b)、[正式论文](https://arxiv.org/abs/2303.10876)、[Stage4已记录审查](../stage4/03_EQMOTION_AUDIT.md)。复现严格报错已足以触发预设停止条件；下一步转向SocialVAE，不无限尝试环境或任意改网络。
