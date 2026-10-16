# 两个新公开候选：版本与选择

本阶段新正式训练=0、优化器更新=0、ETH评价=0。所有新神经输出均为固定随机权重的人工输入功能验证，不是HTraj真实ADE/FDE或论文性能复现。官方预训练权重没有下载、没有严格加载；严格恢复只指本轮随机state_dict往返。模型源码保持原始Git blob不变，依赖兼容别名、包装器与检查记录独立保存。


只详细检查AgentFormer、PECNet。引用的ICLR2022归因论文仅为相关工作，不是第三套候选实现。核查日期使用项目客户端日期2026-10-17。

|维度|AgentFormer维护版|PECNet公开版本|
|---|---|---|
|正式来源|[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html)，[arXiv v3](https://arxiv.org/abs/2103.14023v3)|[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470749.pdf)，[arXiv v3](https://arxiv.org/abs/2004.02025v3)|
|作者源码commit|`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`|`a345ada1557f53dc2acb4c07f71d340d366980da`；旧PECNet URL重定向Human-Path-Prediction|
|许可|MIT/CMU，保留原许可|MIT，保留原许可|
|结构|共享时空Transformer，区分同/不同行人注意力，显式agent_mask|个体终点CVAE，显式N×N非局部关系池化，终点条件路径MLP|
|输入/输出|8帧完整位置/位移；12帧2D，K个联合预测|8帧展平位置、末观测位置、历史mask；11中间点+生成终点=12帧|
|已测参数|5972098|2099692|
|从头入口|[train.py](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/train.py)，ETH/UCY配置齐全；先CVAE再DLow|[training_loop.py](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/a345ada1557f53dc2acb4c07f71d340d366980da/PECNet/scripts/training_loop.py)，公开主体主要是SDD|
|公开权重|作者提供Drive/Baidu链接，未下载/核验|saved_models文件可定位，未下载/核验|
|8/12及单位|ETH配置明确8/12，原位移不是m/s；0.4秒/米必须按数据源核验|SDD optimal.yaml使用scale1.86、initial_pos/1000，不能直接当ETH米制协议|
|实际RTX3080检查|N1/2/5/12及最大源上下文N57原始K20损失反向可运行|N1/2/5/12 float64原始损失反向可运行|
|主要剩余问题|官方完整20帧上下文使用未来存在性，不能直接称严格历史因果loader|官方评价先用GT终点选样再社会插值；跨代理oracle上下文与独立minADE不是同一评价|

**唯一保留的后续复现主干：AgentFormer-CVAE原公开pre阶段。** 选择依据是明确ETH/UCY入口、完整时空交互、排列测试和最大上下文反向通过，非论文成绩。不是已经完成其全部两阶段论文复现，也不是立即可开发创新算法的结论。PECNet核心可运行，但SDD单位/异步关系批次、外置ETH数据处理与GT选终点评价需要更多重建，本阶段不把它列作第二个推荐主干。

AgentFormer作者曾确认旧版以历史+未来共同求scene_orig的泄漏，[官方问题#5](https://github.com/Khrylx/AgentFormer/issues/5)说明已修正并更新结果。当前源码scene_orig只取历史；NaN/大幅改变未来标签未影响实际learned-prior预测。必须冻结维护版，不沿用修正前paper数字。官方test.py本身strict=False，不能以脚本未报错宣布预训练匹配；本轮没有加载作者权重。

环境：Python3.10、torch2.5.1+cu124、torchvision0.20.1、NumPy1.26.4、RTX3080。glob2==0.7、easydict==1.13仅安装到新cache/deps，不改旧环境；AgentFormer原生import因私有_LinearWithBias移除失败，运行时同bias线性类兼容别名后通过，未修改源码、层宽、损失。完整数据loader仍有np.int旧API，下一阶段需独立兼容入口smoke，不能声称整条官方train.py已端到端跑通。

源码53个目标文件（43+10）逐项Git blob/SHA核验；AgentFormer初次3个SSL失败仅重试一次后成功，原失败记录保留。git ls-tree --long会触发partial clone懒取blob，首尝试约44秒后停止，改为只读tree元数据。学术MCP未挂载，OpenAlex429、CVF直接403、OpenReview挑战均保留说明；采用作者/arXiv/正式出版元数据交叉核查，不宣称穷尽检索。
