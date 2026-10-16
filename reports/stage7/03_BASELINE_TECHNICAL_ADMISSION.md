# 实际技术准入：模型、协议和失败分开

本阶段新正式训练=0、优化器更新=0、ETH评价=0。所有新神经输出均为固定随机权重的人工输入功能验证，不是HTraj真实ADE/FDE或论文性能复现。官方预训练权重没有下载、没有严格加载；严格恢复只指本轮随机state_dict往返。模型源码保持原始Git blob不变，依赖兼容别名、包装器与检查记录独立保存。

| 模型 | N | 有限前向/loss/梯度/恢复通过 | 固定z换序max | K20同z换序max | 未来NaN差 | 严格随机权重恢复 | 峰值分配MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AgentFormer兼容 | 1 | True | 0.000e+00 | 0.000e+00 | 0.0 | True | 86.99 |
| AgentFormer兼容 | 2 | True | 1.192e-07 | 2.384e-07 | 0.0 | True | 115.59 |
| AgentFormer兼容 | 5 | True | 2.384e-07 | 2.384e-07 | 0.0 | True | 222.10 |
| AgentFormer兼容 | 12 | True | 4.768e-07 | 4.768e-07 | 0.0 | True | 549.56 |
| PECNet原生 | 1 | True | 0.000e+00 | 0.000e+00 | 0.0 | True | 65.35 |
| PECNet原生 | 2 | True | 5.551e-17 | 5.551e-17 | 0.0 | True | 65.37 |
| PECNet原生 | 5 | True | 2.220e-16 | 2.220e-16 | 0.0 | True | 65.43 |
| PECNet原生 | 12 | True | 4.441e-16 | 4.441e-16 | 0.0 | True | 65.57 |


上表第一轮AgentFormer使用显式固定z，不能单独代表learned-prior全路径。补充AgentFormer_full_prior_checks.json保留p(z|H)的mu/sigma并按ID同步epsilon：N1/2/5/12均通过；mu/logvar也同步置换；K1 prior-mean与K20最大预测差均≤4.77e-7，GT污染差与严格恢复差均0。原始Normal.sample临时只替换标准噪声，loc/scale/模型/decoder不变，退出即恢复。单人/多人数据适配单测验证了同步mask和逆置换，不用未配对随机数制造差异。

N57是既有非ETH源训练manifest的最大上下文人数；原始三项loss含K20、完整前向+反向一次均有限。实测allocated `6391504896` bytes、reserved `9659482112` bytes，总检查约2.077秒（含初始化，非每训练窗口吞吐）。显存余量小，正式运行须串行；不能由N12的550MiB外推所有真实场景成本，不默默删除拥挤窗口。

## 未来信息隔离的两层结果

预测函数层：agentformer_packet只接受[N,8,2]，历史外的20帧包被拒绝；共享容器补零future及零mask以满足官方接口。训练后验才显式附加标签。原始inference(mode=infer)的先验/decoder不使用fut数值，改变为NaN或远移1000也不改变预测。`mode=recon`会使用未来后验，禁止作为可部署K1；官方test虽额外保存recon，但评价目录是samples，不能误报其官方samples就是GT重建。

数据选择层：直接执行原preprocessor.get_valid_id的方法体，保持全部历史相同、仅删除邻居某一未来帧，其上下文ID从[0,1]变成[0]。因此**官方loader未通过严格历史上下文准入**。这是完整20帧离线筛选与历史因果任务的差别，不能用模型层GT数值隔离掩盖。后续冻结离线A任务要明确承认未来存在性条件化，不能把它命名HTraj-Causal8x12-v1；若论文坚持严格历史可部署上下文，本轮整链路No-Go。

## 保留失败与限制

首次完整单测4失败/381通过，原因是测试自身用负stride NumPy反向索引Torch；改为连续copy后，最终385项通过，first_full_tests.xml/txt均保留。AgentFormer原生import失败日志也保留；只是机械兼容修复，不宣称原环境无问题。未运行作者预训练、真实ADE/FDE、优化器更新或正式epoch。人工“loss有限”不等于真实收敛，也未验证DLow阶段。候选源码、脚本、依赖报告与原始JSON全部交付。
