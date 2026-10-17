# 官方权重、从头训练入口与公开基线审查

## AgentFormer作者权重：严格加载通过，完整数值复现尚未执行

作者[README](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)给出公开Drive/Baidu链接。Drive先返回大文件确认页，按其公开下载表单正常下载，非账号认证绕过；517,512,541字节ZIP SHA256 `f192aa9596c25ec115dbf646f96a03f5b17e2c969016cd1b9c7e753db30941f0`。档案14个文件包含五折各CVAE/DLow以及nuScenes权重；本阶段未评价nuScenes。全部文件SHA与路径见 `official_checkpoint_acquisition.json`，只留被忽略cache。

| fold | CVAE索引/参数 | DLow索引/参数 | 双checkpoint strict |
| --- | --- | --- | --- |
| eth | 30/6,497,922 | 5/591,872 | True |
| hotel | 50/5,972,098 | 50/591,872 | True |
| univ | 70/5,972,098 | 15/591,872 | True |
| zara1 | 100/5,972,098 | 40/591,872 | True |
| zara2 | 70/5,972,098 | 25/591,872 | True |

没有strict=False。原DLow把CVAE置于普通list，DLow state_dict只含591,872个采样器参数，不能以DLow单个权重代表完整网络。原构造函数按pred_cfg/pred_epoch另严格加载CVAE，本轮又逐张量确认这部分等于独立CVAE加载结果。ETH多两个out_mlp_dim=[512,256]，参数比其他折多，内部traj_scale=2；不能用Hotel配置替换ETH。

[作者归一化说明](https://github.com/Khrylx/AgentFormer/issues/5)确认旧代码错误地使用未来坐标计算中心，2021-10-07更新了权重和结果。当前冻结源码scene_orig只来自历史。本轮严格加载证明结构兼容，作者链接与修正版README相符；**checkpoint没有可靠源码commit元数据，严格加载本身不证明训练全过程或已复现修正后的五折数字**。本阶段未运行完整模型测试指标。API抓取issue评论曾403限流，作者说明通过GitHub页面核实；该失败不抹去。

## 原训练与采样语义

原CVAE：Agent-Aware Transformer、历史context、FutureEncoder联合后验、learned prior、自回归FutureDecoder；MSE+KL(min_clip=2)+K20 sample loss，权重均1，Adam1e-4，100完整epoch，StepLR(10,gamma=.5)，每10轮保存。原loss不是精确边缘NLL。

原DLow：固定CVAE权重，q_MLP=[512,256]输出20×32的A/b，KL权重.1/clip10、diverse权重5/d_scale5、recon权重5；Adam1e-4，50完整epoch，StepLR(5,.5)，每5轮保存。train_w_mean=True时z=b；infer也mean=True，形成20个确定性context-conditioned latent head。不能把完整模型解释为20次CVAE独立先验采样。CVAE仍在计算图中，但不属于DLow optimizer参数；本轮确认它的权重没有变化，未擅自修改requires_grad策略。

官方train.py没有验证选择逻辑，只按固定周期保存；test.py默认取最新存在的checkpoint。作者ZIP选择的CVAE/DLow索引如表，不能用测试成绩去重选。未来从头复现必须预先冻结验证选模规则；这属于公开记录的训练管理差异，不改网络/loss，也不冒称精确复刻作者未公开的选模全过程。

原train/test/eval.py的完整imports和--help均在兼容环境实际运行成功。真实smoke执行原train函数AST原样函数体与原data_generator，限制为一个预设frame，不是完整epoch。官方源码0修改；兼容仅沿用_LinearWithBias=Linear和为loader增加运行时np.int=int别名。入口日志、严格权重日志和失败记录均保存。

## 四个公开模型的版本与可比性

| 模型/年份 | 输入与候选 | 数据与评价/额外处理 | 本轮证据与严格可比性 |
|---|---|---|---|
| AgentFormer ICCV2021维护修正版 | ETH/UCY历史二维坐标/位移；CVAE+DLow20头 | 作者八录像五fold；完整20帧上下文、允许单人；独立minADE20/minFDE20；无地图/FPC；仅尺度还原 | 五折权重strict成功、真实训练smoke，未完成五折评价；暂不能填HTraj标准成绩 |
| PECNet ECCV2020 | 历史位置、目标初始位置、预测endpoint与nonlocal pooling；20候选 | 论文ETH/UCY米制LOO；SDD配置与ETH notebook不能混用。供应的test先按终点误差选endpoint，再生成一条路径计算ADE，**不是独立minADE@20**；K20用放大采样σ，论文不做低K截断 | commit `a345ada1557f53dc2acb4c07f71d340d366980da`；论文表2已核对，ETH notebook为no_pooling且留有按test选best的示例，不能作为完整最佳社会模型的公平入口。无本轮训练/权重复现，不取较弱ablation值冒充最佳结果；不可直接严格比较 |
| SGCN CVPR2021 | 位移加历史时间通道；稀疏空间/时间图；20高斯样本 | 原完整20帧目标、人数≥2；分别minADE/FDE，无RGB。冻结作者历史训练脚本与论文训练超参不一致，Stage6还记录单人/排列与累积边界问题 | 历史完整代码 `5153702d8ae32a2b0586b093d4e094aee08b0f9b`；当前main已移除代码。表1核对，旧检查保留，本轮源码/论文复核而未训练；目标集合/实现差异未消除，不可直接严格比较 |
| Social-STGCNN CVPR2020 | 历史位移/图；二维高斯20样本 | 完整20帧且默认人数≥2；分别minADE/FDE；无额外RGB。TXP沿节点索引卷积，旧Stage4.1已发现顺序敏感 | 官方维护版本 `333d3a57b4d2705e129b21aefefa09c79b2b9ae1` 不等于精确CVPR2020原版；论文表2核对，旧checkpoint成绩不移植到本协议。无本轮重新训练；不可直接严格比较 |

三篇开放获取PDF与SHA在 `paper_source_checks.json`，作者README/源码获取状态在 `baseline_source_checks.json`。部分历史README URL 404、issue API 403已保留；论文本体与本地冻结源码足以定位上述结论。Academic MCP未挂载，采用作者arXiv/正式论文与仓库主来源，未用第三方排行榜替代。PECNet notebook的完整字节SHA固定在新cache，未执行其测试/训练代码。

## 文献报告值（ADE/FDE米；不是HTraj实测排名）

| 作者报告模型 | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | 作者AVG |
| --- | --- | --- | --- | --- | --- | --- |
| AgentFormer | 0.45/0.75 | 0.14/0.22 | 0.25/0.45 | 0.18/0.30 | 0.14/0.24 | 0.23/0.39 |
| PECNet | 0.54/0.87 | 0.18/0.24 | 0.35/0.60 | 0.22/0.39 | 0.17/0.30 | 0.29/0.48 |
| SGCN | 0.63/1.03 | 0.32/0.55 | 0.37/0.70 | 0.29/0.53 | 0.25/0.45 | 0.37/0.65 |
| SocialSTGCNN | 0.64/1.11 | 0.49/0.85 | 0.44/0.79 | 0.34/0.53 | 0.30/0.48 | 0.44/0.75 |

AgentFormer取[修正后的论文表1](https://arxiv.org/html/2103.14023#S4.T1)及README；PECNet取[原论文表2](https://arxiv.org/pdf/2004.02025v3)；SGCN取[原论文表1](https://arxiv.org/html/2104.01528v1#S4.T1)；Social-STGCNN取[原论文表2](https://arxiv.org/html/2002.11927)。全部为作者报告值，CSV显式标记author_report_only、directly_strictly_comparable=False。PECNet的ADE候选选择、多人过滤、公开实现维护差异决定不能简单按同K20就宣称同目标严格对比。只有以后用相同测试目标/规则完整评价，才能建立本项目标准基准表。
