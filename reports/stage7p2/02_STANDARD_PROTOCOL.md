# 标准五场景协议（smoke前冻结）

用户在Stage7.2正式选择公开ETH/UCY基准。基础81a2e239e8ee79c07d19e319401c0f96c0b5e7ec；冻结提交c28350f。协议 `HTraj-AgentFormer-OfficialETHUCY8x12-K20-v1`：8历史、12预测、0.4秒、最终ADE/FDE米制。Stage7.1严格历史邻居协议及旧结果继续保留，不能与本协议混合排名。

**ETH规则切换：** 原“ETH从任何训练/开发中永久封存”改为作者五折；非ETH折允许ETH_train/val，ETH折排除biwi_eth、只在最终冻结后的ETH测试评价。测试成绩不得调参。本阶段仅审计数据与工程smoke，不算任何ETH模型测试指标。

[官方split](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/data/ethucy_split.py)：ETH测试biwi_eth，HOTEL测试biwi_hotel，UNIV测试students001与students003，ZARA1测试crowds_zara01，ZARA2测试crowds_zara02；每折从八个recording剔除对应测试录像，其余 *_train / *_val分别训练与验证。UNIV为6训练+6验证+2测试文件，其余为7+7+1。测试录像不入该折训练/验证；精确清单/SHA见结果JSON。

官方资格是完整20帧离线上下文，公开任务设计已声明使用未来存在性筛选；这与历史特征读取未来坐标是不同问题。本版本scene_orig仅用历史，不能恢复旧错误的history+future均值。没有再对官方模型强加Stage7.1仅历史资格规则。

**逐折配置不能混同：** ETH CVAE traj_scale=2，内部坐标除2，保存预测时乘2恢复米；ETH max_train_agent=32是作者原配置、训练中随机抽取32（非本轮为避OOM新增规则），且有额外out_mlp_dim=[512,256]；其余四折scale=1、默认max_train_agent=100。DLow的基模型置eval，不受其training分支抽样上限触发，测试亦使用全部合格节点。不能把Hotel配置直接代替五折所有模型。

完整模型必须包括CVAE与DLow，不能称只有pre-CVAE的20次普通采样为完整AgentFormer。官方预训练配置pred_epoch不同：ETH30、HOTEL50、UNIV70、ZARA1 100、ZARA2 70。DLow权重单独严格加载；其state_dict不含作为普通list存放的基础CVAE，需双checkpoint保存/恢复。

**工程smoke冻结：** 仅HOTEL/ZARA2官方train池中首个单人、首个多人、最大N57场景，各一次原始CVAE train()受限调用及一次原DLow train()受限调用，共最多12优化器更新，无完整epoch。官方generator只将sample_list限定为预设一个frame；原train函数、模型、Adam、原loss、原调度器不改。DLow使用本次随机初始化并做1次更新的工程CVAE，不使用作者权重初始化；更改的仅是独立checkpoint引用/输出目录。该文件索引1不是完成1个正式epoch。若非有限或OOM保存现场并停，不削减人数/改loss/无限重试。

本阶段不跑五折正式训练、不据smoke误差选模型。正式训练计划与验收GO/NO-GO在真实测试后独立交付；即使GO，也需下一阶段才执行完整训练。
