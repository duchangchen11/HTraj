# 官方数据审计（smoke前冻结）

来源：[冻结AgentFormer官方数据](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/datasets/eth_ucy)。74/74文件、22个唯一Git blob全部核验；数据保存在独立且被Git忽略的 `data/raw/agentformer_official/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/`，没有以旧五录像替代。

八个来源为biwi_eth、biwi_hotel、crowds_zara01、crowds_zara02、crowds_zara03、students001、students003、uni_examples。官方17列格式的0/1列是采样frame/actor ID，13/15列是二维world坐标，其他多为占位；坐标米制与0.4秒采样据论文数据定义，转换脚本将原四列frame/10、坐标原样写入17列。文件字节追溯到作者发布；更上游原始annotation到作者预切分的全部生成操作未提供独立Git级追溯，不冒充重新标注或重建了原始录像。

官方preprocessor最少历史8、未来12；在20个连续整数采样格均存在的行人同时作为目标与上下文。缺帧目标不进入该完整轨迹任务；单人合法。dataloader按整个文件frame区间stride1枚举，包含会返回None的候选frame，统计区分有效窗口与候选frame。不是按独立行人数统计。

| fold | split | 有效窗口 | 目标窗口 | 最大人数 |
|---|---|---:|---:|---:|
| eth | test | 253 | 364 | 5 |
| eth | train | 3283 | 30307 | 57 |
| eth | val | 733 | 5422 | 42 |
| hotel | test | 445 | 1197 | 8 |
| hotel | train | 3118 | 29676 | 57 |
| hotel | val | 688 | 5203 | 42 |
| univ | test | 947 | 24334 | 57 |
| univ | train | 2719 | 9874 | 14 |
| univ | val | 622 | 2800 | 13 |
| zara1 | test | 705 | 2356 | 14 |
| zara1 | train | 2889 | 28577 | 57 |
| zara1 | val | 671 | 5184 | 42 |
| zara2 | test | 998 | 5910 | 14 |
| zara2 | train | 2681 | 26076 | 57 |
| zara2 | val | 590 | 4262 | 42 |

`official_data_audit.json`保存逐文件SHA/来源Git blob、轨迹长度分布、坐标范围、frame间隔、全部split清单；`official_window_inventory.csv.gz`保存完整窗口与目标ID。原官方loader的15个split均实际初始化，每个序列首个有效和最大窗口共148次检查与独立完整20帧枚举一致。

同源train/val文件frame集合无重叠；边界相邻（相隔1个0.4秒格），不是人为加隔离带。因为每个窗口必须全部20帧落在本split文件，完整支持不跨边界、没有共享frame；邻近时间相关性仍存在，不称独立跨录像验证。跨fold重复利用相同来源是标准留一场景设计，不应在不同折之间合并训练权重。

许可：官方代码MIT不自动赋予原始数据/权重再分发权；本轮不将轨迹、原论文PDF、权重入Git。ETH数据可用于新协议的来源/训练划分审计，但本轮不算ETH模型测试成绩。
