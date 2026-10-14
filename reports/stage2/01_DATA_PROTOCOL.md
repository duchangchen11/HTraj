# Stage 2 数据协议与核查

日期：2026-10-14。独立实现 `src/data/{preprocessing,dataset,collate}.py`；只读取旧目录中的公开轨迹 TXT 及来源元数据以核查身份，没有导入旧 DataLoader、模型、缓存、权重或结果。

## 文件身份与原始格式

6 个 TXT 共 66676 条观测，逐行检查为有限的 `frame_id pedestrian_id x y`，帧号和 ID 为整数，无重复 frame/ID。全部文件 SHA256 与固定上游 Social-STGCNN 提交 `333d3a57b4d2705e129b21aefefa09c79b2b9ae1` 的公开文件在线逐字节核对一致；详细 SHA、URL 在 `configs/data_manifests/stage2_v1/upstream_verification.json`，本地内容审计在 `raw_audit.json`。

这里的“原始轨迹文件”指公开分发的四列 world-coordinate 表，不是摄像机视频或未经坐标转换的原始注释。只复制这 6 个 TXT 到 `data/raw/eth_ucy/`，它们被 Git 忽略；新 manifest 自行生成，未复制旧预处理产物。

| 场景 | 录像 | 行数 | 录像内唯一行人 |
|---|---|---:|---:|
| ETH | biwi_eth | 5492 | 360 |
| HOTEL | biwi_hotel | 6543 | 389 |
| UNIV | students001 | 21813 | 415 |
| UNIV | students003 | 17953 | 434 |
| ZARA1 | crowds_zara01 | 5153 | 148 |
| ZARA2 | crowds_zara02 | 9722 | 204 |

UNIV 的两个录像保持独立，绝不拼接时间或同号行人。行人身份必须与 recording 组合解释，不能把跨录像相同 ID 当一个人。

## 时间和单位的证据边界

内容实测连续有效帧通常间隔 10；ETH/HOTEL/ZARA1 有无人的大间隔，其余录像连续。协议将这些间隔分段，不能压缩为相邻步。公开论文明确使用 0.4 秒采样、8→12（[Social-STGCNN §4.1](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf)）；world coordinates 为米也与 [Y-net 作者论文材料中的 ETH/UCY 协议](https://digicoll.lib.berkeley.edu/record/280995/files/EECS-2023-282.pdf) 一致。

时间映射显式使用 `seconds = frame_id * 0.4/10`，不是从目录名推测帧率。坐标不再变换。数字表本身不含单位/绝对时钟，无法仅凭 4 列数值重新证明相机标定；单位和物理时间来自已核对分发身份与公开协议，数值范围及速度检查提供一致性证据。ETH 速度中位数约 2.45 m/s、最大约 6.48 m/s，明显高于其它场景，保留并报告，不为改善成绩自行删掉。

8 个观测点的首末时间跨度是 7×0.4=2.8s；“3.2s 观察”是文献的采样长度约定。预测点相对最后观测为 0.4–4.8s。保留原帧号、观察和未来时间戳。

## 可观测集合与未来监督

每个窗口的行人集合取 8 帧历史中出现过的 ID 并集；未来-only 人不会进入输入。历史 mask `[8,N]` 与未来监督 mask `[12,N]` 分开存储。截止时已消失的人保留为历史上下文，但不作为预测目标；截止时存在且未来部分可见的人按每帧标签 mask 监督。位置缺失填零，不插值；encoder 忽略 masked 坐标，缺失/重现清零状态。

因此这不是“先筛出完整 20 帧行人再构图”的旧式完整轨迹过滤，样本人数和 ADE/FDE 人群不应与文献表直接横比。未来标签 mask 仅用于损失/评价，不能传给模型或改变邻居选择。网络签名只接受观察、历史 valid、历史 timestamps。

padding 输出 `[B,8,N_max,2]` 和 `[B,12,N_max,2]`，对应 Boolean mask；padding ID=-1。保存场景、录像、行人、sample ID，不跨场景建边。

## 五个独立留一 manifest

每个 `fold_<SCENE>.json` 独立列明 raw hash、训练/验证/测试窗口、起止帧、时间 cutoffs。非测试录像的前 80% 时间段用于训练；验证起点为 cutoff 后额外 20 个采样步。所有跨 cutoff 的完整 20 帧窗口都舍弃；训练和验证的完整时域支持不重叠，验证绝不含测试场景。窗口 stride=1，重叠窗口保留在同一 split。

| 测试场景 | 训练窗口 | 验证窗口 | 测试窗口 |
|---|---:|---:|---:|
| ETH | 2781 | 548 | 582 |
| HOTEL | 2681 | 482 | 717 |
| UNIV | 2492 | 511 | 947 |
| ZARA1 | 2591 | 524 | 796 |
| ZARA2 | 2419 | 459 | 1033 |

时间隔离避免重复片段泄漏，不保证同一行人在长时间训练/验证中永不重复出现；不宣称身份泛化划分。划分不用 future 标签决定输入人群，但完整窗口仍受录像长度和时间段可用性约束。

Smoke 固定 ETH 为 held-out 场景，仅用其余场景的训练/验证；保存 256/96 窗口索引及 sample IDs，种子17。数据加载器只读取当前 split 涉及的原始录像，不加载 ETH 测试文件。五折 manifest 只是设计交付，本阶段不做五折训练。

## 复现

已有文件禁止覆盖；重建时使用新的 manifest 输出目录。

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/prepare_stage2_data.py --manifests configs/data_manifests/stage2_v2
```
