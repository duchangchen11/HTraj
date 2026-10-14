# ETH/UCY任务与划分核查

日期2026-10-15；基础46ecb91169eeadb5ba7ec28b7c648dd0ace837c6。原Stage1–3全部源码/脚本/配置/报告/结果和90个checkpoint共606项SHA保护，不覆盖旧文件。只读取HTraj本地已核查公开四列world轨迹，不使用PedNegoNet自定义模型/缓存。

## 文献与官方代码来源

核心外部参考正式名称：**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**，Abduallah Mohamed、Kun Qian、Mohamed Elhoseiny、Christian Claudel，CVPR2020，14424–14432。[作者稿§5–6](https://arxiv.org/abs/2002.11927)、[官方仓库](https://github.com/abduallahmohamed/Social-STGCNN)。CVF页面/PDF此次403，使用作者arXiv全文；下载PDF只存ignored目录，SHA在external_source_supplement.json。

论文说明0.4s采样、8观察/12预测、ADE/FDE和20样本选优。关于划分，其§5文字描述同数据集部分训练/剩余测试、其它四集验证；[Social GAN作者论文§4](https://arxiv.org/abs/1803.10892)明确四集训练、第五集测试。**不能把所有文章都统一说成完全相同预处理和划分**。GitHub API最初403限流，后通过公开Git过滤bare clone读取锁定版本文件树，完成五个fold的文件名/对象OID核查；未读取轨迹blob或ETH指标。记录在external_official_split_inventory.json。

[官方锁定datasets目录](https://github.com/abduallahmohamed/Social-STGCNN/tree/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/datasets)显示ETH/HOTEL/ZARA1/ZARA2各7个train、7个val、1个test文件，UNIV各6个train/val及2个test文件。各fold train/val录像名一致、与test录像名隔离，符合代码中的留场景测试；论文文字与代码文件清单应分别记录。特别是ETH fold除HTraj五段非ETH录像，还使用crowds_zara03与uni_examples训练/验证。HTraj并未采用这两个recording，训练人口不同。该清单只证实文件组成，尚未逐文件核查processed内容、时间边界或加载样本一致性，**不是官方协议完整复现**。

官方README指定CVPR版本`ebd57aaf34d84763825d05cf9d4eff738d8c96bb`，另锁定原HTraj raw来源`333d3a57b4d2705e129b21aefefa09c79b2b9ae1`，各utils/test/model/train/metrics/LICENSE SHA在external_reference_sources.json。README.MD大小写初始404后已修复下载；软件为[MIT许可](https://github.com/abduallahmohamed/Social-STGCNN/blob/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/LICENSE)，不代表数据再分发也获MIT授权。

## 外部模型静态核查（未复现训练）

- [utils.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/utils.py)：complete20-frame行人过滤，默认min_ped=1实际条件是人数>1，故至少2个完整行人；坐标round4小数；按全局可见frame列表滑窗，没有HTraj的显式空帧gap分段检查。full20筛选同时影响目标和上下文，是离线benchmark的人群定义，不等价于在线只凭历史的邻居资格。
- 同文件seq_to_graph：节点为相对位移，邻接使用不同人的**位移差范数倒数**，相同位移时off-diagonal返回0；自环1，再做规范化Laplacian。不能把它错误说成HTraj当前4m距离/CPA/k8图。
- [model.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/model.py)：ST-GCNN时空图嵌入与TXP-CNN外推，输出[B,5,12,N]的二维高斯位移参数。
- [test.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/test.py)：forward只传V_obs/A_obs；未来V/A为标签，未进入forward。高斯std=exp、相关=tanh、采样20次，再按每人分别最小化ADE与FDE，两项可来自不同样本。不能称配对同一best轨迹。
- [train.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/train.py)：默认SGD0.01、250epoch、累积batch128；与本阶段Adam0.001/30epoch不同。旧NetworkX接口在现代版本的兼容性需独立环境核验，不自动安装或套旧缓存。

此模型是后续可复现外部参考候选，本阶段仅完成身份/代码/协议/输出/许可审计，**没有其本地训练或真实评价值，没有公平性能排名**。

## 两套明确命名的HTraj评价协议

| 项目 | A_complete8x12_history_context | B_partial_mask_v1 |
|---|---|---|
| 任务 | 标准完整8→12目标对照，历史上下文版本 | HTraj原部分观测/部分未来监督 |
| 目标资格 | 历史8点和未来12点均完整 | cutoff可见且未来至少1个有效label |
| 邻居/上下文资格 | 仅历史完整8点，未来是否完整不影响输入 | 8点历史并集，原mask保留；实际图仍双端两点连续 |
| 未来-only行人 | 不进入输入 | 不进入输入 |
| 速度 | 连续两点差分，缺失明确标记 | 同原encoder策略，gap/reappearance reset |
| 时间 | 显式10frame-ID ticks=0.4s，20点支持，不压缩无人gap | 同原manifest与mask |
| 坐标 | 已核查公开world米坐标，无新插值、旋转或归一化 | 完全同原坐标 |
| ADE | 每人12步平均，再对目标平均 | 每人有效future步平均，再对目标平均 |
| FDE | 第12步全部目标有效 | 仅第12步有效目标，不能拿最后可见步替代 |
| K1 | 确定性单轨迹 | 确定性单轨迹 |
| 多模态指标 | 本阶段不适用minADE@20/minFDE@20/NLL | 同样不适用 |

A满足用户完整历史/未来的标准任务属性，但**不是官方loader/官方split/K20指标的逐项复现**。它允许单人窗口、历史可见但未来不完整者仍当上下文，不依未来存活选邻居。只在损失/评价端以future availability选目标，测试验证改future坐标/mask不会改变观察输入或社会分组。B是独立任务，不能混用两种分母，也不能把A平均低/高直接解释为模型改善。

K1多模态模型须额外定义分布均值还是一次随机样本；K20 oracle可参考GT选最小评价误差，却不能用GT选择模型输入或真实部署轨迹。HTraj现有确定性模型不因重复同轨迹20次而成为多模态模型。

8个历史点首末相隔2.8s，文献“3.2s观察”是8个采样步的长度约定；未来点距cutoff为0.4–4.8s，完整20点支持跨度7.6s。隔离按真实起止frame-ID支持计算，不用只比较窗口起点。

## 原数据身份、有效样本与新开发划分

原fold_ETH SHA `c52c85999b1fb401e1f862de06f9086290a0b4acdaa34bab8214731630914ced`。五个非ETHsource的raw SHA每个新manifest保留，loader逐次核实。原2781训练窗口：A2471有监督窗口/28022完整目标/336264标签点；B2778有监督窗口/46287 ADE目标/34575 FDE目标/484422标签点。其余窗口没有该协议合法标签，仅跳过训练loss，不将其伪称有效目标。

新建五个留一recording候选，train只取其它四个recording的**原train**；held recording原train全部不用，validation仅取该recording原val区域，其它原val保留。没有把原验证偷偷加入训练，没有新模型训练/验证recording交集。保留完整20点起止支持190ticks；原时间切割200ticks隔离不变，录像隔离比同recording时间隔离更严格。本次不是重新拼接全录像或全部非ETH标签训练。

首轮预选students003，不根据指标：训练来自biwi_hotel/students001/crowds_zara01/crowds_zara02。A2058有效训练窗口/19034目标；B2365/31458 ADE、23423 FDE目标。验证70窗口，A706 ADE/FDE，B1464 ADE/986 FDE。其余五fold人数/历史/密度/运动与关系完整统计见split_counts.csv和split_group_counts.csv。

这只保证**本次新模型**recording隔离；不保证物理场景或真实身份隔离（students001/003都属UNIV）。这些录像已被旧模型使用，70验证中11个窗口与旧96完全相同，时间支持重叠更多，所以是新的模型级录像隔离开发协议，**不是新的独立确认数据**。ETH外部测试继续封存，新manifest不含ETH split/source。后续整录像验证或更广LORO可研究，但本次未运行，不把当前70区域写成整录像泛化。

## 历史分组与配对

source src/evaluation/benchmark_audit.py：4m/k8/CPA图，密度沿用平均连续帧入度；相向cos<−0.7、同向cos>0.7、交叉abs cos≤0.7且双方速度>0.2m/s、cutoff候选真实关系。一人可同时有不同类型邻居，运动类别不是互斥总和。关系变化是持续候选pair的相对速度变化≥0.5m/s；稳定仅指该速度指标低于阈值，不证明邻居成员/距离不变；无有效变化观测单列。

所有分类只用历史/mask/timestamps。GT只计算误差与标签可用性，不用于构图或上述类别。各模型在同一tuple(recording,sample_id,agent_id)上严格对齐；额外非重叠支持敏感性按metadata预先greedy选20点支持并加8秒隔离，仍是同开发集的抽稀，不叫独立确认。

补充neighbor_eligibility_audit.json明确“no_neighbors”是原双端连续合法图为空，不是物理周围无人。新B该组51目标中48仅一帧历史、47有当前4m物理邻居、46有可计算运动的protocol源；接收条件阻断不能解释为不需要社会信息。新A空图7目标中5有原始历史可见物理邻居，但它们不满足A完整历史上下文资格。此补充只用历史位置/可见性/连续性，不修改冻结的主分组或训练数据。

physical_density_group_comparison.csv另提供不要求连续运动的原始历史4m物理密度组，四组覆盖同一评价人口且逐目标配对。训练后补充探索，不用于选择模型；完整已观察历史无可见4m邻居仅A0/B2，不能据原合法空图组声称真实孤立目标普遍不需要社会信息。全部数值及定义见03_INTERACTION_VALUE_ANALYSIS.md。
