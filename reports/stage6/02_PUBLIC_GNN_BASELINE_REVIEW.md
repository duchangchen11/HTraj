# 公开GNN与补充模型复核

## SGCN：版本与正式来源

[CVPR2021正式论文](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html)，Liushuai Shi等，8994–9003，[arXiv2104.01528](https://arxiv.org/abs/2104.01528)。作者仓库[冻结完整历史发布](https://github.com/shuaishiliu/SGCN/tree/5153702d8ae32a2b0586b093d4e094aee08b0f9b)，当前HEAD `0d4ac44f44a0688a839f25070be1a4faeed5ade1`已移除主要代码，不能把当前main称为可直接训练。MIT许可证；固定Git blob与下载字节SHA逐文件核验。未修改作者源码。

25,369参数，两份HOTEL/ZARA2 val_best各97个state键严格加载。只在RTX3080进行人工N=1/2/3/5/8/12前向，没有真实场景ADE、没有ETH评价、没有训练。官方非ETH权重可能使用ETH作为上游训练来源，不能用于HTraj严格非ETH公平性能结论。

SGCN有显式稀疏有向空间与时间关系，输入8帧逐步位移加时序位置通道，输出12帧每人5个二维高斯参数。空间图[8,4,N,N]，时间图[N,4,8,8]，图卷积两条空间→时间/时间→空间路径；最终共享时序预测。复杂度主要含O(TN²d+NT²d)，图关系/掩码存储O(H(TN²+NT²))，不是稀疏参数就自动线性复杂度。来源：[model.py](https://github.com/shuaishiliu/SGCN/blob/5153702d8ae32a2b0586b093d4e094aee08b0f9b/model.py)。

## 实际有限测试：成功与失败都保留

| 权重 | N | forward | 高斯参数最大差 | 均值路径最大差(m) | 空间邻接恢复差 | 单位图跨人Jacobian | 错误 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hotel | 1 | False | — | — | — | — | AssertionError |
| hotel | 2 | True | 0.021826 | 0.087599 | 0.343069 | — |  |
| hotel | 3 | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |
| hotel | 5 | True | 0.000000 | 0.000000 | 0.000000 | — |  |
| hotel | 8 | True | 0.000000 | 0.000000 | 0.000000 | — |  |
| hotel | 12 | True | 0.000000 | 0.000000 | 0.000000 | — |  |
| zara2 | 1 | False | — | — | — | — | AssertionError |
| zara2 | 2 | True | 0.244070 | 0.136025 | 0.539722 | — |  |
| zara2 | 3 | True | 0.031025 | 0.045553 | 0.520568 | 0.000000 |  |
| zara2 | 5 | True | 0.025004 | 0.075996 | 0.533186 | — |  |
| zara2 | 8 | True | 0.615389 | 0.461632 | 0.875365 | — |  |
| zara2 | 12 | True | 0.284101 | 0.152951 | 0.915425 | — |  |

原始高斯参数差是参数单位；人工历史以米构造，均值位移累加路径差才以米表示，但它不是实测场景误差。部分HOTEL人数例子差为0，不能掩盖N=2反例或ZARA2多个N反例。N=1由`squeeze()`引起rank断言；官方loader本来要求人数大于min_ped=1，因此它是单人接口限制，不自动等于论文宣称支持单人却失败。

**修正Stage5静态推测：SGCN的TXP输入实际为[N,8,4,16]，节点在batch轴。** 独立TXP换序恢复差和其他节点Jacobian均为0。它不沿节点索引直接卷积；不得把Social-STGCNN的问题原样套在SGCN上。空间mask的输入[8,4,N,N]确实沿N×N执行3×1/1×3卷积，学出的邻接不满足同步置换，完整网络也出现换序差。旧报告不改写，新增`structural_supplement.json`记录纠正。单位空间/时间图干预下N=3跨人Jacobian为0，只是结构检查，不是已训练NoSocial性能。

## 数据与训练准入问题

`utils.py`要求整20帧完整行人才进入窗口和上下文，人数≥2；它不是Stage5.1只按历史选择邻居的因果协议。位置读入再差分，输入只用历史，但未来存在性筛选上下文仍是协议差异。官方loader以frame索引序列取样，原始0.4秒/米制需要文件级审计，不能仅凭8/12认定相同。

`train.py`有完整从头训练入口，默认300epoch、batch128窗口累积，实际使用Adam(lr=0.01, weight_decay=0.0001)、MultiStepLR([50,100], gamma=0.5)、Gaussian NLL；解析器另有momentum参数但未用于实际优化器、按val_loss选权重；test使用Gaussian采样，不等于K1 latent-prior。**源码边界累积分支风险：else分支没有把当前窗口的graph_loss加入累积loss，却直接除batch_size并更新；边界窗口可能漏监督。** 这是可定位静态问题，未在本阶段运行训练重现；不能据此宣布所有公开训练都无效。尾批缩放、单人squeeze与NumPy/旧依赖均须未来单测。引用：[train.py](https://github.com/shuaishiliu/SGCN/blob/5153702d8ae32a2b0586b093d4e094aee08b0f9b/train.py)、[utils.py](https://github.com/shuaishiliu/SGCN/blob/5153702d8ae32a2b0586b093d4e094aee08b0f9b/utils.py)、[test.py](https://github.com/shuaishiliu/SGCN/blob/5153702d8ae32a2b0586b093d4e094aee08b0f9b/test.py)。

作者README列Python3.6.2/Torch1.2.0/NumPy1.19.0/2080Ti。当前Torch2.5.1+CUDA12.4、RTX3080严格加载及人工前向可行，完整旧训练环境、收敛、吞吐未验证。不能用“前向成功”代替论文复现。由于排列与训练边界问题，本轮不把SGCN升级为即用的第四章基础，不自动修模型。

## Social-STGCNN旧实验证据复核

CVPR2020论文与本地维护版commit `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`分开。Stage4.1固定官方维护权重在64个非ETH窗口/235目标，置换后恢复的平均位置差0.266736m，最大1.917565m，ADE0.521069→0.567688；差异首次出现在TXP3×3节点轴卷积，非ST-GCN。原Full与NoSocial还曾有BN/PTN归一化差异，不能单因素归因。

Stage4.2受控G0/G1改为相同归一化、1553参数、共享逐人decoder与同初始化；最终同预算到100轮，G1−G0 ADE ZARA2 +0.091252、HOTEL −0.046452m，两者均弱于同目标三CV。PTN毫米扰动风险有固定权重证据，未作移除PTN因果训练，不能指定唯一失败原因。来源：`reports/stage4p1/03_PERMUTATION_AUDIT.md`、`reports/stage4p2_final/05_FINAL_RESEARCH_DECISION.md`及各原始CSV。本轮没有重训或篡改旧指标。

## 唯一补充候选：SingularTrajectory（源码为主）

[CVPR2024正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.html)与[作者仓库](https://github.com/InhwanBae/SingularTrajectory/tree/f7855881976276787bc16d951a9259d1cd50c101)对应，固定commit `f7855881976276787bc16d951a9259d1cd50c101`，27个源码/配置文件Git blob及SHA核查通过。它比SGCN更新，提供8/12配置、训练入口、公开release下载脚本，运动/静止两套Singular表示、历史原点/方向/尺度、原型与扩散已是原方法。**这些内容不能作为HTraj重新命名的创新。**

源码是Transformer场景mask交互，不是替换SGCN的简单显式空间GNN。默认样例K20、batch512、256epoch、AdamW、10步去噪，按验证FDE选择；SVD/原型以train的obs与future初始化，未来标签仅训练侧拟合本身允许，但必须审计只用source train。官方完整上下文仍离线筛选。额外依赖场景图、homography与vector_field，`script/generate_vector_field.py`使用地图图像；与HTraj纯历史坐标协议不等价。domain adaptation任务不能冒充未接触目标域的domain generalization。许可为CC BY-NC4.0，不是MIT。

只做了零速/低速/正常速度的归一化组件检查，没有完整权重加载/完整模型前向/数据评价。裸尺度归一化零速会Inf，但实际静止分支关闭尺度，三个人工样本均有限并能恢复；不能据裸子模块Inf声称整网失稳。CUDA硬编码、依赖与场景资产、数据未来筛选、不同任务配置完整性和RTX3080完整训练显存/耗时均未验证；示例配置可见不代表release全部可复现。公开权重仅定位URL，未下载/核验，不报ADE/FDE。

来源：[model.py](https://github.com/InhwanBae/SingularTrajectory/blob/f7855881976276787bc16d951a9259d1cd50c101/SingularTrajectory/model.py)、[trainer.py](https://github.com/InhwanBae/SingularTrajectory/blob/f7855881976276787bc16d951a9259d1cd50c101/utils/trainer.py)、[dataloader.py](https://github.com/InhwanBae/SingularTrajectory/blob/f7855881976276787bc16d951a9259d1cd50c101/utils/dataloader.py)、[config](https://github.com/InhwanBae/SingularTrajectory/blob/f7855881976276787bc16d951a9259d1cd50c101/config/config_example.json)、[LICENSE](https://github.com/InhwanBae/SingularTrajectory/blob/f7855881976276787bc16d951a9259d1cd50c101/LICENSE)。由于地图侧信息、扩散成本和当前协议不匹配，不作为本轮主基础，也不盲目调试。

## 实际状态对照

|对象|显式图|8/12|从头入口|权重|本轮实际执行|公平消融与协议风险|
|---|---|---|---|---|---|---|
|SGCN历史版|空间/时间有向图|是|有；边界累积待验证|2份strict成功|人工前向/排列/Jacobian|空间mask索引敏感、完整20帧上下文、有效容量|
|Social-STGCNN维护版|社会图+节点卷积|是|旧阶段真训练|旧阶段strict/评价|本轮复核冻结记录|旧Full/NoSocial多因子；G0/G1受控失败不可直接当官方|
|SingularTrajectory|Transformer场景关系|示例是|有|release可定位，未验证|仅归一化组件|地图与train-only表示拟合、K1/K20、计算预算|
|冻结SocialVAE-NoSocial|完全无真实邻居|是|Stage5.3真训练|自训练best/last冻结|本轮引用旧结果|仅seed17；有效容量与Full不同，不用来证明社会收益|

检索MCP未挂载；OpenAlex429、GitHub REST403均保留日志，使用CVF/arXiv/CrossRef及作者Git原始文件降级核查。一次本地目录重复创建的工程失败日志保留，修正后独立v2日志成功；没有隐去或填造模型结果。聚焦审查不是穷尽综述，也不提供新颖性认证。
