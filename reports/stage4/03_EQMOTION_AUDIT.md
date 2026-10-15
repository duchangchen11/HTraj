# EqMotion：结构、数据与有限验证审查

## 来源与冻结范围

论文 *EqMotion: Equivariant Multi-Agent Motion Prediction With Invariant Interaction Reasoning*，CVPR2023；[CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)、[作者官方仓库](https://github.com/MediaBrain-SJTU/EqMotion)、[arXiv:2303.10876](https://arxiv.org/abs/2303.10876)。冻结commit `5aec2e0b61c511fa93a24138dd90da59a089084b`。真实源代码、MIT许可证保存在 `third_party/stage4/eqmotion/`；提供ETH/UCY多头模型与官方checkpoint。只获取HOTEL、ZARA1两个权重和对应非ETH测试文本。

主审查路径为 `eth_ucy/model_t.py`、`gcl_t.py`、`preprocessor.py`、`process_eth_data_diverse.py`、`dataloader_diverse.py`、`main_eth_diverse.py`。没有运行其他人类动作/粒子任务，不能把这些任务结果混入行人结论。

## 运动等变与交互路径

输入 `X:[B,N,8,2]`、相邻位移 `V`、速度模长pattern特征 `[B,N,8]` 和真实节点数mask。预测 `[B,N,20,12,2]`，有20个学习到的输出头；评估仅针对target-centered输入的第0个目标。不是20次高斯随机采样。

代码用速度模长和相邻速度夹角建立不变pattern表示，用有效行人的共享中心化、DCT与沿时间/通道的线性变换建立几何表示。简化地记 $H_i=\phi(\|v_i\|,\mathrm{angle}(v_{i,t},v_{i,t-1}))$、$G_i=\mathrm{DCT}(X_i-\bar X)$。这里的 $v$ 是每帧位移，非另外输入米/秒；固定0.4秒下只差共同尺度。

稠密候选对上由pattern与几何距离推断四类关系概率 $q_{ij}=\mathrm{softmax}(\psi(H_i,H_j,\|G_i-G_j\|)/\tau)$。几何更新可概括为

$$G'_i=G_i+\sum_{j\ne i}\sum_c q_{ij,c}\,a_c(H_i,H_j,\|G_i-G_j\|)\,(G_i-G_j),$$

同时更新pattern、执行行人内部几何注意力和非线性几何变换。这是结构示意，不替代逐层源码公式。核心是不变标量权重作用于相对向量，旋转/反射改变几何而不改变关系类别。时间/通道线性层不混合二维坐标轴，最后恢复共享中心和全局位置。完整输出多头通过各head的图层及时间映射产生。源码的掩码与valid prefix规则必须保留。

具有明确空间图结构，但不是“所有场景行人自动全连接”：先按目标中心预筛选候选，再对保留节点作两两传播。典型图成本 $O(BLN^2Cd)$，20头另有图/解码成本；公开预处理限制N很小，本轮不能据此推断密集场景大N成本。

## 数据与关键陷阱

1. 8历史/12未来，坐标文本为16列，行人位置在第13/15索引列；预处理要求完整历史和未来，使用目标最后观测位置5米内最近邻。官方 `eth_ucy/data/{fold}` 包含预切train/val及测试整录像，名称清单同样含zara03、uni_examples和ETH训练片段，UNIV测试含两个students录像；本轮只解析HOTEL/ZARA1测试文本，未复现原训练池。
2. 官方 `process_eth_data_diverse.py` 实际覆盖命令行total_num：UNIV为2，ZARA1为3，其他fold为1。因此 **HOTEL和ZARA2的正式脚本每样本只有ego**。ZARA1最多两名邻居，不能将HOTEL成绩归因社会交互。
3. `dataloader_diverse.py` 使用机器专属GPFS路径及预处理npy；本地包装器显式传入冻结原始测试文本路径，按原预处理规则生成目标中心样本；没有自行增加邻居或扩训练池。
4. `main_eth_diverse.py` 即使指定 `--test` 也先创建训练数据，评价后还落入训练循环；训练入口每轮按test FDE保存最佳权重。我们不运行此入口、不复制其test选模型方式。仅从AST提取未改写的 `test()`，使用官方已提供权重、严格state_dict加载。
5. 非ETHtest_scale为1；ETH特殊缩放仅查代码，未运行ETH，也未读取ETH成绩。不能据此假定全场景同一尺度策略。

## 环境兼容与功能验证边界

原推荐Python3.7、PyTorch1.8；本轮使用独立Stage4环境。最小兼容处理是包装器 `np.int=int`（恢复被移除别名）、显式本地test-only数据根、隔离评价入口，未改网络/损失。上游strict=False被外部验证改为strict=True，用以识别键缺失，而非静默接受不匹配权重。

训练前预注册范围为HOTEL和ZARA1各按时间顺序最多128个目标样本、batch16、固定官方通道64/128和4层，不看结果再换样本。计划报告K1固定head0和K20逐目标独立最小值。**实际在strict权重检查时停止，评价目标数为0，ADE/FDE未产生**，不能将计划样本数写成已评价数。

诊断结果：HOTEL没有缺失键，但有399个多余键；ZARA1有184个缺失键和399个多余键，二者共有键没有形状差异。多余键包括embedding_gru和多层att_q/att_k等；ZARA1缺失的主要是各gcl/head中category_mlp第2/3类参数。官方strict=False可能悄悄丢掉前者并让后者保留随机初始化。当前证据不足以证明它们只是可安全忽略的兼容性差异，因此不放宽strict、不改类别数、不随机补参数，也不反复寻找版本。

为区分“源码无法运行”和“权重不匹配”，另用固定seed17的**随机权重、合成历史、CPU**进行了源代码结构测试。平移/旋转/反射/真实节点重排均有限，最大预测一致性误差分别约9.35e-6/3.08e-5/0/8.35e-7米，关系类别最大差约2.98e-8；均低于1e-3容差。没有使用真实未来或checkpoint，绝不是预测性能结果。原来计划在真实checkpoint首batch做的几何检查未能执行，合成测试属于明确标记的有限诊断补充。见 `eqmotion_checkpoint_compatibility.json`、`eqmotion_execution.json`、`eqmotion_synthetic_geometry_checks.csv`，原失败日志保留。

本实现实例化参数量：HOTEL配置3,027,268，ZARA1配置5,536,260；与论文/其他任务中模型大小描述不能混用。权重strict加载失败；只有源码合成forward可执行，不能称官方预训练推理成功。RTX3080内存足以放下这两个实例，但真实非ETH性能推理、大N、多种子、从头训练收敛均未验证。

## 是否适合作为后续基础

优点：运动几何性质有明确结构保证；关系推断与几何更新存在可定位模块；适合研究运动表示与交互协同，能避免把普通归一化包装成创新。

成本与反证：多头参数量大于当前轻量Social-STGCNN；数据处理有隐藏N预算、GPFS路径和训练入口问题；`given_category`只替换类别来源，**不能独立关闭所有社会影响**，因为pattern聚合、几何更新和全局中心化仍可依赖邻居。若做无交互对照需完整界定隔离路径并从头重训练，单关一个flag不能声称因果归因。单节点测试不证明交互有效。

判断：EqMotion是有价值的第二候选，不能仅凭等变理论或有限128样本成绩宣布其更优。先冻结统一候选上下文/评价样本与无交互对照，再评估是否升级为第四章主干。本轮没有修改结构，也没有启动大量训练。MIT代码许可不自动解决数据再分发授权；预训练上游可能见过ETH，未来最终ETH封存评价必须重新训练排除ETH的模型。
