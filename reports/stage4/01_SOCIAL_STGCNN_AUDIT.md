# Social-STGCNN：来源、网络和运行协议核查

## 身份与版本

正式名称为 *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction*，CVPR 2020。会议页面、作者官方仓库对应一致：[CVF论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.html)、[官方仓库](https://github.com/abduallahmohamed/Social-STGCNN)。本次冻结官方HEAD `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`，MIT许可证随源码保存到 `third_party/stage4/social_stgcnn/`。权重和数据留在被忽略的 `data/external/stage4/`，未提交。

**官方HEAD不能等同CVPR2020原始代码版本。** README指定论文版本为 `ebd57aaf34d84763825d05cf9d4eff738d8c96bb`。对此前保存的原版源码逐行比较：`utils.py`、`test.py`相同，但 `ConvTemporalGraphical` 有实质改变。原版卷积输出 `out_channels * kernel_size`，重排成K组后进行 `nkctv,kvw->nctw`；当前HEAD卷积只输出 `out_channels`，执行 `nctv,tvw->nctw`。本次使用当前HEAD及其同版本提供的checkpoint，未替换为旧版网络。结论只能写“冻结官方维护版本的checkpoint复跑”，不能写“精确复现CVPR2020原版架构及论文全表”。版本差异记录见 `results/tables/stage4/source_version_comparison.json`。

## 实际计算结构

来源：[冻结model.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/model.py)、[utils.py](https://github.com/abduallahmohamed/Social-STGCNN/blob/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/utils.py)。输入是8帧二维相邻位移和8个图矩阵，网络输入形状 `[1,2,8,N]`、`[8,N,N]`；ST-GCN作图传播与时间卷积，TXP-CNN将历史时间维映射到12步。输出5维高斯参数，转为 `[12,N,5]`。保留上游原有 `.view` 操作，没有自行改成另一种维度置换。

令历史位移为 $d_{i,t}=x_{i,t}-x_{i,t-1}$，第一步位移置零。代码构图为

$$W_{ii,t}=1,\quad W_{ij,t}=\begin{cases}\|d_{i,t}-d_{j,t}\|_2^{-1},&\|d_{i,t}-d_{j,t}\|_2>0,\\0,&\text{否则},\end{cases}$$

$$L_t=I-D_t^{-1/2}W_tD_t^{-1/2}.$$

特别注意：这里比较的是**位移向量差**，不是行人的物理空间距离；相同位移的两人边权为0，而位移差很小但不为0时原始权重可以很大。是否造成选择不足、是否值得改进，需同协议重训练消融，本阶段未证明。当前图乘法是 $Z_{c,t,j}=\sum_i \mathrm{Conv}(V)_{c,t,i}L_{t,ij}$。图计算为稠密 $O(TN^2)$；后续卷积沿节点索引维使用核宽3，不能默认具备行人置换等变性。输入使用位移，预测累加回最后观测位置；没有显式方向规范化或旋转等变约束。

模型预测 $(\mu_x,\mu_y,\log\sigma_x,\log\sigma_y,\rho_{raw})$，其中 $\sigma=\exp(\log\sigma)$、$\rho=\tanh(\rho_{raw})$。训练源码采用二维高斯负对数似然；本轮没有训练，也未改损失。每步采样二维位移后累加恢复米制位置。多模态来自高斯采样，不是20个固定意图类别。

## 官方数据筛选与划分

8帧观察、12帧未来，名义采样0.4秒。官方仓库预切分 `datasets/{fold}/{train,val,test}`，测试fold为ETH、HOTEL、UNIV、ZARA1、ZARA2；UNIV测试合并students001和students003。官方非ETHfold的train/val名称清单还包含 `crowds_zara03_{train,val}`、`uni_examples_{train,val}` 以及 `biwi_eth_{train,val}`；按fold排除对应测试录像，UNIV同时排除两个students录像。名称清单见 `social_stgcnn_tree_inventory.json`。本轮没有读这些训练/验证轨迹，不推测其精确切分比例或时间边界。官方训练池因此不能冒充仅五个非ETH录像、保留旧val区域的HTraj LORO manifest。

loader按文件内已出现的排序帧滑窗，要求节点完整覆盖20步，并且每窗口完整节点数 **大于1**；目标和邻居都经过未来存在性筛选，坐标保留四位小数。模型forward仅接收历史图/位移，没有未来坐标输入；但这是利用未来存在性选择上下文的离线benchmark口径，与HTraj仅按历史选择上下文不同。代码没有显式验证每对全局帧号差都等于10；本轮接受窗口的实际帧间隔核查另存表格，不擅自改官方loader。

官方checkpoint有 `args.pkl` 和 `val_best.pth`。本次仅下载并严格加载HOTEL/UNIV/ZARA1/ZARA2的参数，未读上游 `metrics.pkl`、`constant_metrics.pkl`。原test全局入口会glob所有fold并读取缓存成绩，故未执行；本地仅提取原 `test()` 函数，使用非ETH白名单驱动。聚合分母为4，不沿用全局入口的5。

## 指标和可信度边界

对目标i、样本k：$ADE_{ik}=12^{-1}\sum_{t=1}^{12}\|\hat y_{ikt}-y_{it}\|_2$，$FDE_{ik}=\|\hat y_{ik,12}-y_{i,12}\|_2$。官方K20分别计算 $\min_k ADE_{ik}$ 和 $\min_k FDE_{ik}$，两个最优样本可能不同；也不要求同一场景所有人的最优样本编号相同。它不是K1，也不是可在预测时知道GT的可实现选轨策略。

本次同时保存K1首个采样、K1分布均值、K20独立最小值，三种子仅代表采样波动。没有用均值轨迹或HTraj结果代替官方K20。固定权重、完整四个非ETH官方测试分区上的实跑结果见02报告。

## 环境、硬件与后续研究

原环境约Python3.6/PyTorch1.1.0/NumPy1.16.4/NetworkX2.3/SciPy1.3.0；原requirements包含大量无关系统包，不宜原样安装。独立环境 `experiments/stage4/venv` 使用Python3.10.21、torch2.5.1+cu124、NumPy1.26.4、NetworkX3.4.2、SciPy1.14.1。唯一计算API兼容替代为 `nx.from_numpy_matrix=nx.from_numpy_array`，归一化Laplacian数值已测试；`no_grad`、单进程loader只改变执行方式。上游源码字节未改，具体版本、SHA256和耗时在结果目录。

RTX3080实际可执行本轮推理。模型小，适合作为图结构研究的工程起点；但“动态图、自适应稀疏边、社会门控”本身已有大量工作，不能当作新创新。尤其应先明确当前HEAD与论文版本差异、节点顺序敏感性、历史上下文口径及同容量无交互对照，再决定正式方法。

两个仓库源码均MIT，不代表自动获得ETH/UCY原始数据、视频或其他第三方内容再分发许可；未建立单独数据授权证据，数据与权重均只在本地使用。后续发布数据需另查原始数据条款。本次不读取ETH评价轨迹/成绩。官方非ETH预训练权重可能在上游训练时见过ETH，**不具备未来ETH外部测试的训练隔离保证**；正式最终实验必须重新建立排除ETH训练的流程。
