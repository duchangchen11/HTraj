# 冻结ego与信息来源对照设计

代码src/models/frozen_ego.py；协议01_EXPERIMENT_PROTOCOL.md与configs/stage3p6/protocol.json。在旧PedBase/Encoder/Decoder基础上新增独立模块，无旧模型代码修改。

## 个体预测

输入X∈R^{B×8×N×2}、历史mask、timestamps。既有7维速度/加速度/可见性/速度缺失等特征经原EgoEncoder得到H∈R^{B×8×N×64}；每人共享GRU，缺失/时间gap reset。原头64→64→24输出[B,12,N,2]，共35352参数。

\[
\widehat v_i=\frac{x_{i,7}-x_{i,6}}{0.4},\quad
Y^{CV}_{i,k}=x_{i,7}+0.4k\widehat v_i,\quad
Y^{CVR}_{i,k}=Y^{CV}_{i,k}+D_\theta(h_{i,7})_k.
\]

速度未知显式标记并采用停留回退；完整主任务目标本身有连续8点。CVR与PedBase同一encoder/head/默认初始化，仅加入历史可计算CV主路径，无参数增加。这是运动基线对照，不宣称新颖性。

## 冻结与统一残差

anchor选择、checkpoint文件SHA、预测是CV还是神经输出，在社会训练开始前锁定。requires_grad=False/no_grad编码、anchor始终eval、只将新增参数给优化器；每epoch与结束核对anchor数值参数SHA。模型新checkpoint虽含anchor副本，其anchor张量仍与源checkpoint一致。所有四个分支同14425新增参数，无GRU、注意力、学习型边打分或新loss。

\[
q_{ij,t}=\phi([h_{j,t-\ell},r_{ij,t},w_{ij,t},\tau_{ij,t},d^{CPA}_{ij,t},1]),\qquad
m_{i,t}=\frac1{|\mathcal N_{i,t}|}\sum_{j\in\mathcal N_{i,t}}q_{ij,t},
\]
\[
u_i=\frac1{|\mathcal T_i|}\sum_{t\in\mathcal T_i}m_{i,t},\quad
\Delta Y_i=f([h_{i,7},u_i]),\quad
Y_i=\begin{cases}Y_i^{ego}+\sigma(a)\Delta Y_i,&|\mathcal T_i|>0,\\Y_i^{ego},&\text{否则}.\end{cases}
\]

ℓ=1是Previous/Random，ℓ=0是Current。H的t节点只依≤t历史。边t用两端t/t−1可见运动，均在原4m/k8/CPA规则内。T_i为有消息帧，目标中断丢弃此前累计项；8点完整目标无中断。h_j,0仍含速度未知flag，Previous在t1的payload没有速度观测，Current有，这正是要验证的信息时效差别。均匀时间池化只汇总一阶邻居，不传播邻居的社会状态。

Ego-Residual把q改为φ([h_i,t−1,自身7维运动特征])，不计算用于预测的邻居消息；u为连续自身帧均值。它在无邻居时仍可修正，代表泛运动容量，不伪装成社会分支。相同payload权重维度和残差头，但来源/支持不同，这些差别在报告中保留。

φ为71→64/ReLU，f为128→64/ReLU→24，a为一个共享标量。全分支f最后层零初始化，初始精确ego；a=−2。模型共享每人/每边/每时间的参数，输出不使用未来GT。广播gate仅乘has_message布尔支持，torch.where保证无消息预测逐位等于同一anchor，而非另一个共同训练ego。门控是全局scalar，不声称已经具备场景自适应能力。

张量：关系几何[B,8,N,N,7]按帧生成；候选mask[B,N,N]；payload[|E_t|,71]；消息[B,N,64]；累计状态[B,N,64]；修正[B,12,N,2]。构图O(BTN²)，稀疏payload/汇聚O(Σ|E_t|×71×64)，头O(BN×128×64)，无高阶路径增长。冻结编码缓存O(BTN64+BN12×2)，本阶段只在内存新生成。

## 关系随机化与归因边界

Random独立训练新增分支，在4m可行候选中以坐标/时间hash选择，保持入度预算（罕见精确hash平局整个弃去，记录偏差）、连续性、场景与self排除。只换关系，几何随关系更新，距离分布也会变；记录换边率/距离P05/median/P95，若几乎无法换边，不解释为强负对照。原始轨迹/运动编码从不打乱。

自动测试17项新增、全165项训练前通过：同参同初始化CVR、缺失速度、三个社会模型无消息精确ego、四变体优化冻结不变、同参数/初始输出、Previous/Current人工payload、前缀因果、随机图合法/预算、padding/置换/缓存、future GT分组不变。每run配置还保存源码/data/protocol SHA。

借鉴已有CV、GRU、残差、均值聚合、冻结迁移思想；此阶段贡献是可审计的归因实验结构，不是独立论文创新。与已有复杂GNN比较存在结构/参数/训练差异，不将新结果直接归因为“联合训练必然伤害ego”，仍需以后单独授权的同结构反事实。
