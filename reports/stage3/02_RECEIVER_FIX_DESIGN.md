# Receiver-Fix：分离历史carry与新消息接收

训练前结构冻结：c8d151c8c53367ef361d23d30569c9ea5ff8cf52。实现[src/models/relation_routing.py](../../src/models/relation_routing.py)。原Stage2.6全部模型和结果保留，保护清单440项、78个本地checkpoint。

## 接收契约与公式

令可见性为m_i,t，连续性c_i,t=m_i,t m_i,t-1 I(Δt≤0.6s)。已知速度k_i,t=c_i,t；未知速度是一种缺失观测，不能解释为静止。PedBase编码器已有motion_valid/acceleration_valid标记并在轨迹中断时reset，完全复用。发送者需c_j,t=1，当前速度由两点差分真实计算；投影payload保持旧h_j,t-1。其前一帧编码可能只含一个点，此时其自身缺失速度标记保持缺失，不补造速度，可能限制可用信息，这是保留原消息时刻的明确边界。

旧候选C_old=c_i,t c_j,t I(i≠j) I(d_ij≤4m)。新候选：

$$C_{ij,t}=m_{i,t}c_{j,t}\mathbf1(i\ne j)\mathbf1(\|x_{i,t}-x_{j,t}\|\le4).$$

速度双方已知时原CPA、4.8s截断和closest+0.05distance排序不变，max_neighbors=8，临界同分组整体保留/丢弃以保持置换等变。target速度未知时不能算相对速度或CPA；用当前距离排序和closest，tau/w槽位是缺失占位，显式关系标记u=2。已知关系u=1、非候选u=0：

$$f_{ij,t}=[r_{ij,t};\ w_{ij,t}^{\rm masked};\ \tau_{ij,t}^{\rm masked};\ d_{ij,t}^{\rm safe};\ u_{ij,t}]\in\mathbb R^7.$$

源速度仍须已知，不允许一帧源发送。相对速度缺失掩码和u一起使用，不将占位0作为真实零速度。复用原135→64→1关系MLP，以原[h_i,t,h_j,t-1,f]计算alpha并按receiver softmax。此显式缺失语义仅作用于原先不存在的新接收边；不是添加新的网络模块/参数，也不是只翻转Boolean后默认为未知速度=0。

$$M_{i,t}=\sum_{j\in\mathcal N_{i,t}}\alpha_{ij,t}W h_{j,t-1},\quad
P_{i,t}=c_{i,t}S_{i,t-1},$$
$$a_{i,t}=m_{i,t}\big[(a_{i,t-1}\land c_{i,t})\lor(|\mathcal N_{i,t}|>0)\big],$$
$$\gamma_{i,t}=\operatorname{sigmoid}(G[h_{i,t};\overline f_{i,t}]),\quad
S_{i,t}=a_{i,t}(\gamma_{i,t}P_{i,t}+M_{i,t}).$$

继承P需要自身连续；收到M只需要自身当前可见。新进入者P=0，M可非0；消失时state/support归零。单阶beta=1（有support）/0（无support），保留原R残差：

$$\Delta_i=D(S_{i,7}),\quad g_i=\operatorname{sigmoid}(w_g[h_{i,7};S_{i,7}]+b_g),\quad
z_i=h_{i,7}+a_{i,7}g_i\Delta_i.$$

D为两层无偏置Linear64/Tanh，初始化g=0.05；无消息历史support时z精确等于ego，原12步decoder与loss不变。

## 单因素范围、形状与成本

概念单因素是接收资格政策及其必需的缺失语义。没有改sender连续规则、消息时刻/投影、motion encoder、social carry gate、残差、decoder或训练数据。测试在所有目标连续情况下，与H1同初始化的输出/梯度逐位相同；对部分历史样本新消息可能继承至后续时刻、训练参数可能随新梯度变化，这是预期效应，不能宣称完整历史预测必然不变。

观察[B,8,N,2]与mask[B,8,N]；ego[B,8,N,64]；每帧稀疏边E_t；单阶state[B,8,N,1,64]；输出[B,12,N,2]。所有行人/场景/时刻共享参数，无身份专属学习参数。名义参数73627；单阶fusion.scores的8321参数对softmax无效且梯度为零，有效结构参数65306。旧H1参数完全相同。

当前实现稠密几何/排序O(BT N² log N)，投影/carry约O(BTN d²)，打分约O(ΣE_t d(2d+7))，聚合O(ΣE_t d)。未引入高阶传播或模型B。

## 实测结构验证与归因计划

完整140项测试在训练前通过，JUnit为results/tables/stage3/pretraining_tests.xml。新增22项覆盖新target接收、未历史carry reset、源一帧禁发、缺失标记、安全距离、无邻居精确ego、reappearance/gap、相同已知历史数值与梯度、padding/置换、前缀依赖、GT不在forward接口、CV回退及关系控制合法性。

独立从头训练seed17/30epoch，与冻结PedBase/Dynamic-RN和既有独立H1作诊断对照；主指标仍全部1202个ADE目标/839个末帧FDE目标。必须完整报告完整8帧、1帧、2–4帧、5–7帧、有/无邻居新进入者，以及完整历史密度各组。相同模型用训练好的checkpoint关闭消息仅辅助说明计算依赖；不能代替从头训练的无社会基线。

这是原实现接收契约修复与研究控制，不独立宣称新理论。性能结论见[实验结果](04_EXPERIMENT_RESULTS.md)和[归因分析](05_ABLATION_ANALYSIS.md)。
