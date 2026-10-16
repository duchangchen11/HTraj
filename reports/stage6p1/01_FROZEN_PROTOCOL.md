# Stage6.1冻结训练协议

仅检验一个来源训练采样因素，无新模型、无新loss/decoder/归一化/GNN/社会注意力/门控/运动残差。基础提交af7da3ffda0d65ec8ff1e05a5d0235d62cd53dad，新分支feat/stage6p1-motion-sampling-pilot。冻结源码/配置先本地提交再训练；精确训练代码commit由各run/provenance.json记录，避免循环自SHA。

Stage5.3 SocialVAE-NoSocial实现原样复用，官方源码commit eb23cbae562bac2c2dcf0d81529e6ab2ac280962不改动。名义参数2144002，结构有效上界1459906；两条件完全相同架构、权重初始化及初始化后Torch RNG。保留原GRU、潜变量、decoder和重建+MC KL目标；不能把该loss称为精确边缘NLL。全部run从随机权重开始，不用任何历史checkpoint初始化。无邻居输入路径的既有隔离测试继续适用。

数据HTraj-SocialVAE-Causal8x12-v1：8帧历史、12帧未来、0.4秒、米制。HOTEL与ZARA2分别LORO；训练只使用原manifest的来源train区域，既不包含留出录像，也不提升旧validation。监督目标资格沿用冻结8/12完整性，类别计算只接收8帧历史位置。ETH封存；两个录像均已曝光且用于选择，不是独立确认或严格未见目标域结论。

| 留出录像 | split | 目标窗口/行人ID | 录像来源 | manifest SHA |
| --- | --- | --- | --- | --- |
| biwi_hotel | train | 27145/892 | crowds_zara01,crowds_zara02,students001,students003 | e3f444dd718d0c358b817d5350ebd29e8b91c302beb153929c0302a0ebeae9a1 |
| biwi_hotel | validation | 318/28 | biwi_hotel | e3f444dd718d0c358b817d5350ebd29e8b91c302beb153929c0302a0ebeae9a1 |
| crowds_zara02 | train | 23545/833 | biwi_hotel,crowds_zara01,students001,students003 | 9ed3d7975b6288ec254de7152edbb166aa9d53f4fcbac9e51de43ada0bf13246 |
| crowds_zara02 | validation | 1092/35 | crowds_zara02 | 9ed3d7975b6288ec254de7152edbb166aa9d53f4fcbac9e51de43ada0bf13246 |


四run：HOTEL Uniform、HOTEL Stratified、ZARA2 Uniform、ZARA2 Stratified；seed17，Adam lr0.0005、wd0、batch128、无scheduler/clip/augmentation。各30轮，前5轮是本run内smoke，绝非额外5轮。每轮N次访问、ceil(N/128)次优化，保留尾批。最多四个独立进程并发，单RTX3080资源共享，实际wall时间包含争用，不把并发耗时当独占吞吐。

Uniform=default_rng(17+epoch).permutation(N)，与原Stage5.3规则相同；Stratified用同一独立numpy种子按P有放回抽N个。只改变采样；重复曝光、独特覆盖与有效训练分布确实不同，不能称完全相同训练分布/经验风险，亦不等价于独立证明某个统计量造成速度偏差。参数量/访问/优化步/初始化匹配不排除优化噪声差异。

选择最低窗口加权K1 latent-prior ADE，精确相同取早epoch，同checkpoint FDE。固定保留1/5/10/20/30轮immutable checkpoint和独立best，所有epoch原始曲线均保留，last=30，best/last只作索引。辅助K1首样本、K20分别最小ADE/FDE；固定Stage5.2 target-ID噪声种子与预算，FPC关闭。latent-prior轨迹不是预测分布数学期望。评价用验证GT；GT不能进入采样、输入或图。

正式继续门槛全部同时通过：两录像×窗口/行人权重ADE增益≥max(0.005m,1%Uniform ADE)；同两权重FDE退化≤max(0.005m,1%Uniform FDE)；HOTEL正常行走窗口权重绝对组均值速度偏差下降≥10%；两录像×两权重历史静止组FDE退化≤max(0.01m,5%)；所有数据/预算/ID/GT/配置核验。主速度门槛为|mean(v_pred−v_GT)|，同时报告mean(|每目标速度偏差|)及逐步速度误差，避免均值抵消误读。历史分类阈值0.05/0.4/1.8m/s原样保留。未来持续运动标签仅为事后诊断。

任何门槛不通过则停止该采样假设，不搜索比例、阈值、结构、seed或训练长度；best在30标预算截断，不续训。任何NaN/Inf停止该run并保留失败现场，不重试调参。其余事先冻结run仍可完成。旧100轮NoSocial不替代新30轮Uniform。即便通过也只建议另阶段验证，普通重采样不是论文创新。

协议SHA：`27762097318e8414d015659ae952480a79a2697eb18581e7aed5bddb12998cf1`。完整公式、随机数规则、源数据/manifest SHA和停止门槛在configs/stage6p1/protocol.json；旧文件保护清单独立记录。只本地commit，不push；四run完成后停止。
