# Stage5.5：训练前冻结协议

基础e8c69194f228a19d4e7410a9825adbeedb2b2b42，分支feat/stage5p5-social-stability。只验证真实历史邻居相对冻结Placebo的跨初始化开发收益，不设计模型、修改注意力/loss/图结构或使用ETH。

## 数据与模型

HTraj-SocialVAE-Causal8x12-v1；8历史/12未来、0.4秒、米制。严格复用Stage5.1两个B manifest：HOTEL e3f444dd718d0c358b817d5350ebd29e8b91c302beb153929c0302a0ebeae9a1；ZARA2 9ed3d7975b6288ec254de7152edbb166aa9d53f4fcbac9e51de43ada0bf13246。训练排除整个验证录像，其余非ETH录像只用冻结train区域，不提升旧验证区域。两个开发录像已曝光，不是独立确认。

Full直接使用冻结官方SocialVAE eb23cbae562bac2c2dcf0d81529e6ab2ac280962；Placebo直接导入Stage5.4冻结类与synthetic_context，绝不重写规则：4邻居、±0.8m轴向偏移、符号[+1,−1,+1,−1]、0.2m×tanh目标历史净位移的有界漂移，8帧因果支持，未来12槽全padding。原结构、VAE、decoder、重建+MC KL目标全部不变；后者不称精确边缘NLL。

名义参数两者均2144002，结构有效上界2045698；相同参数数量不代表输入分布或函数类完全一致。Placebo固定近邻、完整历史和恒定初始偏移的混杂继续保留。Stage5.4完整输入分布统计原样复制并核对SHA，不按seed改变上下文规则。

## 8个run与固定预算

新增seed42/2026；各自Full/Placebo × HOTEL/ZARA2，共8个正式run，每个100完整epoch，遍历全部训练目标且保留尾batch。Adam lr0.0005、batch128；无scheduler、clip、augmentation、weight decay、额外loss或预训练初始化。每个run重新seed_all(seed)并独立初始化。相同seed两类模型的全部公共权重与构造后Torch RNG必须完全相同，且run启动时再次比对初始SHA。全部8组预算事先固定，不依据seed42调节seed2026。

4个旧seed17正式Full/Placebo作为冻结参考，形成12组合索引。NoSocial/EgoCapacity仅作为seed17历史参考，绝不纳入3seed的均值/SD。三种解析CV同一目标不随训练seed改变。

固定队列每类2个worker（最多2 Full+2 Placebo），每类顺序42-ZARA2、42-HOTEL、2026-ZARA2、2026-HOTEL；同GPU进程独立RNG，墙钟实耗包含资源竞争，不冒充独占吞吐。排队不依赖模型成绩。

磁盘约5.1GiB，不删除旧权重。保存epoch1/25/50/75/100和best；best每次改善时将模型/Adam/CPU与CUDA RNG序列化到RAM，完整训练后一次写出所选epoch的不可覆盖文件。每轮全部指标、逐目标误差、参数SHA和梯度记录均持久化；周期checkpoint严格恢复检查。非有限值停止该run并保存失败现场，不做超参重试，其余冻结run仍执行。

## 评价与采样

每轮全部验证目标，固定顺序与batch128。最低窗口加权K1 latent-prior ADE选择best，完全相同取更早epoch；FDE、K1首样本与K20均同一checkpoint，不拼接最好指标。K1 latent-prior不是预测分布数学期望；minADE@20/minFDE@20分别独立最优，首样本为同20候选的第0个；FPC关闭。

新评价实现只将身份键seed17扩展为实际训练seed：

    namespace | protocol | seed<run_seed> | recording | start_frame | target_id

prior namespace保持stage5p2-prior-noise-v1，posterior保持stage5p2-posterior-noise-v1；SHA256前8字节取模2^63，目标本地CPU Gaussian生成器，20×12×32 prior及12×32 posterior噪声。相同seed/目标的Full与Placebo噪声完全一致；新评价在seed17上必须与旧实现逐值相同。保留原CUDA跨batch分割毫米级舍入限制，不更换batch追求低误差。完整噪声seed与SHA见evaluation_sampling_seeds.csv.gz。

## 预注册统计与判断

主差值Delta=Full−Placebo，负值代表Full好。逐seed/录像分别报告窗口加权和录像内行人ID等权；另按录像1/2权重给出宏平均。三个seed均值、样本SD(ddof1)、方向一致性、各模型自然波动和逐目标差异全部保留。n=3不足作强统计结论，不按某个SNR阈值筛选结果。

固定seed/录像按行人ID整簇bootstrap2000次；每次有放回抽取原数量ID，保留所选ID的全部重叠窗口及重复次数。窗口加权为抽样簇误差和/窗口数之比；行人等权为抽样ID内均值的平均。相同抽样矩阵同时用于ADE/FDE、两权重与各指标族。bootstrap基种子20261016，派生SHA规则见configs/stage5p5/protocol.json；报告原始draw、分布和percentile 95%区间。

HOTEL仅28 ID、ZARA2仅35 ID；人之间仍可能相关，跨seed不是新数据集，bootstrap不能消除开发曝光偏差。分组仅沿用冻结历史特征，全部保留；ID<10或目标窗口<30标注少样本，空组null，不作选择性分组显著性搜索。

预设：3seed×2录像×2权重的K1 ADE差值全负，且FDE全负，才记录更强的跨初始化联合开发证据；ADE有方向反转即判当前收益不稳定，停止据此开发复杂社会模块。ADE全负但FDE冲突，明确为ADE单项证据，不能宣称联合改善。不把bootstrap区间变成临时训练/停止门槛。即使全部通过，也未排除Placebo分布混杂或获得独立录像确认；HOTEL弱于强CV必须同时报告。

best≤5标记早期最佳，best≥91标记靠近预算边界；比较81–90与91–100段及训练/验证差距，不自动延长。8组完成后停止，不加seed、不重训其他控制、不评价ETH。训练前完整测试与本地commit，最终再次验收历史SHA并只做本地commit，用户自行push。
