# Stage 5.4：容量控制设计与训练前冻结协议

本阶段只验证归因问题，不提出新方法。基础提交为04d95937282e8a84c1f497c6c1170f81a5a753ea；Full与NoSocial采用Stage5.2/5.3冻结结果，不重新训练。官方源码保持eb23cbae562bac2c2dcf0d81529e6ab2ac280962。

## 结构与容量

沿用严格NoSocial目标历史编码，令最终历史状态为h∈R^(B×256)。新增目标自身残差MLP：

    u = h + W2 ReLU6(W1 h + b1) + b2
    W1: 1024×256，b1:1024；W2:256×1024，b2:256

u送入原rnn_fy_init，其后原潜变量先验、监督后验、decoder、loss均保持不变；后验未来编码b保持原NoSocial计算。新增525568个参数，采用PyTorch Linear默认随机初始化。没有任何实际参数掩码；审计中的布尔mask仅描述恒零输入导致的结构不可用参数。

名义参数2669570；loss图连接的模块参数上界2182082；排除从不使用的社会模块和恒零GRU输入列后，结构有效参数上界1985474，相对Full的2045698低2.9439%，满足±5%。ReLU实际激活和长期梯度仍须实测，不能把这些上界当成实际有效容量。计数接近不代表网络函数类相同；新增ego残差路径改变了归纳偏置。其结果只能检验“目标自身容量是否足以达到Full”，不能实现完美容量隔离。常规残差MLP不是论文创新。

张量路径：x[8,B,6]→原embed_s与历史GRU→h[B,256]→ego MLP[B,256]→原rnn_fy_init→原12步VAE生成。MLP时间O(B×256×1024)，每个目标仅执行一次；不引入额外社会消息。neighbor参数为了兼容调用保留，但enc不读取其值、数量、mask或ID。

## 冻结数据、训练和评价

权威配置：configs/stage5p4/ego_protocol.json。HTraj-SocialVAE-Causal8x12-v1，8/12、0.4秒、米制，FPC关闭。ZARA2与HOTEL按Stage5.1 B manifest分别录像隔离；其余录像只使用冻结train区域；ETH封存。两录像历史已曝光，不能称独立确认集。

两个run均重新seed17随机初始化，Adam lr0.0005、batch128、无scheduler/clip/weight decay，100完整epoch，保留尾batch。原始重建+Monte Carlo KL目标不称精确边缘NLL。最小验证K1 latent-prior ADE选best，完全相同选更早epoch，FDE取同checkpoint。latent-prior轨迹不是预测分布数学期望。每轮原始loss、全部评价、逐目标误差、参数状态和梯度审计保存；完整训练池K1在固定epoch及best评价。K1首样本是固定K20候选0；minADE/minFDE分别取各自最优候选。沿用Stage5.2稳定目标ID噪声命名空间、固定batch及顺序。原CUDA跨batch划分毫米级数值限制不掩盖。

原基础模块先分配，随后分配MLP。新增随机分配会消耗RNG，因此不宣称不同架构后续训练潜变量抽样流完全相同；每个run自身复现和评价公共噪声规则保持一致。

每轮指标全部保存；受当前磁盘预算约束，checkpoint只在best改善、epoch1、每10轮和100轮保存，均为不可覆盖新文件。保留best与last及所有best改善快照；不删除旧权重。checkpoint恢复检查在1/每10/best进行。

## 预先冻结的继续/停止门槛

分别按窗口加权与行人ID等权计算；两个录像均须满足。主判断仅用各模型按同一规则选择的best K1及配对FDE：

1. Ego ADE≤Full ADE，且Ego FDE≤Full FDE×1.01：额外ego容量已足以解释当前优势，停止，不执行Placebo。
2. 仅当Full相对Ego ADE改善≥1%，且Full FDE≤Ego FDE×1.01，四个录像×权重单元全部通过，容量/隔离测试通过，才允许第二阶段。
3. 其他情况均停止，报告条件性、冲突或差异过小；不挑seed，不延长，不更换结构。

1%为预注册开发决策阈值，不是统计显著性或正式非劣界。重叠目标窗口不视为独立样本。只有通过第二条才允许另外两组Placebo训练，且必须先单独冻结目标历史合成上下文规则并提交协议；最多总共4组正式run。本阶段无额外seed或100轮以外训练。

来源：reports/stage5p3/03_CAPACITY_AUDIT.md、src/socialvae_attribution/model.py、src/socialvae_capacity/model.py、冻结官方源码与本阶段配置；计数须以capacity_audit.json实测记录交叉核查。
