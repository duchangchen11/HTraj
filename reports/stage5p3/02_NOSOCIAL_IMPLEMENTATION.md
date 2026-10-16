# 严格NoSocial实现与训练前冻结协议

新增类 `src/socialvae_attribution/model.py::SocialVAENoSocial` 继承冻结官方类，**仅覆盖enc**。原始构造顺序、参数注册、prior、posterior、latent维度32、decoder、递推、loss和输出均保留。名义参数2,144,002，相同seed17初始化的全部权重与Full初始fingerprint一致；绝不加载训练后的Full权重。

`h_x,0 = zeros(layers,B,256)`，`m_t = zeros(B,128)`，`h_x,t = GRU_fx([0,embed_s(v_t,a_t)],h_x,t-1)`；监督后验的`rnn_by`输入同样为`[0,embed_s(v_t,a_t)]`。目标未来只进入原始posterior和重建标签。NoSocial的enc不读取邻居的数值、数量、mask、排序或进入时间，不通过`0*neighbor`伪隔离；即使邻居值为NaN也不参与计算。继承forward对邻居张量的切片不构成数值依赖。

输入仍是x[8,B,6]、y[12,B,2]、neighbor[20,B,M,6]；历史motion特征[7,B,4]经原embed_s变为[7,B,128]，零社会消息补足原GRU输入256维。监督后验的未来目标motion为[12,B,128]。输出prior[12,B,2]或samples[20,12,B,2]。恢复全局坐标仍由原decoder位移累积加最后历史位置。原特征构建的no_grad保留；没有新预测loss。

未使用的`rnn_fx_init`、`embed_n`、`embed_k`、`embed_q`保留在state_dict但不调用，也没有loss梯度；两个GRU输入权重的前128列恒无有效输入。有效容量见03报告，不能宣称等容量。所有新权重只保存在`experiments/stage5p3/checkpoints/`，不覆盖旧模型。

## 训练前冻结

配置 `configs/stage5p3/protocol.json`；基础提交 `bc62cf89570c584f297f71e368107015c5498f0c`；分支 `feat/stage5p3-social-attribution`。正式更新参数前提交该配置、源码和结构测试。

- 使用`HTraj-SocialVAE-Causal8x12-v1`，8/12帧，0.4s，米制；Stage5.1两份manifest逐字沿用，训练/验证目标ID和历史/GT SHA与Full核对，三种CV重新校验。
- 顺序执行NoSocial-ZARA2、NoSocial-HOTEL，各自全新seed17模型；每轮23545/27145目标，184/213个batch，保留末batch；100个完整epoch。训练排除整段被验证录像，其余录像只用旧train区域。ETH封存。
- hidden256、radius2接口、Adam lr0.0005、batch128、weight_decay0；不调度、不裁剪、不增强、不改loss。每轮`default_rng(17+epoch)`完整置换；优化/训练RNG规则与Full一致。
- 每轮完整验证集K1 latent-prior ADE最低，完全相同取更早；配对FDE及全部采样指标共用该checkpoint。另报last100轮。K1 prior不是预测数学期望；原loss是rec+MC KL估计，不称精确边缘NLL。
- 原样调用`src/socialvae_baseline/evaluation.py`：使用**Stage5.2**的稳定目标ID seed namespace、相同K20预算/原始标准噪声、FPC关闭，K20两项独立选最优。全部分组与阈值也原样沿用。
- 保存每轮模型、loss分项、梯度、参数SHA、完整验证逐目标误差、运行时间/显存；定期严格恢复测试；新增逐参数实际非零梯度生命周期统计及结构死参数不变检查。无NaN/Inf时按预算完成，不根据成绩调参；非有限值停止该run并保存现场。
- 主归因看同目标Full−NoSocial的K1 ADE，FDE配对；K20另分析。预设“值得验证”：相对ADE改善≥1%且FDE恶化≤1%，两个录像和两种权重同时满足。仅一录像为条件性；两录像都无改善则停止增加复杂社会模块；小收益/口径冲突如实列出。无论结果如何容量混杂未排除前不宣称独立社会因果收益。
- 本阶段只完成两组NoSocial，不追加seed、epoch或Ego-Capacity-Control训练。容量控制单独设计/预算见03，等待后续阶段决定，绝不把未运行指标填0。

隔离测试固定目标历史、固定潜变量噪声，对邻居位置/速度/数量/排列/进入时间/未来槽及非有限值干预；包含训练loss、后验状态、K1/K20、原始输入Jacobian、Full敏感性和无社会Full等价计算。Full原始输入Jacobian的detachment局限不隐藏。完整测试结果另保存。

Stage5.2已记录CUDA跨batch拆分存在毫米级浮点差异；主评价固定相同batch128与目标次序，不通过换batch挑成绩，也不声称消除了所有数值批处理差异。

## 最终实测补充

训练后best权重的50项NoSocial检查（含两录像、K1/K20/后验/loss及Jacobian）全部严格不变。原始输出见`trained_isolation_audit.json`和日志。Full相同固定目标与噪声的功能干预保留输入敏感性，示例最大坐标变化如下（不同干预取最大，仅用于功能诊断，不是预测误差）：

| 录像 | 模式 | Full最大坐标变化/m |
| --- | --- | --- |
| HOTEL | K1 | 0.602607 |
| HOTEL | K20 | 0.792106 |
| ZARA2 | K1 | 0.468043 |
| ZARA2 | K20 | 1.484838 |

位置平移、速度feature修改、复制候选等是人工接口干预，不应称为同分布预测负对照或公平重训练结果。Full原始Jacobian为空仍源于官方特征detachment；NoSocial还同时有源码无读取和精确输入干预证据。

## 主结果的证据边界核对

固定步长0.01m、历史距离<1.9m的3个源位置坐标，中心差分Jacobian结果如下。选择坐标只依赖历史，不用未来GT。Full原始autograd虽然detach，数值差分仍有非零响应；NoSocial严格为0，其他batch目标也无依赖。此小规模功能诊断不能当作全部目标敏感性上界或预测收益证明。

| 录像 | 模型 | 坐标数 | 最大数值Jacobian | 其他batch目标Jacobian |
| --- | --- | --- | --- | --- |
| HOTEL | Full | 3 | 0.010204315185546875 | 0.0 |
| HOTEL | NoSocial | 3 | 0.0 | 0.0 |
| ZARA2 | Full | 3 | 0.008487701416015625 | 0.0 |
| ZARA2 | NoSocial | 3 | 0.0 | 0.0 |
