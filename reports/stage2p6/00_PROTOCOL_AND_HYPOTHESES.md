# Stage2.6预训练协议与假设

日期：2026-10-14。从远程/本地一致、干净的fb8373997134c476f98a1912cf82fbceb536ac39建立独立分支。原源码/配置/报告/表格和所有已有.pt文件222项SHA256保护。旧实验目录均不覆盖，本阶段只做本地commit。

Stage2.5的真实结果：1024/96、三个种子best ADE/FDE均值，PedBase0.439398/1.070541m、Dynamic-RN0.440075/1.063977m、A-R0.442214/1.073782m。A-R固定checkpoint二阶条件ADE收益0.0047–0.0084m，三阶0.0000046–0.0023m，不能代替独立训练。q3 raw读出范数约80–104、beta低，梯度有限非零；残差不是零gate，但特定邻居关系的价值未证明。无邻居41个ADE目标中65.85%只有一个历史点，密度误差混合历史信息量。来源均为四份Stage2.5原报告及其实际CSV/checkpoint hash，保留不改。

## 预先固定的可证伪假设

- H-order：独立训练的H2比H1提供更多预测信息；必须对照同融合、同d和近容量H1-Cap，不能把删阶推理当训练对照。
- H-third：H3比H2有稳定附加价值；若仅状态尺度/参数增加而无三种子一致收益，三阶暂不保留为性能贡献。
- H-time：严格升时比允许同帧边扩展更有效；如果NoOrder不差或更好，则严格时间约束必要性不成立。
- H-relations：正确邻居关系比明确标注合法性/分布偏移的重连图更有预测价值；固定checkpoint的敏感性只能描述使用情况，不能证明独立泛化。
- H-walk：大量高阶覆盖可能来自两人往返与相同身份链的时间重复；要精确计算身份重复与可选择中间行人数，而非只看support覆盖。

以上全部**待验证**，不预写正向结果，不以开发数据作正式测试。

## 同一R基础与初始化

H1/H2/H3都从完整H3 A-R随机初始化模板删除多余阶模块，分别从头训练，没有加载旧训练权重。encoder/head/图/共同投影、carry gate、scorer、delta/gate初始值在同seed完全相同。d=64、同半径4m/k=8、同L2/Adam0.001/batch8/clip1。gate初始0.05，不加LN、碰撞loss、augmentation或额外decoder。

参数：H1=73627、H2=94812、H3/NoOrder=115997。差距大，预先增加H1-Cap/H2-Cap，将同一两层无bias社会delta MLP的内部宽度从64改395/230，输出仍64；参数115995/116060。主H1/H2/H3隐藏维度完全一致。容量辅助控制的参数位置/非线性优化不同，所以是近参数量参考，不叫严格等有效容量证明。

## 严格与非严格传播的计算图

令E_t为完全相同历史候选图及原alpha，m为新收到的消息，c=gamma*s_old，第一阶始终读取同一h_{t-1}。

严格版（与原A1/A-R数值/梯度相同）：

$$m_t^1=E_tW_1h_{t-1},\quad m_t^q=E_tW_qs_{t-1}^{q-1},\quad s_t^q=c_t^q+m_t^q.$$

NoOrder仅放宽高阶边时间相容性：

$$\widetilde m_t^1=E_tW_1h_{t-1},\quad\widetilde m_t^q=E_tW_q(s_{t-1}^{q-1}+\widetilde m_t^{q-1}),\quad\widetilde s_t^q=c_t^q+\widetilde m_t^q.$$

两版均用原continuity/支持mask，维度/参数相同，每帧每阶一次projection+aggregate。NoOrder多一个张量加法，允许边时间t1≤t2≤t3，仍是因果前缀、不是任意逆时间信息。它保留原旧路径源项，没有用新lower STATE去额外乘低阶carry gate；仅新lower MESSAGE参与同帧扩展。不打乱观测帧，不改运动学输入。

NoOrder本质上增加同帧walk，路径数/状态尺度会随时间约束变化，不能同时声称“相同walk总数”。比较隔离了一个计算规则，但路径数和数值尺度是机制相关后果；若需要排除它们，还需单独尺度/路径预算控制。本阶段不混加状态归一化。

人工正序/逆序/同帧/单边无高阶/多条相容路径、前缀/梯度、padding/置换/无邻居、同init/参数和同投影次数先通过。对应模型测试未通过不得启动训练。

## 数据、终点、诊断与冻结对照

原fold_ETH manifest和stage2p5/subset_1024.json原字节复用，1024/同96验证，17/42/2026各30epoch/3840步，best最小ADE精确tie最早、配对FDE，完整报告last30。不评价ETH、不五折、不筛seed/换验证集。

PedBase/Dynamic-RN冻结引用需审计：所有旧数值源码与旧训练commit逐字一致、共同协议字段/索引/hash一致、best/last checkpoint hash一致、CPU重新评价与旧GPU指标误差<1e-6m。新组只扩展报告，不改指标公式或选择终点；失败则不引用。旧完整epoch曲线路径保留在frozen_baselines.csv中。

每epoch新模型记录截止可见目标的before/source/message/carry/after范数均值/P95、全历史after、beta、gate分布、delta和实际修正、clip前投影/carry梯度以及损失/指标/训练秒。每epoch选择指标仍原GPU metric_sums；best/last的全部59组从该epoch已经保存的预测在CPU计算，无重新选择，设备归约末位差另说明。分组源于原严格历史图oracle，与候选模型或干预图无关。

关系控制分为同4m候选集合的最大可变重采样、放宽4m但保持同场景连续端点/每target入度的随机/最大变更、保持入出度的双边交换、删除间接路径及相同保留度的随机删边。保持/放宽条件逐行标注，固定种子，不以强负对照误差给出独立重训练性能证明。本阶段这些只做推理，单独重训练的图负对照明确未运行。
