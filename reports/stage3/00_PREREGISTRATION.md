# Stage3训练前方案冻结

基础2d786785813ee298af1cb148999e13a21ffab0c6；分支feat/stage3-relation-routing。所有旧源码/配置/测试/文献/历史报告/结果和本地权重由configs/stage3/protected_prior_artifacts.json保护。仅新增文件，不编辑旧模型或结果。新模型固定H1、d64、8→12、1024/96、seed17、30epoch、Adam0.001/batch8/clip1、原masked per-track L2、best开发ADE最小和配对FDE、last30。ETH封存。

主要比较CV、冻结PedBase、冻结Dynamic-GNN-RN、从头Receiver-Fix、从头Relation-Routing。机制附表独立训练Routing-Current、Routing-NoUpdate、Routing-Uniform、Routing-Random。NoUpdate与History逐参数同形同初值，仅每帧GRU先前状态清零；Current换为单帧小MLP，参数不同且如实报告；Uniform保留未使用打分参数，报告名义与死参数；Random保持真实图每target入度，源有连续两点、同场景、无自环/重复，明确放宽4m空间约束，不当作纯同分布消融。

Receiver-Fix保持原7维打分MLP、消息projection、carry gate、残差融合、decoder，原同帧双端连续规则改为target当前可见/source有连续两点。receiver carry仍需自身连续。未知target速度用显式feature[6]=2标记（旧已知关系为1），相对速度/CPA槽位仅是缺失占位，tau未定义时占位0、closest退回当前距离；按距离排序。旧已知target的建边/feature/消息计算完全不变。源当前两点速度参与建图，消息仍是旧h_source,t-1，不伪造其前一帧缺失速度；保留该延迟和可能弱payload作为局限，未改成h_t。接收政策新增边上必须定义缺失处理，故不是脱离缺失语义的单纯布尔翻转。

关系输入12维：[相对位置2，已知源速度2，已知时相对速度2，tau1/closest1，target/source速度有效性2，历史可见比例2]。GRU隐藏24，参数跨场景/行人/时间共享。关系消失/任一身份中断/时间gap/重新出现时reset；只影响直接消息alpha，不进行边间传播、增加阶数或改运动编码器/解码器。新进入receiver可接收但其edge-memory也从0开始。

第二轮候选筛选已固定于configs/stage3/protocol.json：最多2个社会模型；整体best ADE至少优于PedBase和Dynamic各1%，配对FDE/完整8帧ADE容忍退化最多1%；或一帧有邻居组>=10可评价目标且ADE改善>=10%，整体ADE/FDE对两个基础最多退化1%，完整8帧ADE最多退化1%。History额外须比等参数NoUpdate ADE好至少0.5%才能提出记忆贡献。合格按整体ADE/FDE排序；17沿用同一次首轮、42/2026从头相同30epoch，不重抽seed。若无模型合格，不盲跑第二轮，不增加模块。

确认集先按原全部548个非ETH验证窗口与开发96的完整20帧支持比较，外加200 frame-ID ticks=8秒隔离。至少30个互不重叠窗口才认为规模可用。结果：没有任何候选；即使隔离缩减为0也没有支持不重叠窗口。configs/stage3/confirmation_manifest.json逐窗口保存冲突数。不能创建独立确认集，不能以重用开发集代称独立验证；不改划分、不动ETH。没有确认性指标可用于结构调整。

所有人工结构/完整旧测试通过后才能训练。本文件与协议在训练前提交；结果、失败和停止决定后续写入规定报告，不回写筛选阈值。
