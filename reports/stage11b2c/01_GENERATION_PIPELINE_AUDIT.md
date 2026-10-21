# 冻结生成链路与信息隔离

模型仅为Epoch30，checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。原K20缓存SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`。开发协议及998目标不变，封存测试不读取。单位为SDD参考图像像素。

原始生成：history8→距离热图+冻结RGB语义→encoder→goal logits[11]→Sigmoid权重→小于max×.01的权重置0→float32全局归一化→固定sample_id CPU Generator有放回10000次抽样→原作者GPU KMeans19→prepend Softargmax→按原顺序解码20条完整路径。没有改网络、损失、采样或参数。

`adapter.py`逐条保留Stage11B-1R Predictor的全部前向/采样语句；AST单测对原return之前的计算作一致性检查。被动导出发生在完整预测之后，额外阈值权重重算也不反馈到采样或模型。保留原精度logits、sigmoid、阈值后未归一化权重、实际原归一化float32权重、int64索引、原始10000点、19中心、Softargmax、K20目标和路径。新CPU统计中的float64比例归一化仅用于数学概率质量/熵，不进入torch.multinomial。

CPU准备阶段核对历史/GT身份SHA，并将实际原始未来记录改为1000000后重新投影为history8，验证输入不变。GPU只加载物理隔离的 `{sample_ids,scenes,history}` 文件、RGB、冻结模型与GT-free原预测/哈希参考；不导入未来标签或误差表读取器。全部998目标完成正序导出、反序全张量逐位回归之后，独立CPU进程才读取GT进行邻域评分。身份核查读取GT不等同生成时利用GT。

全量门槛：每目标匹配原probability/weights/proposal/goal/path五项SHA，以及原缓存goals/paths逐位相同；全部998再反序生成，所有被导出中间张量逐位相同。固定训练2目标先做3次重复/反序、原Predictor对照及GT反事实/语义重算。任一核心不一致则停止机制结论。

Sigmoid不是全局概率；原float32归一化权重也可能因舍入使float64求和不精确为1。离线用p=w/sum64(w)定义数学离散分布，记录原总和；未改变用于抽样的原始float32权重。熵仅是该分布描述，不代表不确定性已校准。

画布坐标按原实现 `x_original=x_network/.25`，padding仅在右/下方。概率质量和样点距离均在原像素域计算；单独记录原参考图、resize后尺寸、网络画布、padding概率质量和GT内外标记，不把padding称为真实场景。

原始来源：冻结官方 `ynet/test.py:99-122`、`utils/image_utils.py:92-119`；维护仓库commit `a345ada1557f53dc2acb4c07f71d340d366980da`（MIT）；Stage11B-1R原adapter只读。
