# Resonance技术审查

官方源码`7358b3d4fd3fcf160c18be9425445a67ee8211ee`；Qpid=`c7233b3f7e7305294b1d5644e695b44727ac0bea`；Luna=`b25d44354a75f7652c25c37aa9c16b353578ad79`。GPL-3.0源码独立置于忽略缓存，未复制进HTraj模型目录，三个仓库tracked内容均未修改。playground可视化子模块未初始化，训练/推理不需要GUI。

## 原结构与计算路径

输入为目标X_i∈R^(B×8×2)、相对邻居X_/i∈R^(B×N×8×2)，可选类别仅异构数据集使用；输出B×20×12×2。完整模型实测3,149,192参数，与论文补充Full参数相同。

Ŷ_i=Y_linear,i+ΔY_self,i(z_s)+ΔY_re,i(z_r)。LinearDiffEncoding用历史最小二乘线性拟合、平移使最后历史点连续，编码真实运动与线性拟合的Haar谱差。SelfBias用4层encoder-decoder Transformer预测waypoints，再作速度连续插值；社会分支用2层Transformer、邻居中心化运动谱成对乘积及角向汇总。作者配置feature_dim128、8heads、Haar/Haar、8角分区，Kc20风格通道、K=1重复构成20候选。

这是单阶段端到端模型，不是CVAE/DLow；训练目标为原Qpid l2/best-of-K距离，无新增loss。本轮直接调用原ADE_2D损失核心，对完整有效GT和单位目标权重计算；未声称完整LossManager/原train入口已经运行。源码[model.py](https://github.com/cocoon2wong/Re/blob/7358b3d4fd3fcf160c18be9425445a67ee8211ee/resonance/model.py)、[社会汇总](https://github.com/cocoon2wong/Re/blob/7358b3d4fd3fcf160c18be9425445a67ee8211ee/resonance/_resonance.py)、[差分运动](https://github.com/cocoon2wong/Re/blob/7358b3d4fd3fcf160c18be9425445a67ee8211ee/resonance/linearDiffEncoding.py)。GraphConv位于多风格候选解码中，不应将其误认为直接跨行人图传播；显式社会信息来自rc与ReBias。

## 真实工程执行

新stage8环境目录复用既有torch2.5.1+cu124/numpy1.26.4依赖只读路径，没有改动旧环境；作者声明torch2.0.0，故这是兼容性执行，非完全相同环境。Qpid导入要求cwd存在dataset_configs，第一次空目录导入失败；使用独立annotation元数据后正常。未修改网络或源码以掩盖该约束，也未调用原train_or_test入口。

只选官方HOTEL fold的*_val文件，采用其完整20帧目标/上下文，直接转换为原模型输入；这是validation适配器，不是Luna官方loader复现。单人、两人、五人、最大42人窗口均保留所有上下文。每个case前向K20、原距离目标反向、梯度有限；无optimizer更新。邻居存储排列差异最大约1.91e-6，平移还原误差最大约7.63e-6米（同随机噪声）。这些是结构数值测试，不是预测优劣证明。

|用例|上下文/目标batch|输出|allocated峰值(MiB)|秒（多前向+反向）|
|---|---:|---|---:|---:|
|single|1/1|[1, 20, 12, 2]|40.59|0.4597|
|two|2/2|[2, 20, 12, 2]|41.45|0.0987|
|multiple|5/5|[5, 20, 12, 2]|42.81|0.1409|
|maximum_validation_context|42/42|[42, 20, 12, 2]|89.39|0.1130|


随机初始化roundtrip权重strict恢复成功。第一次工程probe使用3_7_11 waypoints，不是发布ZARA2的2_5_8_11；已在记录中区分，未以其宣称作者精确复现。

## 作者权重核查

权重仓库Re分支冻结`42d88ba95e45f441f72318d673c5effd5ef356ec`。预先固定ZARA2文件P8_haar_fixed_epoch129.pt，按其2_5_8_11与其他shape参数严格加载，missing/unexpected keys均空；真实42人validation前向有限，未计算ADE/FDE，原文件SHA未变。该权重的源训练录像可能包含当前validation片段，故此项仅是工程功能，不是泛化成绩。

发布的**rehotel目录是no-linear-base/no-self-bias消融**，args为300epochs、output_pred_steps=all，不可充作Full-HOTEL。ZARA2发布args为200epochs、lr1e-4，论文主文写lr3e-4；不能默认所有fold同超参、所有目录均Full。未读取best_ade_epoch或events数值选权重。

论文说RTX3090、batch1000、200epochs；本轮只测≤42目标的小batch，不能保证原batch1000的完整训练成本。模型结构便于关闭self/re分支，但公平归因必须**分别从头训练**并审计容量，不能仅将推理no_interaction开关当新模型消融。结果目前支持工程可运行，不构成小论文选题GO。

原代码的邻居列表允许包含ego；no_interaction也会保留ego并padding其余位置，因此它不保证社会残差为零。learn_re_bias关闭则减少有效容量。后续若解释真实外部邻居的贡献，需要单独容量对照，不能将这些开关的推理结果直接当独立社会收益。
