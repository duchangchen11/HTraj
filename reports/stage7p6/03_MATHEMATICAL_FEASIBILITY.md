# 数学可行性与原模型语义变化

结论：可以定义不虚构未来GT的训练目标，但这需要更改原AgentFormer后验、身份映射及损失聚合，不能称loader修复，也不是已经证明有效的新算法。以下仅为静态数学审计；没有实现或训练替代网络。

## 条件与张量

令c=|C|、s=|S|>0、U=C\S，H_C=E(X_C)，X_C∈R^{c×8×2}，Y_S∈R^{s×12×2}。历史token为8c×d；真实future token仅12s×d；d=256，z_i∈R^32（所审HOTEL/ZARA2配置）。用当前和先前历史定义C、scene_orig和历史关系；训练标签资格S单独保存。推理对所有C预测，再由评测程序提取固定目标，不能先凭未来存在性删节点。

历史缺帧但不满足完整8帧者不在本问题C中。S=∅时没有本定义的监督目标，该窗口跳过监督训练，不能除以零，也不把U当静止监督。

## 为什么只mask原始loss不够

原FutureEncoder把12c个future token自注意力融合后，cross-attend 8c个history token，逐agent时间池化给q_i。其forward没有读取fut_mask。原generate_mask与AgentAware身份分解假设query/memory有同一个agent_num；简单将12c改成12s而不改身份映射会把“同一agent”判错。原KL对所有c聚合并clip，重建/sample即使乘零mask仍按c归一化；加入U稀释监督梯度。NaN×0无效，未知位置作为零/末值参与future attention同样是伪标签。

## 一种自洽但常规的受限变分构造

固定历史后，维护版先验是各节点对角高斯乘积（共享历史上下文不等于全协方差）：

\[
p_\theta(Z_C\mid X_C)=\prod_{i\in C}p_{\theta,i}(z_i\mid H_C).
\]

为说明可行性，可以只为合法Y_S建后验token，自注意力只允许S标签；cross-attention的memory为C历史，query身份S与key身份C用真实ID对齐。未来token的速度只由该监督目标最后历史点和合法未来点得到。U没有“假future token”；但联合decoder仍给U潜变量和预测输出：

\[
q(Z_C\mid X_C,Y_S)=q_\phi(Z_S\mid X_C,Y_S)\prod_{u\in U}p_{\theta,u}(z_u\mid H_C).
\]

其中q_S可保持逐节点对角高斯分布参数输出，允许参数共享已标注节点的合法Y_S信息；这是训练后验，不是推理偷看标签。U由历史prior采样，不能detach prior重参数梯度后仍宣称优化同一个ELBO。没有标签不等于没有latent，也不意味着其latent应被强制置零。

采用原论文固定方差高斯重建解释，联合decoder是从X_C,Z_C到所有未来**均值**的确定性映射f_θ。其时序自回归输入是自身生成均值，不是未标注节点真实future。条件于Z_C，输出噪声独立的假设下，积分未观测Y_U后，已观测likelihood只保留S：

\[
\log p_\theta(Y_S\mid X_C)\ge
\mathbb E_q\!\left[\sum_{i\in S}\log p_\theta(Y_i\mid X_C,Z_C)\right]
-\sum_{i\in S}\mathrm{KL}(q_i\Vert p_i).
\]

U的KL因q_U=p_U逐项为0而消去，不是“给没标签的节点随意mask惩罚”。Y_U完全未被伪造；Z_U通过期望被边缘化。若换成以真实Y_U为输入的随机自回归likelihood，该推导不再自动成立，必须重新积分，不能只照搬这条式子。

该受限q有效但可能很松：Y_S也携带Z_U的信息，而q_U固定prior将忽略它。解析双高斯例`analytic_checks.json`与数值积分测试证明ELBO≤log evidence，同时gap可明显大于0。这是概率公式验证，**不是轨迹性能或新网络**。没有理由仅凭式子就预期ADE提高。

## 从数学ELBO到原工程目标

设m=s；完整12步目标的重建为L_rec=(1/m)∑_{i∈S}∑_{t=1}^{12}||Y_it−f_it||²。它沿用原始每目标时间求和尺度，不偷偷改成每坐标平均；KL_S=(1/m)∑_S KL，原cfg clip_min=2的操作位置应保持在平均之后；sample辅助项为(1/m)∑_S min_k∑_t||Y_it−Ŷ_ikt||²。

当S=C，分母、求和和原有sample/clip可以退回原目标形状。S⊂C时，上述后验、KL范围和分母都不同于冻结原实现。保留clip与best-of-K辅助项的工程目标**不是精确边缘NLL，也不再是纯ELBO**。数学上可定义不等于本阶段获准修改官方loss。

有部分future标签但不足12帧的U，此构造不使用其零散标签；若以后按逐时刻标签集合S_t训练，需要单独定义token/latent与监督权重，不是同一消融。本阶段不把其已知点与未知点一起填成常数。

## DLow不能自动沿用

DLow的历史qMLP生成20个(A,b)，mean=True使用b作为各head潜变量；不是20次独立CVAE prior抽样。它训练时reconstruction需要Y_S，而KL与diversity可对C定义，因为不需要GT。一个明确但尚未实施的选择是recon按S平均，KL/diversity按C平均，保持两项合法的无标签正则；若三项都改S，正则对象也改变，需作为不同目标报告。

新CVAE后验/训练上下文改变后，需要重新训练CVAE和与其配对的DLow；旧作者权重只证明原协议性能，不能当新协议训练结果。重新定义DLow聚合不是原论文完全复现。

## 缺失机制与选择偏差

上述观测likelihood是给定标签索引模式的可定义目标。跨样本直接优化它以恢复全体未来分布，还需要缺失机制可忽略等统计假设。行人离开视野可能与未来位移相关，属于潜在非随机缺失；S可携带未来存在性信息。这里只可声明在固定有标签队列上的估计与评价，不能声称已识别所有行人的真实运动分布。没有建模缺失机制，本构造不解决MNAR偏差，也不支持将“未观测未来”当作真实终止事件。

## 必须申报的语义变动

|位置|变化|性质|
|---|---|---|
|历史loader与origin|C仅按8帧历史；S与GT分离|协议改变|
|FutureEncoder|12S queries，对8C memory；agent-ID矩形映射|核心后验与attention身份语义改变|
|U latent|posterior无标签token，使用历史prior|变分族改变|
|KL/重建/sample|标签范围和分母显式定义|训练目标改变|
|decoder输出|继续预测C；只监督/评分S|节点输出与评价接口需声明|
|DLow|新CVAE配对，标签loss/无标签正则区分|重新训练和目标审计|

这是可行性推导，不是可直接启动的最小创新方案。一般目标/上下文分离、mask及缺失变量边缘化已有研究；目前没有数学机制区别足以通过新颖性门槛。
