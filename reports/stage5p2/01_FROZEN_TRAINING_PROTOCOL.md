# Stage 5.2 冻结训练协议（正式更新参数前提交）

研究目标：在既定严格因果协议下建立公开 SocialVAE 正式开发基线。本阶段不修改模型、损失、网络模块，也不训练 NoSocial。

## 版本与保护

- 基础提交：`b7bcd4a5bfb47cc5c5d4f19d3d3f05dafe8ad301`。
- 分支：`feat/stage5p2-socialvae-baseline`；只本地提交，不自动推送。
- 官方源码：`eb23cbae562bac2c2dcf0d81529e6ab2ac280962`。由原 `src/socialvae_causal/official.py` 校验源码 SHA。官方 forward、模型参数结构和 loss 保持原样。
- `configs/stage5p2/protected_artifacts.json` 保护此前 2569 个文件，含已登记旧 checkpoint；验收时逐项重算，不覆盖 Stage 5.1 smoke 权重。

## 数据与固定预算

完整配置：`configs/stage5p2/protocol.json`。数据：`HTraj-SocialVAE-Causal8x12-v1`，8/12 帧、0.4 秒、米制。沿用 Stage 5.1 两份 B manifest，原始文件由冻结 loader 校验 SHA。目标与输入/GT SHA 逐项核验记录在 `results/tables/stage5p2/cohort_verification.json`。

| Run | 验证录像 | 训练目标窗口/轮 | optimizer steps/轮 | 验证目标窗口 | 验证行人ID |
|---|---|---:|---:|---:|---:|
| A | crowds_zara02 | 23545 | 184 | 1092 | 35 |
| B | biwi_hotel | 27145 | 213 | 318 | 28 |

训练排除整个验证录像；其他录像仅使用旧 train 区域。旧 validation 不进入训练；ETH 不加载、不评价。这两个开发录像已经历多轮研究曝光，不能称为独立确认集。窗口重叠，目标窗口也不是独立行人。

每个 run 独立执行 seed17 随机初始化，不加载 smoke 或官方权重，也不继承另一 run 训练权重。相同种子导致初始参数相同是预设行为。原始 SocialVAE，hidden=256，radius=2m，2,144,002 参数；Adam，lr=0.0005，batch=128，weight_decay=0，无学习率调度、梯度裁剪或数据增强。100 个完整 epoch，保留末尾不足 128 的 batch。每轮 `default_rng(17+epoch)` 打乱全部目标且验证恰好访问一次；两个 run 相同规则。首次非有限值即停止该 run，保留现场，不反复重试；另一预注册 run 独立执行。

## 评价、采样与选择

主规则：每轮完整验证集 K1 latent-prior 确定性轨迹 ADE 最低，完全相同选更早轮；FDE 配对同一 checkpoint。它是将潜变量取条件先验均值后递归解码的轨迹，**不是预测分布数学期望**。

每轮保存 K1 prior、K1 首随机样本、minADE@20、minFDE@20；K20 各指标独立选最优候选。FPC 关闭。另保存原始监督 posterior loss、重建和 Monte Carlo KL 估计，不称为精确边缘 NLL。训练 loss 按当轮各 batch 目标数加权，训练过程中参数不断变化；验证 posterior 是固定该轮权重后的诊断，二者随机数及分布均不同。

评价只临时在原 p_z/q_z 返回的 Normal 分布上提供固定标准高斯 epsilon：条件 loc/scale、log_prob、forward 与 loss 不变；不注册新参数或持久模块，训练前移除全部 hook。每个目标分别使用 CPU generator，通过 `protocol|seed17|recording|start_frame|target_id` 与先验/后验不同 namespace 的 SHA256 前 8 字节（取模 2^63）生成 seed。先验 noise 为 `[20,12,32]`，后验 `[12,32]`；索引布局严格对应官方 K×B 重复。各轮使用同一目标噪声。`sampling_seeds.csv.gz` 保存 seeds 与 noise SHA。epsilon 对 batch 排列/分割精确不变；浮点 batched 运算仍允许舍入误差，测试预测容差 atol/rtol=1e-5。评价 fork/恢复 RNG，避免扰动后续训练序列。

第 1、5、10、20……100 轮及选定 best checkpoint 另外对完整训练集评价确定性 K1，供泛化差距分析；其他轮训练 ADE 留空，不虚填。每轮保留模型参数范数/最大绝对值/有限性/状态 SHA、梯度范数均值/P95/最大值、显存峰值、数据/计算与整轮时间。

## CV 与历史分组

三种 CV 均用同一 Stage 5.1 监督目标重新计算，并核对 ID、历史和未来 SHA 及既有逐目标误差：Last2、OLS4、OLS8。统一 `X_last + k*0.4*v_est`，OLS 使用物理秒回归，预测锚定最后真实历史点。主对照固定 Last2；事后逐录像最佳 CV 仅称“解析基线上界参考”，不能视为一个可部署新模型。

所有分组只用八帧观测，阈值在训练前固定（本阶段定义，不冒充此前阶段同定义）：

- 七个历史连续速度模长均值：静止 <0.05 m/s，低速 [0.05,0.4)，正常 [0.4,1.8)，高速 ≥1.8。
- 转弯：最初/最后三个速度平均方向夹角 ≥30°，两者速度均需 ≥0.2 m/s；否则单列方向不可靠。
- 方向变化：可靠相邻速度（均 ≥0.2 m/s）最大夹角 ≥45°；无可靠方向单列。
- 合法邻居：2m 内第零帧可用位置初始化，或第1–7帧有连续两帧速度的消息；同时统计末帧合法邻居 0/1–2/3–5/6+、七帧平均密度 0/(0,1]/(1,3]/>3。
- 新进入邻居、近距离邻居速度不可观测均分有/无。运动关系以末帧合法源与目标速度 ≥0.2m/s 判断，同向 cosine≥0.7、相向≤−0.7、其余交叉；无可靠方向单列。径向相对速度阈值 ±0.2m/s。多邻居可多标签，不假定组互斥。

全部预定义组（含空组）保存目标数、行人ID数及各模型误差。分别报告目标窗口加权、先行人ID内平均再行人等权、以及各口径的两个录像宏平均；不混用权重，不对重叠窗口做独立样本显著性推断。

## 保存与停止

每轮不可覆盖 checkpoint；best/last 仅引用其中的文件，不复制旧文件。第 1、10……100 轮和 best 严格验证模型、optimizer 与 RNG 恢复。保存全部训练曲线、逐步 loss/梯度、逐轮验证目标误差和运行日志，权重不入 Git。

固定完成两个 100 轮 run 后停止。若最优点靠近边界或末段仍改善，如实报告预算截断，不自动延长。无 NoSocial 不能声称社会信息具有独立贡献。K1 与 K20 结论分开，单种子两录像不能证明跨场景稳定性，也不把常规 VAE 或采样策略当创新。
