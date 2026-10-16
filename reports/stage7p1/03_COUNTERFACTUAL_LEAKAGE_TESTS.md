# 原始数据级未来反事实与训练负例

官方版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；[源码与配置](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。本阶段模型执行代码版本 `6d07b7f4c697aa79142f383fe4040d133e1531b4`，逐文件SHA见 `results/tables/stage7p1/execution_setup.json`。

从两个真实混合标签窗口取原始四列20帧记录，写入本阶段独立cache文件，再由 `Recording.read`重读后重新构建上下文；没有只改预处理tensor。原始来源文件保持不动。五类变更分别是：全部未来位置变化、邻居未来删除、邻居未来缩短、添加仅未来出现行人、修改监督目标之外不完整未来邻居的坐标。HOTEL首个混合窗口邻居仅有1帧未来，缩短至0与删除相同；两项结果不视为独立证据。ZARA2邻居2帧缩短至1帧。

每项检查原始历史行、历史完整ID、部分历史ID、历史位置SHA、模型pre_motion/pre_vel、scene_orig、agent_mask及历史context_enc一致。K1采用learned prior的零epsilon，K20采用稳定录像/窗口/行人ID产生的同组epsilon，未将潜变量固定成绕过learned prior的常量。K1不称为预测分布数学期望。

| 录像 | 原始数据变更 | K1最大预测差/m | K20最大预测差/m | 历史/资格/scene_orig/关系mask |
| --- | --- | --- | --- | --- |
| biwi_hotel | future_only_actor | 0.0 | 0.0 | 全部一致 |
| biwi_hotel | future_positions | 0.0 | 0.0 | 全部一致 |
| biwi_hotel | neighbor_future_deleted | 0.0 | 0.0 | 全部一致 |
| biwi_hotel | neighbor_future_shortened | 0.0 | 0.0 | 全部一致 |
| biwi_hotel | non_target_future_changed | 0.0 | 0.0 | 全部一致 |
| crowds_zara02 | future_only_actor | 0.0 | 0.0 | 全部一致 |
| crowds_zara02 | future_positions | 0.0 | 0.0 | 全部一致 |
| crowds_zara02 | neighbor_future_deleted | 0.0 | 0.0 | 全部一致 |
| crowds_zara02 | neighbor_future_shortened | 0.0 | 0.0 | 全部一致 |
| crowds_zara02 | non_target_future_changed | 0.0 | 0.0 | 全部一致 |

10/10通过，两个K下所有最大预测差均为0。真实修改行数、原始反事实文件路径/SHA和逐项布尔值见 `counterfactuals/*.json`。原始数据副本保存在被忽略的 `experiments/stage7p1/cache/raw_counterfactuals/`，不上传轨迹。

## 不能用监督mask修复原始训练后验

以下是**故意无效的补值负例**，不是真实监督训练。在相同真实历史与完整监督GT下，所有未来不完整邻居的监督mask全置0，只将其未知未来填为末位置或末位置+100m。固定随机状态并关闭dropout比较目标q；还计算目标q对未知邻居补值的梯度。未更新权重，未以该loss汇报预测能力。

| 录像/开始帧 | C/S人数 | 监督q均值最大变化 | 对未知邻居未来梯度L1（补值1） | NaN loss有限 |
| --- | --- | --- | --- | --- |
| biwi_hotel/14950 | 2/1 | 0.559740543 | 0.106153749 | False |
| crowds_zara02/8620 | 9/7 | 0.499241531 | 0.262135141 | False |

两窗口监督目标q都改变且存在直接梯度；sample loss在两种有限补值间相同，因为它走历史先验，但MSE随训练后验改变。KL在这些随机权重例中均被min_clip=2截住，**本例没有实测KL数值变化**；未mask的KL依赖由原loss源码和自动化公式测试确认，不能据此编造数值变化。NaN哨兵使后验、原loss非有限；“NaN乘零”并不屏蔽未知标签。

这证明当前冻结实现的mask-only接法不正确，不证明所有可能的因果CVAE都不能实现。训练后验可合法使用完整监督目标GT；本轮失败点是未知邻居未来参与后验且KL没有对应目标语义。没有编写替代后验、KL或loss来强行通过。
