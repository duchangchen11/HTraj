# Stage3.7：训练前运动表示审计

本报告在本阶段训练前建立。基线来源为Stage3.6提交`e64c0aac49386a94075099fd1fc7ab5f545a533e`，所有旧源码、表格、配置、报告与130个旧checkpoint共973项SHA256保护，清单见`configs/stage3p7/protected_artifacts.json`。本阶段只新增文件，不修改原模型。

## 源码结论

1. `src/models/ego_encoder.py::motion_inputs`不把绝对坐标输入GRU。7维输入为速度2维、加速度2维、观测有效、速度有效、加速度有效3个标记。`positions`仅是返回元数据及decoder锚点，不是运动embedding输入。
2. `src/models/trajectory_decoder.py::TrajectoryDecoder.forward`通过共享MLP直接产生12个相对最后观测点的位移，并加回最后观测位置。它不逐步积分速度，也不采用未来位置教师强制。
3. `src/models/ped_base.py::PedBase.forward`只传历史、历史mask与timestamps到encoder；取最后状态与最后观测点到decoder。监督future仅进入外部loss/指标。
4. 因而原PedBase已具备数学上的平移等变性。把所有历史坐标减去最后位置不改变速度/加速度，输出再加回原点也不改变原模型的计算含义。M1记录为M0的等价别名，不能重复训练后冒充新的归一化收益。
5. 原模型并不保证旋转等变性：vx/vy、ax/ay进入普通全连接层，decoder输出在全局x/y坐标轴。方向分布变化是值得检验的泛化假设，不是已经证实的唯一原因。
6. 速度为相邻可见位置差除以实际timestamp间隔，加速度也除以该实际间隔。超过`1.5*0.4s`的间隔重置连续状态，不作为正常速度步。未来输出间隔固定0.4s，符合冻结协议；这不是任意不规则未来采样预测器。
7. 缺失速度数值槽位为0，但同时明确`motion_valid=False`；模型有条件区分未知速度与真实静止。历史断点重置GRU carry，padding状态与输出为0。没有发现把缺失速度冒充真实零速度的接口错误。

## 数据、分布及因果边界

仅`A_complete8x12_history_context`，目标8帧完整历史/12帧完整未来，0.4s、米制K1。未来mask只筛选评价/训练目标，历史上下文按历史完整性筛选。新M2/M3也只用历史定义局部朝向。

使用Stage3.5已经存在的`val_biwi_hotel.json`与`val_crowds_zara02.json`，与Stage3.6的有效窗口indices完全一致，训练排除整个验证录像，不提升任何旧validation窗口进入训练，完整20帧支持隔离；ETH封存。录像选择有历史开发结果曝光，不能称独立确认。

Stage3.6已保存历史分布显示：HOTEL训练/验证末帧速度中位数约0.5498/0.0791m/s，平均相邻速度变化0.0622/0.1449m/s；训练vx较宽而HOTEL验证vy较宽。ZARA2训练/验证末帧速度中位数约0.5489/0.0215m/s，亦有静止比例变化。因此不仅方向，还有速度、噪声/加速度及样本构成变化，坐标旋转不能自动解决全部差异。出处：`results/tables/stage3p6/historical_motion_distribution.csv`。

本阶段重新计算的目标数、密度、低中高速及可靠方向八扇区分布见`results/tables/stage3p7/split_counts.csv`和`history_distribution.csv`；数据SHA、原始非ETH数据来源SHA在`configs/stage3p7/protocol.json`。这些统计仅用历史运动定义分组，future_valid仅决定哪些完整目标纳入统计。

## 训练前决定

- M0：原PedBase，新训练；M1：M0等价别名，不额外训练。
- M2：同参数GRU/decoder，最后位置为原点，历史当前连续段平均速度为朝向；只旋转不缩放。
- M3：M2同样坐标策略，输出CV先验加神经轨迹残差。额外引用Stage3.6完全兼容的原坐标CV-Residual常学习率结果，检验CV先验固定时坐标策略的影响。
- 两录像、seed17、每模型30epoch，Adam0.001、batch8、hidden64、原L2，按最低验证ADE选择并保留配对FDE/last30。不比较学习率，不增加loss/GRU/社会模块。
- 低于0.2m/s或无有效速度时使用恒等坐标系；不伪造方向、不从future取方向。这个回退不保证低速组旋转等变性，测试必须显式显示边界，而不能声称全局等变。
- 训练后只做固定checkpoint的0.1/0.4m/s阈值敏感性、±90/180度旋转检查和同endpoint训练/验证误差，不增加训练或用这些补充结果选择结构。优化与归纳偏置的影响无法仅靠单seed误差完全分离，应保留限制。

结构测试通过后提交本报告、固定协议、代码及测试，再启动六次训练。后续只提出多seed建议，不自动启动。
