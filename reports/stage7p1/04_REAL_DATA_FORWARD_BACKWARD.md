# 真实数据前向、原loss、反向与恢复

官方版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；[源码与配置](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。本阶段模型执行代码版本 `6d07b7f4c697aa79142f383fe4040d133e1531b4`，逐文件SHA见 `results/tables/stage7p1/execution_setup.json`。

共10个预设工程场景，覆盖真实N1/2/3/6/9/13/15、部分历史邻居排除、历史完整未来不完整/消失邻居及两个验证区域的最大上下文。窗口列表和来自train/validation的来源、选择理由保存在 `execution_setup.json`；均处于Stage5.1冻结支持，不做训练或模型选择。训练与验证含义没有互换。

| 录像/开始帧 | 历史完整N | GT完整目标 | 原loss | 梯度L2 | 原始训练路径 | 严格恢复 |
| --- | --- | --- | --- | --- | --- | --- |
| biwi_hotel/14950 | 2 | 1 | 未定义/未运行 | 未运行 | BLOCKED_UNLABELED_CONTEXT | True |
| biwi_hotel/16170 | 15 | 8 | 未定义/未运行 | 未运行 | BLOCKED_UNLABELED_CONTEXT | True |
| crowds_zara02/8760 | 6 | 6 | 2.432506 | 39.635890 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| crowds_zara02/8620 | 9 | 7 | 未定义/未运行 | 未运行 | BLOCKED_UNLABELED_CONTEXT | True |
| crowds_zara02/9300 | 13 | 13 | 106.033440 | 240.415588 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| biwi_hotel/1060 | 1 | 1 | 2.826255 | 92.319904 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| biwi_hotel/2770 | 2 | 2 | 518.230469 | 2947.158305 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| biwi_hotel/590 | 3 | 3 | 218.459747 | 1831.300206 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| crowds_zara02/990 | 1 | 1 | 463.592651 | 2957.473313 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |
| crowds_zara02/10 | 2 | 2 | 261.292786 | 2119.113093 | PASS_ALL_CONTEXT_LABELS_COMPLETE | True |

7个全标签因果场景执行官方 `set_data → forward → compute_loss → backward`；保留原训练模式、随机场景旋转、后验、MSE+KL+sample(K20)，loss与梯度有限。每例5,972,098个参数具有loss梯度张量，梯度非零总范数见表；这不是每个标量每次梯度都非零的声明。无optimizer构建或更新、无完整epoch、未载预训练。

3个混合标签真实场景：历史先验推理正常，**原真实监督loss/反向不执行**，原因在 `UnlabeledContextError`中保存；不是以零loss代替。HOTEL验证的15人高密度窗口8人GT完整、7人未来不足，未删邻居或困难目标。

10例均保存随机权重checkpoint并 `strict=True`加载到同结构新实例；恢复K1与原预测逐值相同。checkpoint路径/SHA保存在 `real_cases/*.json`，权重本体仅留本机被忽略cache。不涉及训练后optimizer状态恢复，因为本轮0更新。

## 训练入口兼容性

真实执行的是官方模型单batch训练调用链，不是官方 `train.py`完整epoch入口。原入口仍使用依赖完整未来筛选的DataGenerator/loader，**未通过严格因果训练准入**。没有运行官方CLI开始epoch，也未通过改官方preprocessor/np.int等源码掩盖问题。Stage7的运行时 `_LinearWithBias=Linear`导入别名沿用；官方文件不改。原始模型及loss接口可执行不等于混合标签训练可执行。

## 测试与失败记录

全量测试 395 passed / 0 failed / 0 errors / 0 skipped；见 `full_tests.xml/txt`。10项新增契约测试先通过，验证只查询历史、真实标签、NaN缺失、fail-closed、ID固定噪声、原始记录重读、原loss分母/KL数学行为。

首次pytest启动被本机ROS自动插件的缺失lark阻断，保留 `initial_test_launcher_failure.txt`；后续仅设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`隔离外部插件，不安装或更改模型环境。2个故意NaN补值负例、3个原训练拒绝完整保留；不能把“单测全部通过”解释为严格因果训练GO。
