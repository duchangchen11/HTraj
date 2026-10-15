# Stage3.8总报告：历史运动可靠性与稳健速度估计

完成四个零训练参数解析模型在全部五个固定非ETH开发录像上的统一评价，以及8次合法兼容神经checkpoint推理。没有新神经训练、gate、速度截断、额外估计器、ETH测试或自动push。

| 录像 | N目标窗口 | CV-Last2 ADE/FDE | CV-OLS4 ADE/FDE | CV-OLS8 ADE/FDE | CV-Median ADE/FDE |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | 318 | 0.329516 / 0.611038 | 0.237814 / 0.438866 | 0.220144 / 0.408760 | 0.218247 / 0.402738 |
| students001 | 1169 | 0.380756 / 0.833056 | 0.427223 / 0.891753 | 0.507109 / 1.005962 | 0.514984 / 1.020309 |
| students003 | 706 | 0.727884 / 1.627788 | 0.811086 / 1.737995 | 0.931942 / 1.901824 | 0.949848 / 1.937206 |
| crowds_zara01 | 297 | 0.430096 / 0.953000 | 0.465470 / 0.994360 | 0.520416 / 1.076965 | 0.532430 / 1.094044 |
| crowds_zara02 | 1092 | 0.296306 / 0.667523 | 0.331875 / 0.715683 | 0.383175 / 0.787222 | 0.387353 / 0.796291 |

| 口径 | 模型 | 录像数 | N窗口 | 录像×行人单位 | ADE | FDE |
| --- | --- | --- | --- | --- | --- | --- |
| recording_macro | CV-Last2 | 5 | 3582 | 177 | 0.432911 | 0.938481 |
| target_window_weighted | CV-Last2 | 5 | 3582 | 177 | 0.422970 | 0.929466 |
| person_macro_within_recording_then_recording_macro | CV-Last2 | 5 | 3582 | 177 | 0.529462 | 1.117984 |
| recording_macro | CV-OLS4 | 5 | 3582 | 177 | 0.454694 | 0.955731 |
| target_window_weighted | CV-OLS4 | 5 | 3582 | 177 | 0.460170 | 0.973170 |
| person_macro_within_recording_then_recording_macro | CV-OLS4 | 5 | 3582 | 177 | 0.553703 | 1.141019 |
| recording_macro | CV-OLS8 | 5 | 3582 | 177 | 0.512557 | 1.036147 |
| target_window_weighted | CV-OLS8 | 5 | 3582 | 177 | 0.528687 | 1.068718 |
| person_macro_within_recording_then_recording_macro | CV-OLS8 | 5 | 3582 | 177 | 0.633007 | 1.264180 |
| recording_macro | CV-Median | 5 | 3582 | 177 | 0.520572 | 1.050118 |
| target_window_weighted | CV-Median | 5 | 3582 | 177 | 0.536888 | 1.084020 |
| person_macro_within_recording_then_recording_macro | CV-Median | 5 | 3582 | 177 | 0.643257 | 1.283186 |

**结论：三个平滑估计器全部只改善HOTEL，其余4/5录像的ADE和FDE均退化；没有通过预注册规则的候选，停止继续修改CV估计器。** HOTEL的成功保留为条件现象，不能用于重新选录像。可靠性量包含难度信息，但不能区分噪声与真实运动变化，也未形成已验证的跨录像路由规则。

归因边界：本轮改变的是解析外推中的速度估计，没有替换既有神经encoder的速度/加速度序列，也没有对神经输入做同结构反事实训练。因此不能把HOTEL上解析平滑胜过神经模型，解读为“换神经速度特征即可获得同样收益”，或认定神经泛化失败已被唯一归因给速度噪声。与既有神经模型的比较用于强基线约束，不构成该内部机制的因果实验。

已完成988项旧Git文件和142个checkpoint共1130项SHA保护；3582目标窗口、177录像×行人单位完整配对。所有20个历史组、560条含强对照的分组行、19,970条模型逐目标误差保存；窗口并非独立统计样本，另存逐人均值、配对差和两种必需平均口径，不报iid窗口p值。

强对照：PedBase-M0及最佳兼容PedBase覆盖HOTEL/students003/ZARA2；最佳兼容CV残差覆盖HOTEL/ZARA2。缺失项留空，未借用泄漏录像的模型；共享覆盖宏平均另表，不能与五录像宏平均混用。旧模型与新解析指标的float32/64舍入差异已明确。

第四章暂不把当前社会GNN或可靠性协同网络当作已证实主线。最有依据的是协议/基线诊断材料，当前没有足够的新模型创新与独立验证支持直接投稿；下一步应调整可证伪问题及证据设计，非继续堆模块。本轮全部真实输出保留，完成本地commit后停止。

## 交付索引

- 01_MOTION_RELIABILITY_AUDIT.md：评价前固定公式、阈值、数据与强对照。
- 02_KINEMATIC_BASELINE_RESULTS.md：五录像、宏/窗口/行人口径、强对照及真实耗时。
- 03_CROSS_RECORDING_ANALYSIS.md：历史统计、配对行人差值和描述相关。
- 04_FAILURE_CASE_ANALYSIS.md：八个问题、所有历史组、固定案例及失败限制。
- 05_RESEARCH_DIRECTION_DECISION.md：A至E支持/反对证据及停止决定。
- configs/stage3p8/protocol.json：公式、manifest/raw SHA、冻结旧checkpoint及规则。
- results/tables/stage3p8/：完整结果、版本SHA、配对表、单元测试与交付校验。
