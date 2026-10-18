# Stage11A总报告：SDD语义与目标点研究准入

**最终NO-GO：当前不能启动Stage11B正式训练或SCI新算法开发。** 完成了资源、几何、公开基线、指标与文献审计；没有正式训练、没有新模型test成绩或下一阶段自动运行。

1. **数据与许可**：官方SDD8地点/60录像，CC BY-NC-SA3.0。原视频、完整作者split/config/权重/语义图未取得。六个镜像录像的图像/标注共23.19MiB，仅PARTIAL，官方一致性UNVERIFIED；未下载69GB视频或作者完整大包。详情01与resource_inventory.json。
2. **坐标对齐**：六张真实8点历史叠加图及初版可读性修订保留；位置/resize/逆变换核对通过，所读lost=0中心均在各自图内。没有视频/参考帧ID/语义图，不能证明精确原始或语义配准，不能称米制精度。详情02、alignment_results.json。
3. **语义**：Y-Net论文有5种手工stuff，但文件/palette/分割来源未取得，源码默认6通道关系未解；不能把RGB或Semantic Drone Dataset当SDD标签，未生成伪标签。详情03。
4. **基线可行性**：作者原Y-Net核心在Torch2.5.1下B1/B2合成前向通过；工程核心参数1,641,381，不含分割，非完整作者配置。缺SMP/语义权重，原ragged loader失败，Pandas.append和epoch早停等需隔离适配审查，未修改官方源码。PECNet保留参考，Stage7结构结果只引用。详情05与baseline_probe.json。
5. **协议风险**：官方短期训练notebook把test_trajnet作为validation选模。本轮不执行、不用test调参。protocol_draft.json仍DRAFT，确切train/val/test清单未取得，背景跨split共享未知；不能把录像隔离叫未见场景。详情04。
6. **评价差异**：Y-Net FDE为采样goal误差，未必decoded-path末点误差；PECNet GT先挑终点后解码并非全部K paths独立minADE。已制定K1/K20、GoalFDE/TrajectoryFDE、像素/米、oracle、语义边界/出图和重复窗口规范，未造指标。详情08。
7. **查重**：10篇比较；7篇方法正文可读（IAD仅预印版本），CINet/McTrans/GDS-TGTF完整方法未取得。Scene+goal不是新题，SSU已做语义升级，UATP2026已研究语义地图不确定性；不能据摘要宣布机制空白。详情06与literature_comparison.csv。
8. **唯一待验证问题**：优先考虑不可靠语义对goal覆盖/路径约束的影响，当前只是诊断问题；SDD实际预测损害未证明，没有新网络/算法创新声明。最小实验、公平对照与停止条件在07，不为凑数提出第二方案。
9. **资源**：RTX3080工程核心合成前向已测，20次合成decoder调用无OOM、allocated峰约375MiB，未含分割/梯度/TTST/CWS，不能视作官方K20完整推理。正式训练GPU时间尚不可估，下一次若授权先≤1GPU小时工程准入，完整epoch实测后另报。当前不训练。
10. **测试与保护**：全仓库旧501项+新增审计工具15项，516项通过；保留完整pipeline两项失败，不把unit通过当正式复现。最终历史SHA以history_after.json为准；旧源、配置、权重和指标不变。源blob/数据/图像/代码hash及失败记录在results/tables/stage11a。

下一步建议仅补齐作者数据/语义/config资源并审查许可、真实split、分割标签曝光及隔离兼容；Y-Net是条件性复现对象，PECNet是参考。完成这些条件之前不冻结正式实验、不提出新模块或宣布SCI创新资格。

新分支`feat/stage11a-sdd-semantic-goal-audit`，从远程Stage10A已推送521c53a建立；最终commit SHA以Git交付为准。本阶段明确授权推送新分支；如认证失败保留本地commit与完整错误日志，不force、不动main。完成后停止。
