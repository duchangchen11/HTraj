# Stage11B-2B：有限训练延长与Goal覆盖持续性审计

适用范围：PROTOCOL_DEVELOPMENT_ONLY / CONVERGENCE_DIAGNOSTIC_ONLY。单seed17、5个开发验证录像、998个重叠目标窗口，并非998个独立场景。单位为SDD参考图像像素，不能与ETH/UCY米制排名。分割语义ID映射与预训练成员身份未完全核实，验证可能存在语义预训练曝光。未读取封存测试集，未新增模型、loss、调参或数据增强。

**执行：COMPLETED_EPOCH50；H1：H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET。**

完整恢复原Epoch30模型/Adam/RNG，从Epoch31继续到50，累计94100更新，新增GPU进程wall 10443.724秒。不存在从随机权重重新训练、空Adam续训、修改学习率/结构/loss或测试选模。首次GPU准入的合成标签越界失败、修订及重新冻结全部保留；正式训练未发生对应错误。

| epoch | ADE1 | FDE1 | minADE20 | minFDE20 | A25_miss_count | A25_miss_fraction | A25_error_share | GoalPathMismatch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 32.093100 | 60.539021 | 16.403136 | 25.634029 | 164 | 0.164329 | 0.610817 | 7 |
| 35 | 33.985591 | 63.409883 | 18.381538 | 28.586645 | 212 | 0.212425 | 0.675726 | 4 |
| 40 | 32.110529 | 61.329038 | 17.425740 | 27.620325 | 208 | 0.208417 | 0.672282 | 7 |
| 45 | 34.647615 | 66.144515 | 18.762894 | 30.458641 | 243 | 0.243487 | 0.690487 | 4 |
| 50 | 34.464677 | 64.960104 | 18.562667 | 29.347204 | 210 | 0.210421 | 0.654702 | 2 |

最终Epoch50与固定Epoch30：minADE20变化+2.159531px（正值为退化）；minFDE20 25.634029→29.347204px；ADE1 32.093100→34.464677px，FDE1 60.539021→64.960104px。上述维度分别解释，不能把oracle改善等同所有预测能力改善。

原164目标中恢复31，新失败77，持续失败133。A25误差份额61.0817%→65.4702%。H1按最终完整验证而非最佳checkpoint判断。

全体最佳Epoch30，新最佳Epoch40，均按窗口minADE20选。最佳文件 `experiments/stage11b1c/cache/checkpoints/epoch_030.pt`，SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。最后训练epoch与最后完整验证均单独记录。完整旧7+新4节点、NOT_RUN状态以及checkpoint/cache SHA见validation_metrics.csv、validation_status.csv和checkpoint_manifest.json。

报告导航：01恢复与模式；02执行、预算及选模；03固定/动态覆盖及K1/K20；04全部录像及历史组；05逐项科研问答和停止判断。10类图均有PNG/SVG与对应CSV/JSON，案例身份沿用上一阶段。

录像宏平均指标另见recording_macro_metrics.csv；窗口加权是主口径，两者不混用。

完整测试、历史SHA及交付检查见full_tests_final.txt/xml、history_after.json和delivery_audit.json（最终检查后生成）。未将skip写为pass。源码与数据来源见protocol_frozen.json及verified_validation_sources.json，GPU失败/成功进程均计费，旧成果保持只读。

本阶段不构成新算法，不证明充分收敛或语义因果，不自动启动任何后续训练。下一阶段至多建议在批准后进行覆盖机制与新颖性审查。

最终核验：完整pytest为590 passed / 1 skipped（原CUDA专项保留跳过）；生成31组PNG/SVG（9类定量图、21个固定案例对及1张案例总览），坐标与缓存逐项核对。deathCircle_3仍是主要失败来源（143/210），窗口加权H1不等同所有录像一致成立；录像等权A25误差份额为48.83%，详见04报告及事后逐录像敏感性表。
