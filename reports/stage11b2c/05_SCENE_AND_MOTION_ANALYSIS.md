# 录像与冻结历史运动分层

适用范围：PROTOCOL_DEVELOPMENT_ONLY，冻结seed17 Epoch30、SDD5个既有开发验证录像、998个重叠目标窗口；不是998个独立场景，不做显著性检验。单位为SDD参考像素；不与ETH/UCY米制排名。未训练/更新参数、未用测试集、未改TTST/KMeans/head顺序。观察性分层不等于因果。语义映射及预训练曝光限制沿用旧阶段。

| scope | targets | P_mass_mean | proposal_cover_fraction | final_cover_fraction | final_fail_support_zero | final_fail_positive_sampling_miss | final_fail_after_proposal_hit | compression_loss_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_998 | 998 | 0.235310 | 0.921844 | 0.835671 | 77 | 1 | 86 | 0.086172 |
| deathCircle_3_443 | 443 | 0.189948 | 0.846501 | 0.699774 | 67 | 1 | 65 | 0.146727 |
| OTHER4_555 | 555 | 0.271518 | 0.981982 | 0.944144 | 10 | 0 | 21 | 0.037838 |
| bookstore_2 | 423 | 0.273872 | 0.978723 | 0.943262 | 9 | 0 | 15 | 0.035461 |
| deathCircle_3 | 443 | 0.189948 | 0.846501 | 0.699774 | 67 | 1 | 65 | 0.146727 |
| gates_7 | 24 | 0.282885 | 1.000000 | 0.958333 | 0 | 0 | 1 | 0.041667 |
| hyang_7 | 45 | 0.323147 | 0.977778 | 0.933333 | 1 | 0 | 2 | 0.044444 |
| nexus_4 | 63 | 0.214506 | 1.000000 | 0.952381 | 0 | 0 | 3 | 0.047619 |

五录像等权宏平均单独保存，不能与998窗口加权混用：

| radius | P_mass_mean | proposal_cover_fraction | final_cover_fraction | compression_loss_fraction |
| --- | --- | --- | --- | --- |
| 10 | 0.053590 | 0.952474 | 0.542648 | 0.409826 |
| 25 | 0.256872 | 0.960600 | 0.897417 | 0.063184 |
| 50 | 0.561877 | 0.971564 | 0.951034 | 0.020529 |

固定deathCircle443与其他4合计555同时报告。最终A25失败来源原样保持deathCircle133/164；不能用单录像高误差推出语义错误。原像素距离和参考图对角线归一化距离都保存在样本表及scene_mechanism_summary.csv。画像性质受录像构成和重复窗口影响。

| factor | group | targets | proposal_cover_fraction | final_cover_fraction | compression_loss_fraction | minADE20 |
| --- | --- | --- | --- | --- | --- | --- |
| hist_speed_group | high | 497 | 0.893360 | 0.766600 | 0.126761 | 23.886801 |
| hist_speed_group | low | 266 | 0.966165 | 0.958647 | 0.007519 | 5.452103 |
| hist_speed_group | medium | 235 | 0.931915 | 0.842553 | 0.089362 | 12.971617 |
| hist_turn_group | intermediate | 211 | 0.924171 | 0.805687 | 0.118483 | 15.671851 |
| hist_turn_group | straight | 322 | 0.940994 | 0.866460 | 0.074534 | 13.128514 |
| hist_turn_group | turning_ge30deg | 270 | 0.855556 | 0.722222 | 0.133333 | 29.206982 |
| hist_turn_group | undefined_direction | 195 | 0.979487 | 0.974359 | 0.005128 | 4.873345 |
| hist_speed_change_group | intermediate | 236 | 0.915254 | 0.788136 | 0.127119 | 19.450657 |
| hist_speed_change_group | stable_CVle0.15 | 394 | 0.936548 | 0.837563 | 0.098985 | 14.217757 |
| hist_speed_change_group | undefined_low_speed | 204 | 0.985294 | 0.980392 | 0.004902 | 4.574846 |
| hist_speed_change_group | variable_CVge0.4 | 164 | 0.817073 | 0.719512 | 0.097561 | 31.981155 |
| hist_stationary_group | all_steps_le1px | 147 | 0.979592 | 0.972789 | 0.006803 | 4.573171 |
| hist_stationary_group | other | 851 | 0.911868 | 0.811986 | 0.099882 | 18.446620 |

所有运动组来自Stage11B-2A冻结历史8帧标签，未用未来运动定义分组；frozen源SHA见protocol_frozen.json。类型与误差关联见mechanism_type_summary.csv、sample_level_mechanism.csv，均不能作因果归因。

历史组实际对照（25px，全部为既有历史分组，非因果）：

- hist_speed_group / low（n=266）：抽样命中后最终失覆盖率0.7519%；实际支持为0率3.3835%；最终失败率4.1353%。
- hist_speed_group / high（n=497）：抽样命中后最终失覆盖率12.6761%；实际支持为0率10.4628%；最终失败率23.3400%。
- hist_turn_group / straight（n=322）：抽样命中后最终失覆盖率7.4534%；实际支持为0率5.5901%；最终失败率13.3540%。
- hist_turn_group / turning_ge30deg（n=270）：抽样命中后最终失覆盖率13.3333%；实际支持为0率14.4444%；最终失败率27.7778%。
- hist_speed_change_group / stable_CVle0.15（n=394）：抽样命中后最终失覆盖率9.8985%；实际支持为0率6.0914%；最终失败率16.2437%。
- hist_speed_change_group / variable_CVge0.4（n=164）：抽样命中后最终失覆盖率9.7561%；实际支持为0率18.2927%；最终失败率28.0488%。

速度high/low是来源训练历史对角线归一化速度的冻结三分位组，不是绝对米/秒。上述群体有录像构成/重复窗口混杂，不构成独立因果或统计显著性判断。具体数值见history_motion_contrasts.csv。
