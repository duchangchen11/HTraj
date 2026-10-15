# Stage4.2最终验收总报告

两个录像上，G0/G1整体ADE和FDE均未超过三种重新配对的CV。 G1低于G0的录像：biwi_hotel；两录像预注册收益/FDE门槛同时通过：False。 停止当前HTraj逐人PTN图家族及继续堆叠社会模块的开发，不采用为第四章正式基础。

## 唯一100轮预算下的完全配对结果

|录像|模型|ADE(m)|FDE(m)|目标窗口|
|---|---|---|---|---|
|crowds_zara02|G0|0.514221|1.003209|5833|
|crowds_zara02|G1|0.605474|1.174621|5833|
|crowds_zara02|CV-Last2|0.325740|0.728451|5833|
|crowds_zara02|CV-OLS4|0.365459|0.780967|5833|
|crowds_zara02|CV-OLS8|0.428842|0.873811|5833|
|biwi_hotel|G0|0.980086|1.820474|1053|
|biwi_hotel|G1|0.933634|1.737428|1053|
|biwi_hotel|CV-Last2|0.322666|0.616897|1053|
|biwi_hotel|CV-OLS4|0.249866|0.478500|1053|
|biwi_hotel|CV-OLS8|0.241865|0.456838|1053|


CV均本次重新计算，ZARA2 5833和HOTEL1053目标窗口共同GT/ID/窗口，数据hash和逐目标history/future SHA保存；旧Stage3.8指标未用于替代。新旧预算独立列出，验证均值ADE选模、FDE配对，同一规则覆盖四个神经run；K1首采样/K20/NLL仅辅助。

延长前930f92d已冻结方案，全部从原epoch30恢复31–100、共280个新checkpoint，无模型/损失/epsilon/学习率修改。100后停止。训练及PTN审查均有实际日志，不把计划算作结果。固定权重扰动记录330528行，全体队列及全部运动组覆盖，不将一个极端案例推广全部目标。

社会归因门槛：

|录像|G1−G0 ADE|G1−G0 FDE|ADE达标|FDE达标|
|---|---|---|---|---|
|crowds_zara02|0.091252|0.171412|False|False|
|biwi_hotel|-0.046452|-0.083046|True|True|


报告：01冻结协议，02同目标CV，03训练预算，04全量PTN敏感性，05最终研究决策。真实配置/曲线/配对误差/分组/耗时/数据与checkpoint SHA/测试在`results/tables/stage4p2_final/`。新checkpoint只存本地`experiments/stage4p2_final/extension/`；全部历史由SHA保护。

限制：seed17、两个反复曝光开发录像、重叠窗口非独立、100仍可能未充分收敛。ETH封存，未运行正式测试或额外种子/G2/EqMotion，不把本结果外推所有GNN。下一路线建议为EqMotion公开版本兼容性审查，依据[原论文](https://arxiv.org/abs/2303.10876)与既有Stage4失败记录，尚不具备直接用其开发新方法的条件。

运行必要测试并校验保护清单，证据见`results/tables/stage4p2_final/delivery_checks.json`。仅本地commit，不push；最终SHA以git log为准。用户可手动`git push -u origin feat/stage4p2-final-audit`。完成后停止。
