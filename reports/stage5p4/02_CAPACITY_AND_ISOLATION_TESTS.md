# Stage 5.4：容量与隔离检查

训练启动前运行tests/test_socialvae_capacity.py：对位置、速度、邻居数增减、排列、进入时刻、未来槽和NaN邻居进行干预，检查确定性K1、公共噪声K20、监督后验、原始loss和参数梯度完全相同。另检查训练/推理原始输入Jacobian无依赖、目标未来不进入历史状态、严格checkpoint恢复、MLP实际非零梯度和更新、结构死参数保持不变。

原官方enc在no_grad中构造原始特征，所以单凭autograd返回None不能证明Full不使用社会信息。本控制同时采用输入干预；不把detach误认为隔离。

审计scripts/stage5p4/audit_capacity.py在冻结ZARA2训练池的四个预定位置各取16目标（共64），只前后向，不做优化。记录module forward调用、loss连接、非零梯度标量以及结构上界。长期无梯度情况由正式训练每个step的梯度并集与每轮参数审计记录，不能用一次probe代替。

原始目标、历史和GT SHA核对覆盖两个划分的全部训练与验证目标；三个CV重新计算核对误差。全套测试及真实probe结果保存在results/tables/stage5p4/。训练前配置与代码提交SHA由provenance.json记录。最终追加训练后隔离、历史保护验证和完整测试记录。

## 最终真实验收

完整测试337项通过；历史保护4526文件SHA未变。第一阶段结果与checkpoint在第二阶段前另建保护清单且核验未变。四组正式run均100完整epoch；未读ETH指标。

|模型|样本范围|名义参数|loss连接模块参数|结构有效上界|实测非零梯度标量|未观测非零梯度标量|
|---|---|---|---|---|---|---|
|Full|64_target_fixed_training_probe|2144002|2144002|2045698|1783553|360449|
|NoSocial|64_target_fixed_training_probe|2144002|1656514|1459906|1278481|865521|
|EgoCapacity|64_target_fixed_training_probe|2669570|2182082|1985474|1602243|1067327|
|PlaceboSocial|64_target_fixed_training_probe|2144002|2144002|2045698|1699502|444500|
|NoSocial|biwi_hotel|2144002|1656514|1459906|1455838|688164|
|NoSocial|crowds_zara02|2144002|1656514|1459906|1456224|687778|
|EgoCapacity|biwi_hotel|2669570|2182082|1985474|1981885|687685|
|EgoCapacity|crowds_zara02|2669570|2182082|1985474|1982792|686778|
|PlaceboSocial|biwi_hotel|2144002|2144002|2045698|1951344|192658|
|PlaceboSocial|crowds_zara02|2144002|2144002|2045698|1950566|193436|

64目标probe为4批只前后向、0优化更新；录像行是100轮梯度并集。loss连接模块参数含恒零输入列，结构上界另外排除这些标量；实际非零梯度受数据和激活影响。长期未观测非零不等于永久理论死参数。Full历史训练没有逐标量梯度并集，不能从checkpoint精确倒推该数；旧Adam二阶矩仅是代理，未填成精确值。参数数量接近不代表函数类/秩/归纳偏置相同。

训练后：EgoCapacity与Placebo各自最佳checkpoint均通过真实历史目标上的位置、速度、数量、顺序、进入时间干预；K1、公共噪声K20、监督后验与原始loss全部严格不变，原始邻居输入Jacobian无依赖；有限差分也为0。原Full的no_grad特征构造不等价于无邻居依赖，本阶段不借此抹去其社会输入功能。诊断未更新模型权重。

所有原始单测XML/TXT、每轮参数梯度、训练后隔离/数值Jacobian、checkpoint恢复和来源SHA均保留。真实数据manifest、目标历史/GT SHA和公共采样seed与冻结Full/NoSocial一致。Placebo社会路径实际执行、各社会模块获得非零梯度，但恒定初始偏移与合成分布仍限制容量解释。
