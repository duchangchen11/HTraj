# Stage 5.4：容量与隔离检查

训练启动前运行tests/test_socialvae_capacity.py：对位置、速度、邻居数增减、排列、进入时刻、未来槽和NaN邻居进行干预，检查确定性K1、公共噪声K20、监督后验、原始loss和参数梯度完全相同。另检查训练/推理原始输入Jacobian无依赖、目标未来不进入历史状态、严格checkpoint恢复、MLP实际非零梯度和更新、结构死参数保持不变。

原官方enc在no_grad中构造原始特征，所以单凭autograd返回None不能证明Full不使用社会信息。本控制同时采用输入干预；不把detach误认为隔离。

审计scripts/stage5p4/audit_capacity.py在冻结ZARA2训练池的四个预定位置各取16目标（共64），只前后向，不做优化。记录module forward调用、loss连接、非零梯度标量以及结构上界。长期无梯度情况由正式训练每个step的梯度并集与每轮参数审计记录，不能用一次probe代替。

原始目标、历史和GT SHA核对覆盖两个划分的全部训练与验证目标；三个CV重新计算核对误差。全套测试及真实probe结果保存在results/tables/stage5p4/。训练前配置与代码提交SHA由provenance.json记录。最终追加训练后隔离、历史保护验证和完整测试记录。
