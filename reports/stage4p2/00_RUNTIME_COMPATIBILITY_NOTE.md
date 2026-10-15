# 训练功能验证前的CUDA兼容性修复

首次G0/ZARA2执行在第1轮训练后、第一次验证积分时停止：PyTorch2.5拒绝在`torch.use_deterministic_algorithms(True)`下执行CUDA `cumsum`。没有非有限loss，也未完成任何验证指标/保存epoch checkpoint。原run_config、smoke_failure和日志全部保留，不作为成功实验或零指标。

这是评价积分内核兼容问题，不是超参数或结构失败。保留严格确定性开关，把评价中的12步累积位移改为固定顺序的12步加法；数学预测公式、NLL、网络结构、优化器、种子与数据协议不变。增加CUDA自动化测试，核查严格确定性执行和与CPU累积公式一致性。修复先提交，再从同一种子重新开始；全部四个有效run使用相同修复，路径增加`deterministic_scan/`，不覆盖首次记录。

中止尝试的部分训练耗时未单独测得，不混入有效30轮曲线。有效训练仍须5轮smoke、同checkpoint恢复后继续至30轮。未因验证成绩更换超参数。
