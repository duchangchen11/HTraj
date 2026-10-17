# 官方评价等价性（smoke前冻结）

来源：[test.py](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/test.py)、[eval.py](https://github.com/Khrylx/AgentFormer/blob/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5/eval.py)。版本及SHA在 `evaluation_equivalence.json`。

对目标窗口i、20候选k，ADE_ik=mean_t ||pred_ikt−GT_it||₂，FDE_ik=||pred_ik,12−GT_i,12||₂；先分别min_k，再按目标窗口平均。不能固定同一个最优k同时报ADE/FDE，也不要求同一场景全部行人共用一个最优k。原AverageMeter按len(agent_traj)加权；UNIV两个sequence按目标数合并。五场景最终宏平均每场景权重1/5，不能用全部目标池平均冒充它。重叠窗口不是独立行人样本。

官方CVAE infer从learned prior采样；完整DLow infer使用mean=True令z=b，输出20个学习得到的latent head，**不是20次独立Gaussian先验采样**；share_eps=True影响训练，原配置train_w_mean=True训练亦用b。未来encoder只为训练后验/recon，真正基准必须读samples目录，而非recon。test.py先算recon再infer，二者分别保存。K1辅助固定为同一infer输出的第1候选，CVAE称首个先验样本、DLow称第1latent head，不称数学期望或GT最优。FPC等额外后处理不启用。

原save_prediction按未来frame/actor ID写四列绝对位置，恢复cfg.traj_scale后以%.3f落盘；eval通过frame/ID与原17列GT对齐。ETH必须乘2，其他折乘1。原test.py的strict=False不能用于本项目验收，单独严格检查CVAE与DLow全部参数；不改官方文件。

实际检查：人工构造ADE赢家0、FDE赢家1的反例，通过原函数与独立公式；首样本与best20明确区分。使用官方crowds_zara02_val的真实完整20帧窗口，以CV加固定偏移生成20候选，仅作评价程序工程检查；原save_prediction序列化→原align_gt→原eval.py CLI与独立参考一致。另保存精确GT recon目录，但只评价samples，未将重建当预测。未计算模型基准成绩。

浮点及失败记录：首次自写汇总断言用了浮点精确相等导致失败，原eval CLI与逐目标公式已正常；修正为有解析真值的1目标误差1、3目标误差3反例，目标加权均值必须2.5而非窗口均值2。旧输出/失败栈保留，新执行目录v2，未改官方评价。三位小数写盘引入的实测差值见JSON（<0.001m）。

12项新增契约测试通过：五折隔离、ETH内部scale=2的米制还原（用ETH折的HOTEL训练文件，非ETH测试成绩）、完整20帧未来存在性资格、独立指标赢家、目标加权、错误K拒绝。新旧数据协议不得混排。官方五场景论文标准复现表需完成全部五折后才可建立；目前仅能列文献作者报告值，标明不可直接严格比较的差异。


---
执行顺序更正：上述规则/核验在smoke前已写成，但完整材料补充commit被新日志末尾空行检查拦住；6场景12工程更新结束后才进入93e8a5a。不得把标题的“smoke前冻结”解释为完整证据已预先入Git。初始预算协议c28350f已预提交；未运行正式epoch。详见06报告与commit_order_actual_outcome.json。
