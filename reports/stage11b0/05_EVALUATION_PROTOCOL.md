# 后续评价规范：公式冻结，正式实验仍未批准

8历史/12未来，0.4s，原SDD参考图像像素。公式实现evaluation.py只在人工数据上测试，不打开真实val/test评价。预测恢复原像素scale后统一计算：ADE1=首候选逐时L2均值；FDE1=该候选完整路径末点L2；minADE20/ minFDE20分别对20条完整路径取最小，各自可来自不同head。

GoalFDE单列为采样目标与GT末点误差，不能用Y-Net官方Goal FDE冒充完整路径FDE。K1首TTST候选对应softargmax预测goal后解码，不是分布数学期望。K20候选预算Ke20×Ka1；TTST10000是proposal，不是10000最终轨迹；CWS关、T1、rel_threshold0.01。未来所有对照使用同样对象、候选数和采样规则。

GT只用于离线oracle误差，不提供在线选模能力。minADE与minFDE不同head的人工反例已测试。推理无future参数，训练期GT teacher forcing不会流入该函数；本轮只执行工程K1依赖测试，无预测指标排名。

未来拟按稳定sample_id产生torch/NumPy采样seed（base17），单独保存随机状态。窗口加权和录像宏平均分开，不能假定共享录像/重叠支持为iid。预测出图不clip后再评价；非有限输出停止诊断，不删难例。语义合规指标待类别/ignore/边界规则可靠后再启用，不把合规等同安全。

未来选模建议：独立开发集最低minADE20、同值取早轮、配对FDE来自同checkpoint；只是方案，必须另行批准冻结训练预算和完整数据协议。本阶段不保存按真实精度选出的best，无测试选模，不将不同输入/单位/预测时长的表格混排。
