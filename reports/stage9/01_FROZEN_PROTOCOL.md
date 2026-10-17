# Stage9 冻结训练协议

本轮由用户明确授权一次真实受控训练，替代Stage8“暂不训练”的阶段停止指令；不改变既有失败结论。基础提交 d9bfed15abf32c9004b82bc41862f20f955dd8a2；新分支 feat/stage9-resonance-pairing-controlled-pilot。只比较A/B/C，不添加模型、seed、损失或后续阶段。

## 数据与推断范围

使用Stage7.2冻结的AgentFormer官方HOTEL折：7个其他来源的`*_train`、`*_val`，不读取留出的`biwi_hotel.txt`。完整8帧历史和12帧未来资格同时确定目标与上下文，0.4秒、米制、K20。原始输入14文件逐字节与Stage7.2 SHA核对。每个目标记录sequence/frame/person_id/history_SHA/GT_SHA。所有目标保留，无人数裁剪；训练3,118窗口/29,676目标窗口/最大57人，验证688窗口/5,203目标窗口/最大42人。

逐来源frame集合与(frame,actor)集合无交集；完整20帧支持全部位于相应区域；原始frame编号未重置。train/val边界相邻，存在同录像相近时间的相关性，不称录像隔离或独立确认。仅凭同actor ID或重复静止位置不能判定时域泄漏。`source_isolation.json`给出精确区间和重合检查。未使用测试集误差。

适配数据入口而非官方Re预处理复现。上下文为全部完整20帧行人，包含ego；外部邻居分组排除ego。邻居坐标提前减去目标末历史位置；Qpid原Move只移动目标输入并恢复预测。统一填充至57，用原INIT_POSITION和原mask，不将padding当真实观测。

## 固定网络与优化

Re 7358b3d4fd3fcf160c18be9425445a67ee8211ee；Qpid c7233b3f7e7305294b1d5644e695b44727ac0bea；Luna b25d44354a75f7652c25c37aa9c16b353578ad79。第三方源码不改。Full：128维、8分区、Kc20、Haar/Haar、waypoint 2/5/8/11、noise_depth16、move100、K=K_train=1。原线性分支、SelfBias、ReBias和输出完全保留。

seed42随机初始化；新增分支seed4201；不加载任何作者或smoke权重。相容共享张量在初始时完全相同，结构测试逐键核对。Adam lr=0.0001，默认betas/epsilon，无scheduler、无梯度裁剪、FP32，无增强，仅原Move。batch128，最后不足128仍训练；无梯度累积。目标顺序由`default_rng(9000+epoch)`确定，每轮完整访问29,676目标，三个模型完全相同。选择此固定lr是受控适配实验，不声称复刻作者所有训练超参数。

原Qpid默认l2实际为`ADE_2D`：完整轨迹欧氏距离先对12步平均，再取20候选最小值，对目标平均。全12帧标签、unit loss weight、valid mask=1，直接调用原函数；测速同时与原`l2.forward`逐值等价核对。不改变训练目标。

## 预算及终点

每模型50次真实训练测速，测速参数丢弃。未计算任何验证误差前，按P90每步时间、全部三模型、数据/checkpoint/验证额外开销以及1.2安全系数，从200/120/80/40降序选定共同预算。最终配置`configs/stage9/training.json`必须本地commit成功才启动正式训练。

固定共同最终epoch checkpoint作为主要比较。无验证挑轮、无best ADE/FDE拼接。每20轮保存模型/Adam/RNG/checkpoint SHA，最终严格恢复。验证仅最终执行，三个模型按同一稳定目标顺序、batch128及由batch目标ID哈希派生的固定torch噪声seed推断。分别取完整候选minADE20、minFDE20，不逐步换head；候选选择为离线GT oracle指标，不是可部署选择器。

GPU总预算43,200秒，包含预检、测速、训练、验证、恢复及持有CUDA进程时的I/O；接近预算则停止，禁止缩减目标/K或只延长某模型。

## 预设分组与筛选

沿Stage8：同8角分区至少2个外部邻居，末帧径向距离差≥0.5米、历史平均速度向量差≥0.3m/s。保存歧义/非歧义、0/1/多个外部邻居及逐来源全部结果。

先BvsA，再CvsB。C继续条件：窗口与source+person等权ADE都降低至少max(0.005米,B的1%)；两种权重FDE退化不超过max(0.005米,B的1%)；至少2来源ADE下降；歧义组ADE下降，其他预设组ADE/FDE退化不超过max(0.02米,B组的5%)；参数/协议/预算均合法。群体空组只标记无样本。单seed开发筛选不宣称跨seed稳定、新颖性或显著性，不做重叠窗口IID检验。
