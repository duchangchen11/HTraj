# Stage11B-1 冻结开发协议

协议：**HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY**。

基础提交620532a2b70f6cc8c6a26f9b0e37ad57565763e9，官方MIT Y-Net源码a345ada1557f53dc2acb4c07f71d340d366980da。只新增Stage11B-1文件；旧训练adapter只读复用。

25个训练录像7496目标；5个验证录像998目标；17个测试录像2829目标保持封存。验证录像固定为bookstore_2、deathCircle_3、gates_7、hyang_7、nexus_4。`results/tables/stage11b1/data_audit.json`保存来源、独立加载器文件SHA、清单SHA、图像SHA及身份检查。8494个来源训练/新验证目标的history/GT SHA重新计算一致；录像与窗口跨划分不重合。测试pickle未加载。

每个窗口8历史+12未来，原帧步长12，作者报告30fps，采样间隔0.4秒；坐标及所有误差为原始参考图像像素。不是ETH/UCY米制协议，也不是Y-Net论文官方测试成绩。处理池中每(scene,track)只有一段20帧，不臆造滑窗步长；重叠时间和同背景不等于独立行人/未知场景。跨录像自然人身份尚未核实。

预测核心seed17随机初始化，不加载作者预测权重或旧smoke/one-pass权重。分割权重SHA75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194，所有参数冻结、eval、BN不更新；原cv2 BGR预处理保留。Adam0.0001、目标batch4、场景batch1、不增强、resize0.25、waypoints[11]，原始Goal/Path平均BCE各乘1000；训练GT waypoint只用于原训练监督。无AMP、额外梯度裁剪或调度器。详见`configs/stage11b1/train.json`。

K20：TTST10000 proposal→19原始KMeans中心+1 softargmax目标→20完整轨迹，T1、阈值0.01、Ke20、Ka1、CWS关闭。预测API没有未来参数；每sample_id稳定SHA种子，与目标排序/GT无关。所有998个预测完成后才评分。ADE1/FDE1同首候选；minADE20与minFDE20分别取完整路径最优；目标采样点GoalFDE与路径终点FDE分别计算。详见`configs/stage11b1/evaluation.json`。

最多30完整epoch，每轮全7496个目标恰好一次；固定验证1/5/10/15/20/25/30，完整验证最低窗口加权minADE20选best，精确相同取早轮。所有指标来自同checkpoint。预算总8小时、安全收尾7.5小时；先用训练目标完整TTST K20测时。若30轮+7次验证预估不满足预算，**正式训练不启动**，只提交预算审计。不得降低K、proposal或目标数。

已知限制：SDD数值语义ID/名称与ignore映射未完全核实；当前分割checkpoint的确切训练录像清单未公布；原短轨迹生成manifest不完整；新验证从作者原训练池切出，分割器可能已见其语义标签。原论文补充4.1说明分割器按对应轨迹训练场景训练，但不能据此证明当前权重没有新验证曝光。这些限制允许开发实验，不支持独立未知场景泛化或论文创新结论。

预先固定图示规则：若产生真实训练/验证记录，绘制实际epoch曲线，不补点；best完整验证目标按minADE20的.05/.10/.15与.85/.90/.95顺序统计量取低/高误差各3例；按minGoalFDE20的.85/.90/.95取目标误差3例，允许重合并披露。原图叠加真实像素轨迹与ID/SHA；GT只用于离线评价/绘图。Python为已保存绘图偏好。
