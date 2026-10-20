# Stage11B-1C 正式训练前冻结

协议HTraj-YNet-SDD-Development-v1，性质PROTOCOL_DEVELOPMENT_ONLY。25训练录像7496目标、5验证录像998目标（bookstore_2/deathCircle_3/gates_7/hyang_7/nexus_4），17测试录像2829目标继续封存。数据源、history/GT/reference SHA与Stage11B-1完全一致，2个原出界训练点保留，不按误差过滤，所有目标每完整epoch恰好访问一次。

作者源码a345ada1557f53dc2acb4c07f71d340d366980da（MIT），Stage11B-0训练adapter与Stage11B-1R采样器均只读。seed17预测核心从头初始化，不加载旧/作者预测权重。仅加载已核验分割权重，冻结参数、eval与BN。原Goal/Path BCE各乘1000；GT waypoint只允许训练teacher forcing。Adam0.0001、batch4/scene1、无增强/AMP/新梯度裁剪/调度器，最多30完整epoch。

正式训练前需通过新入口数据/参数/训练/恢复/采样/指标/预算测试；临时GPU前检模型不用于正式初始化。完整源码（包含训练、验证、整理、绘图）、配置和停止规则先提交，再进行前检与训练准入记录。正式训练期间不修改数值实现或超参数。

K20用Stage11B-1R独立CPU Generator和隔离原KMeans随机域：原设备float32相对阈值、10000有放回proposal、19聚类中心+softargmax、20完整轨迹，CWS关闭，逐目标稳定SHA种子，单进程串行。正式验证仅1/5/10/15/20/25/30；全部998预测与缓存完成后，才单独读取GT评分。非完整验证不可参与best；窗口加权minADE20最低、精确同值较早epoch为best，所有best指标配对同checkpoint。

模型共53155339参数，预测核心1641381；每epoch1882更新。训练/验证8/12、0.4秒，所有误差为原参考图像像素。ADE1/FDE1同第一完整候选，minADE20/minFDE20独立完整路径oracle；GoalFDE与解码路径终点分开。另报告goal-path差、候选两两距离及≤1像素终点近重复率，10/25/50像素GT终点覆盖率；只作几何描述，不将其当真实行为模态或语义碰撞。

GPU硬预算28800秒、安全阈值27000秒，包括整个GPU进程启动、加载、GPU前检、训练、验证与checkpoint。先复核30epoch+7次完整验证预估，超预算则不启动。运行前预留下个epoch及到期完整验证成本，逐batch/目标守卫；遇非有限值、身份/采样/恢复失败或OOM立即停止，保留边界和失败状态，不自动重启。初始与每轮边界不可变checkpoint全部保存，best/last通过清单引用，不覆盖旧权重。

科学图采用既定Python/matplotlib工作流：真实点、不平滑/补齐。曲线显示数据协议、版本、seed、checkpoint和K预算。best案例按Stage11B-1冻结的minADE20 .05/.10/.15低分位、.85/.90/.95高分位及minGoalFDE20 .85/.90/.95选择，同值sample_id稳定排序，跨组重复披露。RGB叠加全部20轨迹和目标点，不移动坐标、不删坏候选；GT只用于离线评分与绘图。输出PNG/SVG与坐标CSV。

现有语义ID映射、分割器精确训练成员、短池生成manifest和新验证语义曝光限制保留；不宣称未知场景泛化、官方论文测试复现或新创新，不计算语义合规率。预算结束后停止，不自动延长50/100/300epoch，不开发新模块。
