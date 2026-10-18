# 结构验证与真实检查记录

最终全仓库测试命令及JUnit：`results/tables/stage10a/final_unit_tests_stdout.txt`、`final_unit_tests.xml`。旧测试458项与本阶段43项共501项通过。此前497项通过后补充同向/相向/交叉及动态入半径的4个明确测试，再运行全套。没有以结构通过冒充预测收益。

|要求|检查与证据|
|---|---|
|排列等变与梯度|E/S/D/R，N=1/2/5/12，输入/图随ID同步换序并逆换输出；float64预测及输入梯度atol1e-10/1e-9；同分按稳定ID确定|
|平移等变|全局+[8,-11]m，预测相应平移；原点无绝对坐标编码|
|padding/mask|padding坐标NaN由mask在运算前清除；不同padding宽度不影响有效预测；无效输出0；缺失与重入carry重置|
|零/单邻居与密集|N1、N2，GPU B8/N79无裁剪；无消息s严格0，与同权重E逐位相同|
|未知/静止速度|连续静止valid=True；首次观测velocity_known=False，两端已知才用相对速度；位置-only合法，missing标记显式|
|前缀因果|完整8帧前4步图等于只输入4帧；改后4帧不影响前4步图和encoder；raw帧查询Guard禁止窗口外访问|
|未来GT隔离|人工原始数据改变全部未来，图、输入、随机权重预测逐位相同；HOTEL/ZARA2真实原始记录反事实图与输入逐位相同，未计算验证预测或误差|
|DCA/tCPA独立核对|100对独立NumPy标量实现，包括未知、零/极小速度；与Torch tau/DCA/q一致atol1e-12|
|人工运动|同向相对速度0→tau0；相向和交叉同速到达→tau1s、DCA0；进入3m后动态图新增边，静态不追认|
|R/D不同选择|固定末点：target0；近邻1m远离、另一邻居3m以1m/s接近；K1结构fixture下D选ID20，R选ID30；见synthetic_edge_witness.json。实验K8另由真实训练图统计验证|
|确定性/排序|重复建图逐位相同；shuffle存储顺序、保持ID关联，恢复后风险排序/边相同|
|梯度/跨scene|合成forward/backward无NaN/Inf；E其他行人Jacobian=0；D合法邻居梯度非零；不同batch scene无交叉依赖|
|恢复|临时合成state_dict strict保存/加载，恢复预测逐位相同，未使用旧权重|

预固定的64个来源训练窗口（每fold均匀取32，可能跨fold重叠）中D/R都有59,924条时空有向边，2,909条D边被R替换，约4.85%；23/64窗口有差异。全部窗口逐接收者预算、候选mask、消息特征一致。其余窗口可能度数≤K或排序一致，不将相同图伪称无机制。记录在train_graph_selections.csv、selection_summary.json，不使用GT误差挑反例。

首次结构检查10失败/23通过：9项由PyTorch stable argsort要求dim为关键字引起，1项为人工风险例的sigma=1期望错误；第二次仅后者失败。修复API参数，并让人工例显式使用已冻结的sigma=.5，未修改风险公式或数据阈值。风险排序并不保证所有“接近者”都压过当前近邻，指数时间/距离权衡须按公式计算，不能先写期望再调验证参数。失败文本及最终通过记录全部保留；为git空白检查对文本尾空格规范化，原始gzip和SHA另存。

Stage10A GPU smoke仅作合成平方输出的反向，优化器更新0，权重前后逐位一致；不是正式预测loss或小规模训练。全仓库旧单元测试只为回归验证，不据其人工输出形成新性能指标。
