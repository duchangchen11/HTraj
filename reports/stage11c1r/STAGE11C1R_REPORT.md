# Stage11C-1R：无语义训练成本与KMeans预算审计

- ENGINEERING_AUDIT: **GO**
- KMEANS_ROOT_CAUSE: **PARTIALLY_SUPPORTED**
- BUDGET_ADMISSION: **UNCERTAIN**
- PROTOCOL_COMPATIBILITY: **PRESERVED**
- FORMAL_TRAINING: **NOT_AUTHORIZED**

本阶段为RESOURCE_PROFILING_ONLY。Stage11C-1正式训练仍0epoch、正式验证0次。
本阶段获准的1epoch仅标记DIAGNOSTIC_ONLY，不能直接作为原正式实验起点；未来复用必须另行批准及核验严格续训。
没有新ADE/FDE、覆盖正确率、真实未来终点分析或测试集成绩。验证诊断只读取物理隔离的8帧history-only缓存和旧语义缓存。
训练未来标签仅用于原始训练Loss。15个目标不是998目标全量验证，也不是独立统计样本。
记录范围审计：360次主路径预测、90次独立原KMeans、180次CPU诊断均实际核查caller RNG不变；
额外6次原Predictor仅核验输出逐位相同，未单独测其RNG，已标NOT_SEPARATELY_MEASURED。
原日志副本保留，未改变预测、时序或判定。


|状态|固定目标数|迭代中位数|迭代范围|KMeans中位秒|每迭代中位毫秒|完整K20中位秒|
|---|---:|---:|---|---:|---:|---:|
|NoSemantic-initial|15|76|35–117|0.410104|5.3198|0.555366|
|NoSemantic-epoch001|15|21|11–37|0.111340|5.3412|0.273991|
|Semantic-epoch001|15|34|2–72|0.181259|5.3500|0.348380|
|Semantic-epoch005|15|28|12–104|0.148246|5.2945|0.313984|
|Semantic-epoch010|15|40|28–81|0.216575|5.3220|0.383619|
|Semantic-epoch030|15|36|18–60|0.191646|5.3228|0.354524|

原作者`utils/kmeans.py`固定在a345ada1557f53dc2acb4c07f71d340d366980da：

- 第50–58行：X.float()后.to(device)，初始化中心来自同设备X。
- 第43–44、73行：Euclidean分支调用pairwise_distance_function(X,initial_state)，未传第三个device。
- 第145–158行：pairwise_distance默认CPU，两个参数.to(cpu)，距离矩阵CPU；第75行argmin的choice_cluster也在CPU。
- 第79–84行：CPU选择结果索引CUDA X；selected.mean与center赋值在CUDA。隐式索引传输的具体实现和字节数未测。
- 第86–105行：CUDA center_shift参与Python停止条件，存在标量读取/同步路径；平方位移阈值0.001优先于1000次上限。
- 第16–17行：NumPy无放回初始化；空簇分支第81–82行使用CPU torch.randint，再在原设备X索引。

每轮明确的D2H有效载荷估算=(10000×2+19×2)×4=80,152字节；这是静态有效载荷核算，
不是硬件传输量或带宽实测。CPU mask/index的隐式H2D、内核启动、选择和归约同步未被定量拆分。

固定数学语义（未修改）：

\[
q_i=\arg\min_c\sum_d(x_{id}-\mu_{cd})^2,\qquad
\mu_c=\frac{1}{|I_c|}\sum_{i\in I_c}x_i,\quad I_c=\{i:q_i=c\}.
\]

空簇使用原CPU `torch.randint` 选一个采样点；收敛条件为

\[
\left(\sum_{c=1}^{19}\|\mu_c^{new}-\mu_c^{old}\|_2\right)^2<0.001.
\]

随后才检查1000次上限。19中心用NumPy无放回初始化，TTST10000点仍为原CPU Generator有放回采样。


本轮观察到迭代范围2–117次，
未监测原KMeans单迭代中位5.3348ms。
迭代数与wall的描述性Pearson=0.9997073793755661（不作显著性检验，不等于因果分解）。
如果执行1000次，以本轮每迭代中位成本投影约5.3348s，
与旧约5.2秒量级相容；但这是条件外推，旧临时预测状态没有迭代trace，**不能据此追认旧样本实际达到1000次**。
旧C1的2个训练目标、临时多步更新模型，与本轮15个验证历史及随机初始/Epoch1并非同一配对实验。
CPU常驻诊断中心逐位相同1/90，
最大中心坐标差3.05175781e-05网络坐标像素。
CPU路径和原路径使用完全相同的采样点及初始化种子，但归约、设备访问、内核调用和同步共同改变。
所以设备混合路径及其成本风险有源码和运行证据，**传输具体贡献多少秒、占比多少仍UNVERIFIED**。
不能把两条路径耗时之差全部算作传输。


诊断训练循环实测462.09766270499676秒。

|情景|未来完整运行秒|未来小时|与当前剩余额差距秒|假设|
|---|---:|---:|---:|---|
|A_HISTORICAL|16352.19|4.542|0.00|Semantic timings only; no guarantee for new NoSemantic distributions|
|B_POST_EPOCH1|17613.24|4.893|369.86|All7later validations use postdiagnosticEpoch1 mean; later checkpoint behavior unmeasured|
|C_SLOW_STRESS|22352.89|6.209|5109.51|All7validations retain slower initial/epoch1 per-sceneP90; conditional stress, not empirical runtime bound|

情景A来自旧语义完整30轮和7次全量验证的真实记录，迁移到无语义模型仍是假设。
情景B以本轮完整epoch耗时外推30轮，用epoch1后每录像3个固定目标的普通完整预测耗时均值，
按真实录像目标数投影998个目标和7次验证，另计重复检查、旧评分/保存开销、31个checkpoint、启动、异常和退出预留，增加10%时间裕度。
情景C对每录像取NoSemantic初始与epoch1的较慢P90，假设7次验证均保持该慢速，增加20%裕度。
这是条件性压力情景，不是总体统计上界，也不能假定实际epoch30仍为初始速度。
没有对未测epoch做插值；所有30轮和998目标的新增成本均是外推，未执行。
当前剩余额已扣除Stage11C-1和本轮全部GPU相关进程wall；诊断checkpoint未获准复用，因此成本从头计算。


原C1已用66.674948s；本轮GPU相关进程689.949887s；合计756.624836s；剩余17243.375164s。失败进程若有也计入，无另扣CPU反事实时间。

完整pytest：695通过，1跳过，0失败，0错误；原警告和SKIP保留。

历史95751文件SHA复核passed=True。

**Q1 旧停止是否主要因早期成本估计偏高？** 旧拒绝确实使用临时早期模型的约5.2秒聚类投影。当前随机初始/epoch1 KMeans中位分别0.4101/0.1113秒；这检验了早期估计的代表性，但不能把变化单因归于语义有无。

**Q2 实际迭代次数？** 全部固定状态/目标范围2–117；逐样本见kmeans_iteration_counts.csv。

**Q3 有设备切换成本？** 明确存在CUDA输入转CPU距离、CPU指派与CUDA更新的混合路径，见源码与设备事件。

**Q4 切换成本已量化？** 仅明确D2H有效载荷和两条整体路径wall；切换秒数、占比UNVERIFIED。

**Q5 实际一epoch多久？** 训练循环462.097663秒，完整GPU父进程另含构造、保存、恢复与退出。

**Q6 初始与epoch1成本变化？** 初始/epoch1每15目标原KMeans总计5.9176/1.8619秒；14/15变快，1/15变慢，每ID配对保存，未删除退化样本。

**Q7 原语义epoch1/5/10/30有变化？** 见相同15身份的逐样本及中位/P90表；不以不同样本组冒充配对趋势。

**Q8 有可靠完整预算？** 具备完整epoch和配对采样实测锚点，但以后epoch及全998成本仍外推，三情景不是完整运行证据。

**Q9 5小时够吗？** 预算准入=UNCERTAIN，不能用情景A或B最乐观结果保证安全完成。

**Q10 额外预算多少？** 建议提交用户审查的累计预算为23400秒（6.50小时），比原18000秒增加5400秒；它覆盖本轮条件压力情景及已用预算，不是实际耗时保证，也不构成正式训练授权。

**Q11 有严格等价工程加速？** 存在I/O与加载的候选机会，尚未验证完整998等价；CPU迁移不能直接认定等价。

**Q12 需改原数值协议？** 本轮不需要且没有修改。预算不足应交用户审查，不通过换KMeans、减少K或跳验证解决。

**Q13 具备正式准入？** FORMAL_TRAINING=NOT_AUTHORIZED；本阶段不能自动启动。

**Q14 阻断条件？** 完整30轮+7验证未实测；后续概率分布和聚类成本未获保证，预算情景及授权需审查；不能擅用诊断checkpoint续训。

**Q15 下一步唯一建议？** 用户审查原样完整训练的预算与明确授权；保持数值协议，不自动替换聚类、不训练其他seed。
