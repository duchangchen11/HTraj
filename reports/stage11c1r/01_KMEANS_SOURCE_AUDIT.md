# 原KMeans源码审计

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


本阶段为RESOURCE_PROFILING_ONLY。Stage11C-1正式训练仍0epoch、正式验证0次。
本阶段获准的1epoch仅标记DIAGNOSTIC_ONLY，不能直接作为原正式实验起点；未来复用必须另行批准及核验严格续训。
没有新ADE/FDE、覆盖正确率、真实未来终点分析或测试集成绩。验证诊断只读取物理隔离的8帧history-only缓存和旧语义缓存。
训练未来标签仅用于原始训练Loss。15个目标不是998目标全量验证，也不是独立统计样本。
记录范围审计：360次主路径预测、90次独立原KMeans、180次CPU诊断均实际核查caller RNG不变；
额外6次原Predictor仅核验输出逐位相同，未单独测其RNG，已标NOT_SEPARATELY_MEASURED。
原日志副本保留，未改变预测、时序或判定。
