# 训练代码、初始化与工程入口

作者版本`a345ada1557f53dc2acb4c07f71d340d366980da`，源文件复制到新cache后逐SHA核验，不修改原源码。源码MIT、现有HTraj安全loader/数据adapter接口已核查，来源写入新脚本。环境继续使用隔离Stage11A+ venv（Torch2.5.1+cu124/SMP0.1.0），不装新库或降级旧环境。

作者论文/补充与YAML为batch8，短期notebook实际传batch4；本轮明确选**scene batch1、target microbatch4（notebook值）**，首个smoke target1，其后4/4。scene指一个录像下的轨迹集合，不是同时可见人数。没有把整个scene的几百目标当一个optimizer batch，也未按拥挤程度删样本。

原短期train.py在batch>25后break，增强后只访问26scene；原25录像池有200增强scene，作者30录像池则240，不能称完整epoch。本轮只做**无增强完整25scene一遍**，另用CPU核查8x增强访问数，未运行增强GPU epoch。

固定seed17、Adam lr1e-4、FP32、无新loss/clip/AMP/调度器。仅个体历史/静态场景；卷积编码器及goal/path解码器沿用原结构，没有新增模块。原YNet核心随机初始化，未读取作者sdd_trajnet预测权重；仅加载作者分割权重。3个smoke更新后重置到同一随机核心和空Adam，再开始完整测量，不能把smoke访问算进一次无重复遍历。

原损失严格为L=1000*mean_BCE(goal12,GT_gaussian12)+1000*mean_BCE(path12,GT_gaussian12)。所有像素/time/目标取mean，两个分支权重1；GT Gaussian沿用normalize=False、kernlen31、nsig4。path训练按GT第11索引目标距离图teacher forcing，合法训练监督，不是部署泄漏。推理函数没有future参数，改future后原history相同、预测相同；本轮不算ADE/FDE。

训练adapter唯一明确的运行状态修正：net.train后始终segmentation.eval且requires_grad=False；在no_grad下每录像计算一次分割特征。原作者在早期eval计算后切回train，晚期>=unfreeze会在train模式再调用，冻结参数不等于BN冻结；本轮不沿用该隐式BN漂移，也不冒充完全相同训练日程。

checkpoint含模型、Adam一/二阶矩、CPU/CUDA/NumPy/Python RNG、进度和配置；原方法无scheduler，记录None和原因，不丢失一个实际存在的调度器。本阶段无best选择，只有smoke与完整遍历last；恢复后相同下一步参数和Adam状态严格一致=True。不覆盖作者权重或旧checkpoint。

代码/配置训练前冻结commit `6166eb7e220ea2fbdb9394b50e1ea7714057d504`；运行源码版本及数据SHA均写日志。
