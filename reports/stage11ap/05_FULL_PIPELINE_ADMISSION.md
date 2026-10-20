# 真实训练数据完整推理准入

结果：**REAL_TRAIN_INPUT_PIPELINE_PASS**。不称OFFICIAL_REPRODUCTION_PASS，不是新算法，不是正式性能复现。

实际运行六个合法原train录像，各取一个真实20点目标，但预测器只接收前8点；未使用候选validation的5个录像，也未预测任何test目标。调用链为原RGB→实际分割权重→历史距离图→原编码器→原goal热图→TTST20goal→原path decoder20条12帧轨迹。CWS关闭，Ka=1，Ke=20，10000是proposal数量，不冒充最终候选数。

六例均输出[20,1,12,2]原像素坐标，分割[1,6,H,W]概率和与有限值检查通过。best-goal或future没有从GT选取；原evaluate函数包含GT/误差计算，本阶段故意不调用，独立推理wrapper仅调用其原推理算子。每例真实预测点在forward_*_predicted_paths.csv，无ADE/FDE、无候选排名。

固定同一训练目标历史，将未来12点替换成极大数，重置相同torch/NumPy seed后再次TTST推理，goal及全部K20轨迹逐值相同，最大差0.0。是API输入隔离证据，不证明未审查的所有官方入口都安全。原训练GT teacher forcing与本轮推理严格分开。

所有参数和buffer在推理前后逐值相同；inference_mode、eval、optimizer.step=0、loss计算0、test评价0。checkpoint完整键形状与严格加载已验证。详细状态full_pipeline_admission.json、real_train_future_isolation.json及stdout保留全部源变更/meshgrid警告。

工程适配：SMP0.1.0在新venv安装通过；逐scene float32数组替代NumPy隐式ragged array；batch1 scene collate显式as_tensor；原旋转/翻转调用保留、Pandas append换独立concat适配器。官方源不变，未改模型forward/loss。原26scene cap也不删；提供明确无cap的独立数据迭代入口供未来方案选择，**未建立或运行新训练runner**。timm0.1.20因torch._six缺失仍导入失败，但已确认本ResNet101分割不走timm分支；不安装其他骨干/不盲目修复无关分支。

完整单元测试范围说明：用户要求运行旧完整测试套件；其中既有frozen-ego/AgentFormer/NoSocial容量等测试会在临时人工模型上调用backward/optimizer.step以验证恢复或梯度。这不是Y-Net训练实验，不使用真实数据训练、不改旧权重。上述Y-Net入口的loss/backward/optimizer计数均为0，不能把它扩大成整个pytest进程没有任何优化器调用。原测试源码未修改。
