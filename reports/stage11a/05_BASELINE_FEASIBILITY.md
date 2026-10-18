# Y-Net/PECNet基线准入

两个模型来自同一作者仓库固定`a345ada1557f53dc2acb4c07f71d340d366980da`，MIT源码；数据/图像适用SDD独立许可。源码逐Git blob/SHA核对，未修改第三方文件，未加载Y-Net权重，也没有执行真实数据预测。

|项目|Y-Net|PECNet|
|---|---|---|
|输入|8帧自身坐标热图+RGB经分割得到的场景语义/特征|历史坐标、其他参与者位置/社会pool；无RGB或语义图|
|时序/场景编码|时间作为8个热图通道，UNet型空间编码；非GRU/CVAE|past MLP、endpoint CVAE、非局部社会MLP池化|
|目标与路径|12步goal/waypoint热图；多尺度条件路径解码器|采样CVAE终点，再预测其余11步并拼接终点|
|多模态|Ke个goal×Ka条路径；TTST先抽10000再聚类，CWS条件waypoint|latent采样及truncation trick；20终点不是Y-Net同算法采样|
|loss|goal/future热图BCE+GT waypoint条件下路径BCE|KL+endpoint重建+trajectory重建|
|训练阶段|分割预训练/微调资源+轨迹网络；不能只算轨迹参数|主要一个联合CVAE/社会预测训练流程|
|外部资源|作者config、分割完整module、预测state_dict、SDD图像/划分|源码/3份作者权重Git树可见；池化数据来源/身份需重建|
|原评价差异|minADE对完整paths；代码FDE使用waypoint采样目标，不一定decoded-path末点|GT先挑best endpoint后仅解码相应路径，非全部K paths独立minADE|

## 当前真实检查

Python3.10.21、Torch2.5.1+cu124、NumPy1.26.4、Pandas2.2.3、OpenCV4.10.0.84。作者原环境为Python3.8.3/Torch1.5.1、SMP0.1.0；当前缺segmentation_models_pytorch及其若干旧依赖。CPU import成功；使用作者原生`segmentation_model_fp=None`分支、工程配置6通道语义输入/32-32-64-64-64编码器，B1/B2、64×96合成编码/goal/path/softargmax均有限。**这不是实际分割网络或作者精确配置复现**。

该工程核心参数1,641,381，不含分割模型，不能代替完整正式Y-Net参数量。PECNet旧Stage7同commit合成N1/2/5/12准入及2,099,692参数仅冻结引用，未新增真实性能。完整Y-Net参数和训练耗时仍未知。

真实失败：①分割依赖ModuleNotFoundError；②原SceneDataset在不同人数场景构成ragged array时，NumPy1.26报ValueError。另静态发现augment_data依赖Pandas2.2已没有的DataFrame.append、raw loader的header=0、SDD8帧训练外层在26场景batch后break。`num_epochs`不能直接当全部目标的完整epoch。分割参数虽requires_grad=False，train/eval模式还可能更新BN统计，须取得实际module后审查。失败记录在baseline_probe.json，不用核心forward成功掩盖完整流程未通。

最小兼容建议：建立独立环境，核对旧SMP/torchvision/torch组合或只在独立数据适配层处理ragged列表和pd.concat，明确diff与数值等价测试；不修改冻结作者Transformer/网络/loss，也不在现环境全局降级。尚未实际应用这些修复，未声称它们已足够。

RTX3080对工程核心合成前向可运行，未包括分割、梯度、TTST/CWS或正式大图训练。GPU记录见gpu_probe.json；不根据冷启动/三组forward估算完整epoch。正式训练GPU小时未知，不能把旧AgentFormer/SocialVAE耗时套用Y-Net。后续如资源补齐，建议先≤1GPU小时工程准入上限，测一轮完整epoch后另行审批训练预算。

结论：Y-Net值得作为**有条件的复现准备对象**，现在未具备正式复现条件。PECNet保留终点条件参考，不将不同输入/容量/评价的二者成绩当语义独立贡献。暂不启动任何新网络或正式训练。
