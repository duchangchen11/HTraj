# 官方配置与checkpoint

冻结作者源码`a345ada1557f53dc2acb4c07f71d340d366980da`，仅拷贝到本阶段cache执行，源码SHA与Stage11A完全相同；MIT来源记录在frozen_source_manifest.json。没有修改第三方模型。

正式短期配置：obs8/pred12（notebook）、resize0.25、encoder[32,32,64,64,64]、decoder[64,64,64,32,32]、semantic_classes6、use_features_only=False、waypoints=[11]、temperature1、TTST=True/rel_threshold0.01、CWS=False。最后索引11就是目标点，**没有额外中间waypoint**。配置在official_sdd_config.json。

输入：参考彩色图按cv2的BGR顺序读入；官方分割预处理采用默认RGB/ImageNet参数但没有显式BGR→RGB换序，本轮保留这项公开代码行为，不能悄悄改色序后称原样复现。经ResNet101-Unet得到6通道softmax；再拼接8历史距离图，输入[batch,14,H,W]。Y-Net编码→12通道goal/waypoint热图；第11通道用于终点分布。TTST按原函数采10000点，19个kmeans中心加1个softargmax组成20个goal。给各层编码特征拼接goal距离图金字塔，原path decoder和softargmax输出20×B×12×2。推理无GT waypoint。

分割权重SHA `75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194`。预测完整state dict SHA `4be566ba49c386892dd729c9ea3c634a9ca6b37c206d015c0c293fccf796982e`。776键/形状与实际模型完全一致，`strict=True`，无缺失/额外键；**没有strict=False**。总参数53,155,339，其中分割51,513,958、轨迹核心1,641,381。这些是实际模块参数，不是“有效loss梯度参数量”，本阶段不计算loss/梯度。

原torch.save(module)包含旧式persistent module引用，weights_only先失败，独立白名单loader恢复分割对象，再将该对象装入原YNetTorch、严格载入完整state dict，等价于改变加载流程而不是模型前向。预测checkpoint同时含semantic_segmentation.*，最终值以整套预测checkpoint为准，不混接一个任意分割权重。

旧库源码警告共比较17类，12类文本改变，完整差异见legacy_library_source_diff.txt及CSV。没有隐藏Torch1.5→2.5/torchvision等变化；当前属于功能性兼容，不保证历史环境浮点数值逐位一致。

训练入口存在yaml batch8 vs notebook BATCH_SIZE4，原短期epoch还有26scene上限；不能把两配置混用或将300名义epoch称300完整遍历。unfreeze150控制分割前向调用位置；原train入口始终将分割参数requires_grad=False，不能仅按配置名字宣称150轮后真正解冻参数。未来方案必须预先明确这些语义，本轮未运行训练。
