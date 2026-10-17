# MoFlow技术审查

官方源码`f1b89b0e80ce95646214923e271e3c280d305cfa`，与Stage7.5已审版本相同；MIT许可。这里补齐完整train/eval/data/teacher/student源码核查，没有重复正式训练，没有执行test评价。

## 输入、输出与阶段

ETH原加载器提供B×A×8×6：绝对历史、相对最后观测点历史、相邻位移（不是已除0.4s的m/s），加K=20噪声未来与flow时间。teacher backbone输出B×20×A×24及分数；流采样多步积分，student以一次网络调用生成候选集合。ETHEncoder社交Transformer后还有agent-index位置编码和agent_order_embedding，关系可用性必须结合实际数据A判断。

训练链为：teacher flow matching → 在来源训练池离线生成teacher样本 → IMLE student训练。Y_t=(1−t)Y_0+tY_1；原目标不是普通MSE微调，而是K-shot回归/分类等。IMLE代码将student和teacher候选做双向最近集合距离，按agent聚合，可另加GT监督；这些都已是原论文机制。默认配置teacher150epochs、student150epochs，README teacher样本生成示例100采样步。不能把只跑一个teacher网络forward称为完整一步MoFlow复现。

## 数据和选模实质问题

[ETHDataset](https://github.com/DSL-Lab/MoFlow/blob/f1b89b0e80ce95646214923e271e3c280d305cfa/data/dataloader_eth_ucy.py) original分支将traj转换为[:,None,:,:]，因此A=1，虽然保留SocialTransformer类，也没有真实其他行人消息。LED分支为另一套数据，不能推定资格/样本与original相同。

原始转换器使用排序unique frame切20步；要求每目标覆盖完整窗口，且num_peds_considered>min_ped(default1)，排除仅一个合法行人的窗口，然后再将每条轨迹独立存储。原AgentFormer包含单人窗口，二者测试对象不等同。

主训练入口只构建train/test loaders；Trainer令val_loader默认等于test_loader，并按该验证分支的ADE选best。**不能把这个默认入口直接当合法validation选模**。本轮没有调用它。后续若重训必须由外部冻结入口显式传入独立validation；不能混入旧validation区域，不应修改目标对象后仍拿论文表格直接排名。

## 运行、权重与依赖

CPU实例化/teacher backbone人工前向通过。A1 teacher4,700,977参数，student4,086,320参数；A2 teacher4,701,233、student4,086,576。student只实例化计数，未做其forward/IMLE loss。A2同步置换历史、噪声和输出后最大差0.1739883，验证当前固定agent-index编码有顺序敏感性；人工随机权重不说明论文实际指标受损，A1场景更不受这个N2测试直接约束。

第一次人工输入漏past_traj_original_scale字段，报KeyError；补齐作者声明字段后原网络通过，失败现场保留。未修改模型。作者声明Python3.11/torch2.0.1/numpy2.2.4；本轮未安装其完整accelerate/EMA等trainer环境，只运行已记录的兼容CPU组件，不称完整训练入口已通过。

官方HuggingFace元数据可获取，含五折ETH/UCY **teacher** checkpoint及配置；本次返回清单未见对应ETH/UCY IMLE student，不能把NBA student文件当ETH权重。文件OID/LFS SHA保存于moflow_checkpoint_availability.json；未下载和strict加载MoFlow权重，不算性能复现。整个MoFlow GPU测试耗时为0。它在本轮“社会关系”为主的选题中面临A1数据、排列/协议与两阶段成本问题；纯准确性/效率选题也没有新的本地缺陷证据。
