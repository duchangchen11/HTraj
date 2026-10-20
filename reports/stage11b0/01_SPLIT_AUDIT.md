# 数据身份与时域隔离审核

**SPLIT_IDENTITY_PASS**，不是FULL_PROTOCOL_READY。沿用Stage11A+不重选：train7,496目标/25录像；val998/5，固定bookstore_2、deathCircle_3、gates_7、hyang_7、nexus_4；test2,829/17。

源manifest SHA `71d36755136cd8abf317b7077e3de9899ac03ad0f519c5a1490b0e3b36afbb12`。原作者train.pkl内8,494个目标的history/GT/reference SHA重新核对；每个目标20行、frame差12、单个scene/track，窗口为first..first+228原frame。train/val/test录像身份和窗口ID互斥，按录像定义的全部20帧支持不会跨集合。test.pkl仅作opaque SHA核对，未反序列化test未来，不做预测。

训练专用副本只含7,496目标/25录像，来自已核验作者train池，行值/排序不变。GPU入口只加载该副本，不打开含预留val的原train.pkl，更不打开test.pkl。对应哈希在training_data_artifact.json，实际访问ID在actual_target_visits.csv。

两个已知出图点都在nexus_0/track125，frame2904和2916，y1950.5/1956.5超过原H1947。原值保留，按作者0.25与round得到y488/489，在补边后H512范围内；没有clip、过滤或改GT。原图外点的监督不代表真实图像内标签，未来评价仍保留。out_of_image_points.csv可复查。

限制：同地点多录像共享背景；原train/test共享coupa/gates/hyang/nexus，新的train/val也不是未知地点。原始视频全局时间/自然人跨录像身份无对应证据；只证明录像ID级隔离，不能把该结果扩大为完全独立自然人确认。作者短期TrajNet选窗的原始生成manifest不全，原pkl没有类别/lost/occluded/generated，未伪造这些字段。

数据来源绑定作者Drive ZIP及SHA，仍不是完整原始SDD。未按误差删除任何目标；本轮训练实际完成范围由training_execution.json记录。语义不确定性不妨碍用户本轮明确授权的train-only工程测试，仍阻止完整科研协议自动冻结。
