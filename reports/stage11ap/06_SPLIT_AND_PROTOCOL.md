# 作者协议与HTraj开发候选

作者train_trajnet.pkl：30录像、8,494目标、169,880行。test_trajnet.pkl：17录像、2,829目标、56,580行。不是全原始SDD60录像。逐metaId检查20行、单录像/track、有限坐标、严格frame差12通过；没有凭名字把test认作val。官方notebook却确实用test选最低ADE，本轮拒绝该入口，不计算测试误差。

作者train/test录像交集、完整窗口身份交集均0。因此同录像同track的20帧支持不会跨两个split；track数值跨录像重复不代表同一人。站点背景交集为coupa/gates/hyang/nexus，不能称未知场景泛化。跨录像是否跟踪同一自然人没有标注，不能声称已全部身份去重。

每个(scene,trackId)在短期包里只有一段20帧，没有可观测相邻窗口stride，**不能据此编造stride20**。作者原短期TrajNet选窗/原标注到pkl的完整链缺失；原预处理源可查但不是这些pkl实际生成参数的证明。所有输入和GT按sample_id/source(scene)/person/frame/history_SHA/GT_SHA记录在author_target_manifest.csv。训练池有2个出图坐标点，保留，未clip或删难例；未来loader边界处理需声明。

独立开发候选：仅在作者训练录像池内，每个至少2录像的地点，以固定SHA256('HTraj-SDD-dev-v1:'+sceneId)最小者作val。仅有coupa_3的地点留在train。固定val为bookstore_2, deathCircle_3, gates_7, hyang_7, nexus_4。得到train7,496目标/25录像、val998目标/5录像、sealed test2,829目标/17录像。录像隔离使完整228原frame时间支持不跨新split；不需用时间块隔离带弥补本已不同的录像。仍非未知地点；未根据预测误差调整划分。

已下载预测权重可能训练过这5个新val录像，也可能用官方test选模；**不能用它报告独立开发泛化成绩**。新开发协议只能用于之后单独批准的从头训练方案；作者原train/test清单单独保留，不混称论文数值复现。

时间：30fps依据作者downsample文档，包内帧差12→2.5Hz/0.4s；没有下载原视频实测fps。8历史首末2.8s，未来末观测至终点4.8s，20帧首末7.6s。像素单位，无米制假设。原pkl缺少lost/occluded/generated/label，不伪造这些字段或把缺失当0；数据状态与来源局限见protocol_candidate.json。

**NOT_READY**：数值身份/隔离通过，但SDD类别名称和ignore规则、分割预训练标签曝光、原短期目标选择/stride来源仍待核准，新训练日程也未冻结。正式协议不能由一个成功forward自动批准。默认test封存、不开训练、不开验证排名。
