# Stage11A+总报告

**工程进展：正式checkpoint严格加载与真实训练输入完整推理通过。总体NO-GO：当前协议NOT_READY，不启动正式训练；SCI创新性也未获准。**

1. 作者指定资源下载成功：995,391,402字节，SHA `3e4aa09c900c729711c37eb6e82f0eedf354798f70d67bbd3d6b444ba62fbc11`，ZIP/CRC/路径与空间检查通过；未下载69GB原视频。
2. 获得作者短期SDD池（8,494 train + 2,829 test目标）、47 RGB、60 masks、3 configs及5模型权重；原SDD完整视频/annotations仍不全。不能把子集称全原始数据。
3. 真实SDD单通道mask存在，ID1..5；分割模型6通道0..5。SDD ID名称/ignore无专用权威表，inD README仅作线索，不能假定第六类background。分割预训练用了哪些test/val标签未知。
4. 六个训练案例RGB/mask尺寸同网格、真实历史与train未来离线叠加已保存；图像边界粗粒度，原视频精确配准未证，不使用米制误差。
5. 776键/形状严格加载；参数53,155,339（分割51,513,958、预测核心1,641,381）。没有strict=False、没有修改冻结模型。
6. 真实输入完整链六例通过：[20,1,12,2]，TTST10000→20goal，CWS关，所有输入仅历史/RGB、无GT oracle，未来反事实差0、权重不变。无ADE/FDE、无训练。
7. 原包无独立val，notebook用test选模不能照搬。作者train/test录像不重合，地点共享4处。生成唯一source-train录像级候选7,496 train/998 val，原test2,829封存；只身份/支持核验，没有验证排名，协议保留NOT_READY。
8. 独立环境解决SMP、ragged、append所需数据兼容；旧式模块安全加载有专用受限fallback，保留旧库源码变更警告。timm未用分支仍不兼容；旧Torch数值等价未证明。训练26scene早停未静默删除，未来日程需单列。
9. CINet作者声明源码新增静态核查；McTrans/GDS正式接口只取到元数据，三篇完整方法仍有缺口。此前查重限制继续，不把工程适配称创新。
10. 实测GPU两个进程共9.912s，allocated峰403.81MiB；正式训练GPU小时未知，不能由B1推理计时填入训练预算。

Q1下载成功：是。Q2完整原始SDD：否，作者短期任务资源齐，但原生成来源部分未验证。Q3语义mask：有。Q4类别/坐标：同网格通过，名称/ignore和原视频精配准未全明。Q5严格加载：通过。Q6真实完整前向：通过。Q7正式split冻结：未通过。Q8是否仍有泄漏风险：原test选模、分割预训练标签曝光，以及将已有预测权重评估新val的风险均未被当作合法评价；本轮没有实际执行这些泄漏路径。Q9文献是否覆盖：广泛相关机制已覆盖，三篇完整细节仍未核实，不能宣布空白。Q10值得正式复现：Y-Net可条件性继续资源/协议准备，当前不批准正式训练。

下一步仅建议核实SDD类别与分割训练来源、审核source-train-only开发划分和训练预算定义，然后单独决定是否批准完整epoch测量。工程PASS≠论文性能复现PASS≠创新GO。

所有证据在results/tables/stage11ap；测试和历史SHA以full_unit_tests.xml、history_after.json、final_checks.json为准。分支feat/stage11ap-ynet-resource-admission，基于36e76a3；最终commit以Git交付为准。完成后停止，不进入正式训练。

完整单元测试范围说明：用户要求运行旧完整测试套件；其中既有frozen-ego/AgentFormer/NoSocial容量等测试会在临时人工模型上调用backward/optimizer.step以验证恢复或梯度。这不是Y-Net训练实验，不使用真实数据训练、不改旧权重。上述Y-Net入口的loss/backward/optimizer计数均为0，不能把它扩大成整个pytest进程没有任何优化器调用。原测试源码未修改。
