# SDD短期行人预测协议草案

配置为`configs/stage11a/protocol_draft.json`，状态**DRAFT_NOT_FROZEN**，未获正式训练批准。选择可追溯的Y-Net短期SDD/TrajNet任务作为来源，但其确切目标与split文件未取得，所以不伪称已经对齐官方基准。

|项目|草案/未解决内容|
|---|---|
|时间|原30fps需视频或正式数据资料核实；12原帧一采样才推导2.5Hz=.4s，不能套ETH/UCY的10tick|
|8/12|8历史点首末2.8s（名义8采样槽3.2s）；未来从最后观测起4.8s；20点首末7.6s，原帧支持start..start+228，含229个原帧编号|
|原始采样|Y-Net按各连续track分段后每12行采样；必须确认连续frame增量为1，不能把删失后的相邻行当相邻帧|
|窗口stride|草案20个采样步不重叠；**不是声称官方短期原stride=20**，待取得目标manifest后决定复现/独立适配，不能混排|
|目标|仅Pedestrian，完整8/12原标注支持；其他类别不冒充预测目标|
|flags|lost行去除后按缺帧分段；occluded/generated保留标志，沿作者已有标注使用，不宣称全是人工关键帧；不得另造位置|
|邻居|Y-Net原场景热图为目标自身历史+静态场景，无显式社会邻居；PECNet社会pool资格另审，不能混为同输入|
|坐标|原参考图像素，显式每录像resize/padding，未验证米制scale；出图目标策略待审，不能默默裁剪/删除|
|身份/hash|source/video/person、全部20采样frame、history_SHA、GT_SHA、reference_SHA、semantic_SHA和transform_SHA；没有有效GT不编目标hash|
|重复与隔离|所有split不得有相同目标窗口或同目标交叠时域支持；按source/video/person报告相关窗口，不作iid统计|

## split尚未准入

官方train_SDD_trajnet.ipynb把`data/SDD/test_trajnet.pkl`作为VAL_DATA_PATH，再按最低val ADE保存权重。这是可见代码事实，不应把该test再次称为独立模型选择外的确认集。本轮未执行该notebook、未读取其test标签或成绩。

完整作者train/test身份清单及独立validation尚无。protocol里的train/validation/test均为null，不能填写凭空的录像名单。以后应先取原官方train/test清单，保留test封存；只在来源train内采用已有可追溯validation或预先声明的录像隔离开发拆分，审核后一次冻结，并承认这与原作者选模流程不同。若需时间分块，必须以20点完整支持排重并留≥完整支持长度隔离带，不能只比起点。

官网8个地点与论文使用的20个“scene”不是已核验同一粒度；sceneId常对应录像。缺少清单时无法判定哪些背景跨split共享，状态UNVERIFIED，不能把video隔离称未知物理场景泛化。取得清单后应同时按site与video列交集，必要时比较reference指纹；重复静态背景不等于目标窗口GT重复，但泛化结论范围不同。

六个镜像样例仅数据格式/坐标审计，不定义训练/开发/测试，不用于模型选型的误差。没有把它们拼成一套“官方训练集”。指标口径和oracle边界另见08_METRIC_SPECIFICATION.md。协议草案经科研审核及资源补齐前，不冻结正式训练。
