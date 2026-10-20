# 实际资源清单与状态

AVAILABLE表示已经取得并检查；PARTIAL表示不全；UNVERIFIED表示不能确定某项含义/来源；UNAVAILABLE表示本地未取得，非断言永久无法获取。

|资源|状态与真实内容|
|---|---|
|作者短期SDD轨迹池|AVAILABLE：train_trajnet.pkl 8,494目标，test_trajnet.pkl 2,829目标；每个目标20行|
|SDD全部原始数据|PARTIAL：没有原视频、完整原annotations；不把作者47录像预处理子集冒充60录像原始全集|
|短期参考RGB|AVAILABLE：train30录像、test17录像；47张均覆盖短期目标|
|SDD语义掩码|AVAILABLE：train_masks43、test_masks17，共60张；SDD专用ID→名称/ignore映射UNVERIFIED|
|独立validation|UNAVAILABLE：原包没有val.pkl或val目录；test不是合法开发验证集|
|模型配置|AVAILABLE：sdd_trajnet、sdd_longterm、inD_longterm三个yaml；本阶段只用短期SDD|
|SDD分割权重|AVAILABLE：SDD_segmentation.pth，完整旧式Unet模块|
|Y-Net预测权重|AVAILABLE：sdd_trajnet_weights.pt，776状态键，严格加载通过|
|其他文件|inD权重/4个语义图与ETH/UCY数据仅作目录/文件SHA清点，未加载模型、未评价轨迹|
|SDD palette/class README|UNAVAILABLE：SDD mask均L模式无palette；只有inD自己的README类别表|
|预处理脚本|冻结官方Git源码AVAILABLE；资源包本身没有脚本或原轨迹生成manifest|

全部成员、字节、图像模式、尺寸、SHA见resource_inventory.json及archive_members.csv。pkl实际列为frame/trackId/x/y/sceneId/metaId，不包含类别、lost、generated、occluded；不能凭文件名或删除字段推断逐条原始来源都已验证。

数据在新venv用精确numpy/pandas构造函数白名单反序列化，未知GLOBAL拒绝；没有运行任意pickle callable。预测state dict用weights_only=True。完整旧分割模块的weights_only=True因旧module persistent ID失败；随后仅对已验证作者SHA采用独立受限Unpickler，而非无条件pickle fallback。所有来源和旧库警告保留。
