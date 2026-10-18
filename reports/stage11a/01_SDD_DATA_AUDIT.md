# Stage11A：Stanford Drone Dataset可用性

**当前不具备完整可训练SDD语义数据。** 根目录与origin Stage10A均为521c53a，工作区原本干净，新分支`feat/stage11a-sdd-semantic-goal-audit`。Stage1–10A按SHA保护，不使用Mamba-T等旧项目的模型或结果。核查边界、分类与逐资源状态在`results/tables/stage11a/resource_inventory.json`。

[Stanford官方页面](https://cvgl.stanford.edu/projects/uav_data/)说明8个物理地点、60段录像，2016版，6类为Pedestrian、Bicyclist/Biker、Skateboarder/Skater、Cart、Car、Bus。各地点录像数为bookstore7、coupa4、deathCircle5、gates9、hyang15、little4、nexus12、quad4。官网完整视频包约69GB；本轮未下载。官网声明数据CC BY-NC-SA3.0，不能用第三方镜像的软件许可证替代。正式论文为Robicquet等ECCV2016，不混用早期镜像README的投稿题名。

|资源|当前状态|证据和限制|
|---|---|---|
|原始视频|UNAVAILABLE|本项目和已检查常见数据目录未发现；原下载端点超时；没有ffprobe实测帧率/帧尺寸|
|原始格式annotation|PARTIAL；官方来源一致性UNVERIFIED|镜像6录像：5份最多4MiB前缀、little/video0完整文件；不是60录像训练集|
|reference.jpg|PARTIAL；官方来源一致性UNVERIFIED|6幅真实RGB参考图，各自完整镜像Git blob核验，未核对Stanford原包|
|Y-Net预处理轨迹/划分/config|UNAVAILABLE|作者补充包网页存在，未取得包内容；目录或文件名不等于可用数据|
|PECNet处理轨迹|UNVERIFIED|官方Git树列有pickle池；没有解包/反序列化或核对目标，test池未读取，不接受为本轮有效数据|
|手工场景语义图/类别palette|UNAVAILABLE|论文描述不是取得文件的证据；不能把RGB、背景或灰度当标签|
|分割/预测权重|UNAVAILABLE（Y-Net）|补充包内文件未取得，未严格加载，未宣称完整复现|
|米制scale/homography|UNAVAILABLE|未得到经核验的每录像标定，也没有使用estimated_scales.yaml|
|作者模型源码|AVAILABLE|Human-Path-Prediction固定a345ada1557f53dc2acb4c07f71d340d366980da；小源码26项及MIT LICENSE逐Git blob核验|

本地搜索覆盖HTraj数据/实验目录、`/home/lrj/datasets`、data0109、Downloads和两个既有数据审计目录，排除Python环境/Git对象扫描；不是声称遍历全部磁盘。旧审计中gate/quad脚本命名的SDD假阳性已排除，没有把其他项目文件当数据。

为完成有限几何检查，从[固定第三方镜像](https://github.com/flclain/StanfordDroneDataset/tree/9179f3c0dd9ac0bd67a6223703d04a7577a9fb18)仅取六个video0的参考图和标注前缀，共约23.19MiB；上限42MiB，未下载视频或完整大型数据包。镜像blob匹配仅证明固定镜像版本，不提升为官方来源AVAILABLE。原始样例只在忽略cache，Git只提交必要派生图、少量历史点和hash。所有图显式标注UNVERIFIED来源。

源码查询有已记录副作用：初次partial Git的大小遍历触发惰性对象获取，代码对象缓存约157MiB；后续改成不查询blob大小的树清单及逐文件小源码下载。未加载其中二进制预测权重或反序列化SDD测试池。没有下载69GB原视频或作者完整大型补充包。

Stanford Drone Dataset与[TU Graz Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387)不是同一数据集。没有把后者的航空语义类别/掩码移植到SDD。SDD轨迹单位是原参考图像素；帧率30fps目前来自Y-Net官方预处理文档，未由本轮原视频metadata独立核验。作者短期任务为2.5Hz，不能凭ETH/UCY的frame编号规则代替时间标定。

数据获取失败、网络响应和下载上限分别见network_access.json、author_bundle_metadata.json、sdd_sample_download.json。正式训练前应取得作者原补充包或有可核验来源的SDD annotation/reference，核查大小/许可/必要性；大型下载需另获用户批准，确认页标示作者包约949M；当前未获大型资源下载批准，本轮不提交下载确认，不声称该资源永久不存在。
