"""Evidence-based reports; unavailable results remain unavailable, never zero."""
from common import *

def report(name,text):
    p=ROOT/'reports/stage11a'/name
    with p.open('x') as f:f.write(text.strip()+'\n')

def main():
    alignment=json.loads((OUT/'alignment_results.json').read_text())
    probe=json.loads((OUT/'baseline_probe.json').read_text())
    report('01_SDD_DATA_AUDIT.md',r'''
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
''')
    table='|scene/video|原图宽×高|已读标注行|lost=0中心越界/总数|历史track/frame|缩放往返最大误差px|\n|---|---|---|---|---|---|\n'
    for c in alignment['cases']:
        table+=f"|{c['scene']}/{c['video']}|{c['image_width']}×{c['image_height']}|{c['annotation_rows']}|{c['visible_centers_outside']}/{c['visible_centers']}|{c['track_id']}, {c['first_history_frame']}–{c['last_history_frame']}|{c['inverse_roundtrip_max_px']:.2e}|\n"
    report('02_COORDINATE_ALIGNMENT.md',r'''
# 原始标注、像素坐标和参考图对齐

结果为**PARTIAL几何核验，不是原视频精确配准或语义对齐已经证明**。六个真实录像样例均有对应参考图与原格式标注；镜像来源仍UNVERIFIED。完整过程见`scripts/stage11a/sdd_tools.py`和alignment_audit.py，逐文件SHA/矩阵/统计见alignment_results.json；无虚构轨迹。

## 字段与坐标定义

每行10列：track_id，xmin，ymin，xmax，ymax，frame，lost，occluded，generated，带引号label。track_id只在该录像内标识轨迹；同人跨录像身份不推断。lost=1表示出视野；occluded=1表示遮挡；generated=1是标注系统插值，而非本轮神经网络伪标签。坐标为参考图左上原点、x向右/y向下的像素，中心为((xmin+xmax)/2,(ymin+ymax)/2)。

新审计parser明确header=None，保留首行；作者Y-Net load_SDD用header=0且names，会把无头原文件首行丢掉，这是源码可定位差异。本轮不改作者loader，也不声称新parser就是官方精确预处理复现。

## 真实覆盖

'''+table+r'''
各图位于`results/tables/stage11a/alignment_figures/`，包含原图、缩放图和原像素局部crop，标注scene/video/frame/track、原图尺寸、坐标单位及变换。初版短位移轨迹被帧标号覆盖、可读性不足，因此保留于alignment_figures_initial；终版选择首条连续8点且净移动至少20px的真实历史，仅为图形可读性，不是预测分组、数据筛选或改善案例。青色起点/黄色终点，所有点仍是原标注行，未追加插值。

6/6样例的lost=0中心均在参考图边界内；near-boundary数量另存，原lost行和occluded/generated数量没有被隐瞒。这是对已下载前缀的统计，不能外推全部60录像。部分图像在道路/步道附近，另有树冠/绿化边缘歧义；reference的采样帧未知，没有对应原视频帧，不能证明每个框时序位置准确、亚像素标定或所有轨迹都处于可通行区。

本轮假设annotation→reference为恒等像素映射，未引入未核实homography。缩放使用每幅图实际输出宽高：T=diag(W'/W,H'/H,1)，逆变换往返误差小于1e-9px；这只是数学自洽，**不是配准误差小于1e-9px**。Y-Net把轨迹乘统一resize，再对图像OpenCV缩放、右/下padding32；尺寸四舍五入及像素中心约定带来亚像素差，独立结果在resize_convention_audit.csv，不能把它与本轮PIL边缘坐标矩阵无声等同。

六个录像分别使用各自尺寸/矩阵，未共享世界变换。人工逐图检查发现，deathCircle/video0的track1历史投影接近树冠/绿化边缘，hyang/video0也有阴影/遮挡邻近情况。没有语义GT及原视频时，这只能记录为场景解释歧义，不能判定轨迹错误、真实不可通行或模型失效；它也说明不能把tree/terrain标签直接当确定障碍。

没有米制标定，不使用近似estimated_scales来宣称精确米定位。语义图未取得，D项语义对齐无法执行；不把RGB静态场景当作D通过。
''')
    report('03_SEMANTIC_RESOURCE_AUDIT.md',r'''
# 场景语义资源审计

**人工语义标签、palette与分割预训练权重均未取得。** 原SDD的bbox轨迹/类别标注不是道路/建筑物等逐像素语义。六幅reference为真实RGB示例帧，不是语义图。

Y-Net [原文数据与实现部分](https://openaccess.thecvf.com/content/ICCV2021/papers/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.pdf)描述为SDD手工标注5种stuff：pavement、terrain、structure、tree、road；并使用预训练后在具体数据集微调的分割网络。[作者README](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/README.md)把数据/config/权重指向单一Google Drive补充包。链接网页存在不等于已得到其中每种文件。本轮实际只有元数据及约949M包的下载确认页，未提交大型下载确认；语义文件/权重在当前环境仍未取得。

源码有RGB经SMP分割分支，以及seg_mask输入按类别ID one-hot分支；YNetTorch默认semantic_classes=6而论文stuff为5，可能背景/实现配置不同，**未取得配置和palette前无法确定映射**。本轮合成6通道只是工程接口参数，不当作真实类别定义。不能用任意RGB值或灰度自动解释成这些类别。

训练、验证、测试的手工语义图数量、质量、label制作流程与分割训练划分均UNVERIFIED。分割模型是否见过评测场景的人工标签也未知；以后必须披露分割预训练与目标预测各自split，不能把包含测试场景监督的分割器称为完全未知场景泛化。也不能在未知事实下指控原作者泄漏。

可行走性不是简单二分：structure可能对应实体障碍；tree包含树冠遮挡，不能直接视作地面不可走；terrain/road/草地也不应一律排除。场景标签未提供出入口/通行连通性/动态障碍的可靠GT。当前无法计算真实scene compliance或“语义违规率”，更不能称其为安全性。

参考图只表达某时刻背景，可能含动态参与者，不能等同整段录像的动态语义。未尝试用Semantic Drone Dataset替代标签，未运行任何批量伪标签，也未把现有通用分割器输出当人工GT。备用方向只是未来在独立环境审计公开分割模型和人工核对一小部分数据；需要单独授权和来源/质量验证。

当前阻断项：作者语义资源实际取得；ID/颜色/ignore类定义；与每reference的像素尺寸及变换；分割模型来源和标签split；障碍/未知类别规则。任何一项未解决，都不能启动Q2/Q3定量验证。
''')
    report('04_DATA_PROTOCOL.md',r'''
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
''')
    report('05_BASELINE_FEASIBILITY.md',r'''
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
''')
    rows=cread(OUT/'literature_comparison.csv')
    table='|论文/发表状态|阅读范围|核心重合与未核实项|\n|---|---|---|\n'
    for r in rows:
        table+=f"|[{r['title']}]({r['paper']})，{r['year']}，{r['venue']}|{r['actual_reading_scope']}|{r['overlap']}；{r['unresolved']}|\n"
    report('06_NOVELTY_AUDIT.md',r'''
# 场景语义与目标条件预测：创新性审计

检索流程为nature-academic-search multi-source-search。学术MCP未挂载，OpenAlex回退HTTP429已记录；使用CVF、作者arXiv、正式出版页、EuropePMC原文XML及冻结源码。逐篇去重/区分版本与实际阅读范围。检索关键词覆盖scene-aware、goal-conditioned、semantic-guided、endpoint refinement、scene compliance、semantic uncertainty、multimodal goals。

十篇比较如下，完整列含输入、核心模块、goal机制、语义融合、loss、代码链接及未解决问题，保存在literature_comparison.csv。七篇方法正文可读（IAD为2025预印），三篇近期指定工作仍缺完整方法；不伪称十篇均全文精读。

'''+table+r'''
重要版本限制：GDS-TGTF为2026-10-14新上线pre-proof，不假定正式卷期；UATP为2026 arXiv，不标成已接受会议。IAD的AAAI2026元数据与DOI已核实，但正式PDF访问失败，方法依据arXiv2508.07146；不拿另一篇2024 Intention-aware Denoising Diffusion Model冒充它。Goal-GAN最初mediatum文件实为作者博士论文合集，已改核验CVF ACCV2020单篇原文。SSU直接Springer/PMC访问受阻，后来通过EuropePMC官方公开XML取得全文并阅读方法/实验。上述失败均保留。

## Q1—Q3逐项审查

Q1：历史不能明确目标时，场景是否有独立收益？作为具体条件分组仍可验证；**一般的场景帮助预测/goal分布不是新问题**。Y-Net、Goal-GAN及SSU已使用场景/目标分解，SSU更明确比较RGB特征、semantic embedding和完整分割。不能把换个注意力或加入semantic通道当新贡献。本轮没有训练后预测，不能声称SDD此子群已证明存在模型缺陷。

Q2：不可靠语义是否伤害目标预测？是三个问题中较值得优先诊断的一个，但不是自动新创新。UATP已显式处理位置和语义地图不确定性；其nuScenes在线vector map与SDD静态栅格不同，**换数据域不等于数学机制创新**。IAD的semantic主要指运动意图，不应和像素语义图混为一谈。本地没有真实语义标签/模型误差，不能断言SDD存在已量化的语义噪声损害。

Q3：相同K约束能否减少不可行目标且保精度？Y-Net/Goal-GAN已有场景条件目标与路由，MG-GAN已讨论OOD候选和accuracy/coverage权衡，CINet/McTrans/GDS也覆盖环境/goal或refinement主题。固定阈值屏蔽、投影终点到步道、普通goal refinement都不能直接称创新。还必须区分目标点FDE与真正路径末点、全路径穿障碍、树冠语义不确定性和GT oracle。

当前**未提出可主张新颖性的预测模型**。CINet/McTrans/GDS完整公式未核验，是明确的查重缺口，不是“它们没有做”的证据。没有充分数据/文献依据启动SCI算法开发；有限诊断方案见07，正式新算法NO-GO。
''')
    report('07_MINIMAL_RESEARCH_PLAN.md',r'''
# 唯一保留的候选研究问题：语义可靠性对goal/路径约束的影响

这是一项待验证的诊断方案，不是三个创新点或已完成新模型。没有为了凑数给第二个网络方案。

1. **问题**：预测目标候选的可达性与覆盖是否会被错误/不确定语义破坏，约束是否在减少不合规目标时丢掉合理模式。
2. **真实证据及局限**：六份实际参考图/历史对齐支持数据几何可审计；存在generated/occluded标记和遮挡场景，但没有真实semantic map或训练后预测，故SDD预测损害尚未证明。不得把人工扰动的结果先填为真实失败。
3. **已有处理**：Y-Net场景热图+goal/waypoint；SSU明确语义编码升级；UATP地图语义/位置不确定性；MG-GAN关注不合理样本。广义思想已覆盖。
4. **尚未验证的差异**：SDD静态栅格中语义类别边界/树冠/terrain歧义与目标候选留存的关联，和同K下整个路径一致性；不是声称没人研究。
5. **可能技术机制**：本阶段不确定新算法。先对原Y-Net输入语义做预注册、GT无关的缺失/边界扰动和质量分层；若只有事后oracle投影有效，停止。普通置信度gate、attention或障碍阈值不当创新，真正机制需在证据与查重完成后另审。
6. **最小实验**：先补全官方train/test、合法source-train validation和语义资源，再严格加载Y-Net做少量validation功能验证。冻结同一目标/图/GT、Ke20×Ka1、TTST/CWS规则；不先碰test。不运行本计划中的任何训练。
7. **公平对照**：同目标CV-Last2/OLS4/OLS8、原Y-Net、PECNet参考；独立归因还需同骨干history-only/RGB/semantic/semantic+goal对照。不同输入通道、分割预训练、头和有效容量不一致时必须标混杂，不能单用PECNet差值证明语义有效。
8. **消融与辅助证据**：真实语义/固定程度扰动/缺失语义，固定权重干预只作OOD敏感性；独立重训练才检验学习分布。扰动种子/程度和K在误差前固定，所有源录像和历史难度组保留，任何未来GT不得生成输入/挑候选。
9. **预算**：数据/语义/配置未齐，正式训练GPU小时无法可靠估算。建议之后工程阶段先预留≤1GPU小时上限，资源测量后按真实完整epoch/全目标数另报；不是批准训练，不能用本轮core-forward乘epoch冒充预算。
10. **否定/停止**：原baseline/数据身份/标签对齐任一不成立即停；语义质量下降无稳定goal/path损害，则不支持Q2；同K约束不增合规覆盖或明显恶化ADE/FDE，则不支持Q3；只有GT oracle/额外容量收益、或与UATP等公式等价，则停止算法方向。不增加模块补救。

Stage11B应先做**Y-Net官方资源与兼容性准入**，数据优先作者短期SDD的来源训练区域和合法validation，test封存。PECNet保留参考，不替代共同输入消融。当前不具备SCI小论文算法开发资格，也没有已证实性能增益。只有补齐数据、基线可信且至少一个独立技术问题成立，才讨论真正的方法差异。
''')
    report('08_METRIC_SPECIFICATION.md',r'''
# 后续评价规范草案

本阶段不计算任何新模型真实ADE/FDE或scene-compliance成绩。以下公式是规范，不是实验结果。统一同source/video/person/20帧支持、同GT及reference/semantic/transform SHA，输出单位原图像素。

对K条完整预测路径，e_kt=||Yhat_kt-Y_t||₂，ADE_k=(1/12)Σ_t e_kt，FDE_k=e_k,12。minADE_K=min_k ADE_k，minFDE_K=min_k FDE_k，独立取最小、不要求同一候选。K1须固定采样/确定性定义，不与K20直接排名。CV只有一条，不作为同候选预算战胜K20的证明；预测像素误差不和ETH米制混排，不同预测时长不排名。

GoalFDE_K=min_k||g_k-Y_12||；必须另存路径FDE。单一终点的“Endpoint ADE”在该时间点等于Endpoint FDE，没有另外12步平均含义；如果评waypoints则明示时间集合。Y-Net原test.py的val_FDE使用waypoint_samples末目标，ADE却使用future_samples完整路径，二者可不相等。PECNet原test_pretrained_model.py先用GT挑best endpoint、才解码一条path，不能把它叫独立minADE_K。将来应同时保留原口径和独立K完整路径重评，不偷偷更换官方数字。

TTST使用10000 proposal再聚类输出20goal；Ke×Ka=20只等于最终候选数量，不等于生成计算预算。所有对照同时记录proposal数、decoder次数、TTST/CWS开关、候选随机种子和耗时。离线GT oracle最小误差不是在线选择能力，不能用GT过滤/重排部署候选。

## Scene Compliance前置条件与定义

先取得与reference逐像素对齐的人工class map、palette、ignore类及质量说明。当前未满足，不填0或假违规率。不可通行类别必须按真实标注任务定义，不默认terrain/road/树冠不可行走。建议structure等经核实障碍单列，歧义类保持unknown，且报告有效语义覆盖。

对每个候选分别记录：终点in-bounds、终点障碍/unknown/boundary；以及全部12点的障碍率、任意点违规路径率、出图率。图像外单独计失败，不能clip到边界后记合规；细路径可能穿越两采样点间障碍，若做连线采样需冻结步长并分开报告。语义边界设置预定原像素不确定带，报告边界和已知内部各自分母；不得用丢弃unknown美化结果。

同K报告全候选合规比例、至少一条合规候选的目标覆盖、候选多样性及ADE/FDE权衡。GT选最准确合规候选只能标oracle；对所有candidate同样计算，不只显示成功者。静态标签一致性不代表真实避碰、安全性、动态路权或人的行为意图。

统计同时保留目标窗口加权、source/video/person先平均后等权及来源录像/物理site宏平均。重叠窗口不作独立样本，若以后给不确定区间按source/person聚类并注明有限source数量。相同背景跨split的泛化范围必须说明。
''')
    report('STAGE11A_REPORT.md',r'''
# Stage11A总报告：SDD语义与目标点研究准入

**最终NO-GO：当前不能启动Stage11B正式训练或SCI新算法开发。** 完成了资源、几何、公开基线、指标与文献审计；没有正式训练、没有新模型test成绩或下一阶段自动运行。

1. **数据与许可**：官方SDD8地点/60录像，CC BY-NC-SA3.0。原视频、完整作者split/config/权重/语义图未取得。六个镜像录像的图像/标注共23.19MiB，仅PARTIAL，官方一致性UNVERIFIED；未下载69GB视频或作者完整大包。详情01与resource_inventory.json。
2. **坐标对齐**：六张真实8点历史叠加图及初版可读性修订保留；位置/resize/逆变换核对通过，所读lost=0中心均在各自图内。没有视频/参考帧ID/语义图，不能证明精确原始或语义配准，不能称米制精度。详情02、alignment_results.json。
3. **语义**：Y-Net论文有5种手工stuff，但文件/palette/分割来源未取得，源码默认6通道关系未解；不能把RGB或Semantic Drone Dataset当SDD标签，未生成伪标签。详情03。
4. **基线可行性**：作者原Y-Net核心在Torch2.5.1下B1/B2合成前向通过；工程核心参数1,641,381，不含分割，非完整作者配置。缺SMP/语义权重，原ragged loader失败，Pandas.append和epoch早停等需隔离适配审查，未修改官方源码。PECNet保留参考，Stage7结构结果只引用。详情05与baseline_probe.json。
5. **协议风险**：官方短期训练notebook把test_trajnet作为validation选模。本轮不执行、不用test调参。protocol_draft.json仍DRAFT，确切train/val/test清单未取得，背景跨split共享未知；不能把录像隔离叫未见场景。详情04。
6. **评价差异**：Y-Net FDE为采样goal误差，未必decoded-path末点误差；PECNet GT先挑终点后解码并非全部K paths独立minADE。已制定K1/K20、GoalFDE/TrajectoryFDE、像素/米、oracle、语义边界/出图和重复窗口规范，未造指标。详情08。
7. **查重**：10篇比较；7篇方法正文可读（IAD仅预印版本），CINet/McTrans/GDS-TGTF完整方法未取得。Scene+goal不是新题，SSU已做语义升级，UATP2026已研究语义地图不确定性；不能据摘要宣布机制空白。详情06与literature_comparison.csv。
8. **唯一待验证问题**：优先考虑不可靠语义对goal覆盖/路径约束的影响，当前只是诊断问题；SDD实际预测损害未证明，没有新网络/算法创新声明。最小实验、公平对照与停止条件在07，不为凑数提出第二方案。
9. **资源**：RTX3080工程核心合成前向已测，20次合成decoder调用无OOM、allocated峰约375MiB，未含分割/梯度/TTST/CWS，不能视作官方K20完整推理。正式训练GPU时间尚不可估，下一次若授权先≤1GPU小时工程准入，完整epoch实测后另报。当前不训练。
10. **测试与保护**：全仓库旧501项+新增审计工具15项，516项通过；保留完整pipeline两项失败，不把unit通过当正式复现。最终历史SHA以history_after.json为准；旧源、配置、权重和指标不变。源blob/数据/图像/代码hash及失败记录在results/tables/stage11a。

下一步建议仅补齐作者数据/语义/config资源并审查许可、真实split、分割标签曝光及隔离兼容；Y-Net是条件性复现对象，PECNet是参考。完成这些条件之前不冻结正式实验、不提出新模块或宣布SCI创新资格。

新分支`feat/stage11a-sdd-semantic-goal-audit`，从远程Stage10A已推送521c53a建立；最终commit SHA以Git交付为准。本阶段明确授权推送新分支；如认证失败保留本地commit与完整错误日志，不force、不动main。完成后停止。
''')

if __name__=='__main__':main()
