"""Evidence-derived Stage11A+ reports; no scores or training results invented."""
from common import *
def main():
    report=ROOT/'reports/stage11ap';report.mkdir(exist_ok=True)
    d=json.loads((OUT/'download_manifest.json').read_text())
    arc=json.loads((OUT/'archive_audit.json').read_text())
    s=json.loads((OUT/'split_identity_audit.json').read_text())
    p=json.loads((OUT/'full_pipeline_admission.json').read_text())
    iso=json.loads((OUT/'real_train_future_isolation.json').read_text())
    literature=json.loads((OUT/'CINet_source_manifest.json').read_text())
    runtime=json.loads((OUT/'runtime_environment.json').read_text())
    warnings=json.loads((OUT/'checkpoint_serialization_audit.json').read_text())
    protocol=json.loads((CONF/'protocol_candidate.json').read_text())
    docs={}
    docs['01_AUTHOR_RESOURCE_DOWNLOAD.md']=f'''# 作者资源下载与完整性

日期2026-10-20（Asia/Shanghai）。用户已授权该指定约949M资源，不授权69GB原视频。来源为[冻结Y-Net README](https://github.com/HarshayuGirase/Human-Path-Prediction/blob/{MODEL_COMMIT}/ynet/README.md)中[作者Drive资源](https://drive.google.com/file/d/1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa/view)，ID `1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa`。来源绑定作者链接，不宣称有作者签名或作者公布的checksum。

**下载成功**。`ynet_additional_files.zip`实际 **{d['bytes']:,}字节**（约{d['bytes']/1024**2:.2f}MiB），SHA256 `{d['sha256']}`。HTTP MIME/Content-Disposition在download_manifest.json；文件PK签名、zipfile识别、所有成员CRC均通过，非HTML。页面949M不是已验证字节数。

下载前剩余磁盘{d['disk_free_before']/1024**3:.2f}GiB。首次传输因会话中断留下{d['resumed_interrupted_bytes']:,}字节，后使用标准HTTP Range续传，未绕过登录/验证码；实际状态与请求链见download_manifest。标准大文件确认表单只用于作者公开文件的正常下载，不是身份验证绕过。源HTML确认页留在忽略cache，不发布临时表单字段。

新隔离venv安装gdown时默认镜像站出现TLS EOF，日志在cache/gdown_install.txt；PyPI安装成功。实际资源下载由审计脚本requests流式/续传完成，未改动任何旧环境。

压缩目录220项（含69个目录），文件151个，预计展开{arc['uncompressed_bytes']:,}字节（约{arc['uncompressed_bytes']/1024**3:.3f}GiB）。先验证路径穿越/绝对路径/反斜杠/重复/符号链接/加密/异常压缩比和至少5GiB解压余量，再按资源扩展名逐项提取；未执行包内脚本，没有盲目extractall。archive_members.csv、archive_audit.json和extracted_resource_SHA256.csv可复核。

ZIP及解压内容都在`experiments/stage11ap/cache/`，不入Git；未下载原SDD视频或其他大型数据包。SDD按[官网](https://cvgl.stanford.edu/projects/uav_data/)CC BY-NC-SA3.0管理，Y-Net源码MIT；补充包没有另附覆盖全部资产的许可证，不据此扩大原始数据使用权。
'''
    docs['02_RESOURCE_INVENTORY.md']='''# 实际资源清单与状态

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
'''
    docs['03_SEMANTIC_ALIGNMENT.md']='''# 真实语义图、参考图与轨迹对齐

60张SDD掩码均为单通道L模式，实际像素ID集合**仅1、2、3、4、5**，无0、无调色板。47个有参考RGB的录像全部尺寸一致；另13个仅有mask，没有该包内RGB/短期目标。真实掩码与普通RGB严格区分，未生成伪标签。

短期yaml明确semantic_classes=6，正式分割head为Conv2d(16,6,3)+Softmax(dim=1)，预测模型第一层输入14通道=6+8。因此6输出通道对应0..5，但实际提供mask仅出现1..5。**不能据此给未出现的0命名background/ignore**。包内inD专用README写0:unlabeled、1:pavement、2:road、3:structure、4:terrain、5:tree；这是inD的已知映射，只能作为SDD的待核查线索，不能无声明迁移。SDD类别名称、ignore、训练loss如何处理0仍UNVERIFIED。

Y-Net论文描述人工标注五种stuff，作者包提供离散区域图并与该描述一致；没有逐文件标注者、自动生成/修订历史或分割训练脚本，不能独立证明每个像素的制作过程。SDD_segmentation checkpoint实际用过哪些train/test标签未知；包里有所有17个test录像mask，这不证明权重一定训练过它们，也不能假定未曝光。

六组**作者训练池**真实案例：bookstore_0、coupa_3、deathCircle_0、gates_0、hyang_4、nexus_0。每组包含RGB、语义ID图、8历史点、12个train GT离线叠加，scene/video/frame/person/meta、reference_SHA/semantic_SHA及变换见alignment_cases.json与alignment_points.csv，PNG在alignment_figures。GT只在离线对齐脚本读取，预测函数根本没有future参数。

采用作者原像素坐标，RGB→mask同尺寸identity；推理使用标称0.25倍坐标、cv2.resize(fx=fy=.25)，然后只在下/右侧补齐32倍数。图像整数舍入引起的实际尺寸比例与标称.25细微差异不能叫精确世界定位。无米制比例尺、单应矩阵或estimated_scales.yaml。

视觉检查：可见道路/建筑/草地大区域在RGB与mask上大体对应，但边界为粗粒度手绘多边形；gates_0与nexus_0的轨迹接近或跨越标签边界，不能在名称/边界未核准前当作真实碰撞。当前证明的是作者配套图像的几何同网格和实例可读性，**未证明原视频的精确时空配准，也未证明逐像素语义正确**。语义合规率本轮不计算。
'''
    docs['04_OFFICIAL_CONFIG_AUDIT.md']=f'''# 官方配置与checkpoint

冻结作者源码`{MODEL_COMMIT}`，仅拷贝到本阶段cache执行，源码SHA与Stage11A完全相同；MIT来源记录在frozen_source_manifest.json。没有修改第三方模型。

正式短期配置：obs8/pred12（notebook）、resize0.25、encoder[32,32,64,64,64]、decoder[64,64,64,32,32]、semantic_classes6、use_features_only=False、waypoints=[11]、temperature1、TTST=True/rel_threshold0.01、CWS=False。最后索引11就是目标点，**没有额外中间waypoint**。配置在official_sdd_config.json。

输入：参考彩色图按cv2的BGR顺序读入；官方分割预处理采用默认RGB/ImageNet参数但没有显式BGR→RGB换序，本轮保留这项公开代码行为，不能悄悄改色序后称原样复现。经ResNet101-Unet得到6通道softmax；再拼接8历史距离图，输入[batch,14,H,W]。Y-Net编码→12通道goal/waypoint热图；第11通道用于终点分布。TTST按原函数采10000点，19个kmeans中心加1个softargmax组成20个goal。给各层编码特征拼接goal距离图金字塔，原path decoder和softargmax输出20×B×12×2。推理无GT waypoint。

分割权重SHA `{p['segmentation_SHA']}`。预测完整state dict SHA `{p['checkpoint_SHA']}`。776键/形状与实际模型完全一致，`strict=True`，无缺失/额外键；**没有strict=False**。总参数{p['parameters']:,}，其中分割{p['segmentation_parameters']:,}、轨迹核心{p['core_parameters']:,}。这些是实际模块参数，不是“有效loss梯度参数量”，本阶段不计算loss/梯度。

原torch.save(module)包含旧式persistent module引用，weights_only先失败，独立白名单loader恢复分割对象，再将该对象装入原YNetTorch、严格载入完整state dict，等价于改变加载流程而不是模型前向。预测checkpoint同时含semantic_segmentation.*，最终值以整套预测checkpoint为准，不混接一个任意分割权重。

旧库源码警告共比较{warnings['compared']}类，{warnings['changed']}类文本改变，完整差异见legacy_library_source_diff.txt及CSV。没有隐藏Torch1.5→2.5/torchvision等变化；当前属于功能性兼容，不保证历史环境浮点数值逐位一致。

训练入口存在yaml batch8 vs notebook BATCH_SIZE4，原短期epoch还有26scene上限；不能把两配置混用或将300名义epoch称300完整遍历。unfreeze150控制分割前向调用位置；原train入口始终将分割参数requires_grad=False，不能仅按配置名字宣称150轮后真正解冻参数。未来方案必须预先明确这些语义，本轮未运行训练。
'''
    docs['05_FULL_PIPELINE_ADMISSION.md']=f'''# 真实训练数据完整推理准入

结果：**REAL_TRAIN_INPUT_PIPELINE_PASS**。不称OFFICIAL_REPRODUCTION_PASS，不是新算法，不是正式性能复现。

实际运行六个合法原train录像，各取一个真实20点目标，但预测器只接收前8点；未使用候选validation的5个录像，也未预测任何test目标。调用链为原RGB→实际分割权重→历史距离图→原编码器→原goal热图→TTST20goal→原path decoder20条12帧轨迹。CWS关闭，Ka=1，Ke=20，10000是proposal数量，不冒充最终候选数。

六例均输出[20,1,12,2]原像素坐标，分割[1,6,H,W]概率和与有限值检查通过。best-goal或future没有从GT选取；原evaluate函数包含GT/误差计算，本阶段故意不调用，独立推理wrapper仅调用其原推理算子。每例真实预测点在forward_*_predicted_paths.csv，无ADE/FDE、无候选排名。

固定同一训练目标历史，将未来12点替换成极大数，重置相同torch/NumPy seed后再次TTST推理，goal及全部K20轨迹逐值相同，最大差{iso['max_abs_prediction_difference']}。是API输入隔离证据，不证明未审查的所有官方入口都安全。原训练GT teacher forcing与本轮推理严格分开。

所有参数和buffer在推理前后逐值相同；inference_mode、eval、optimizer.step=0、loss计算0、test评价0。checkpoint完整键形状与严格加载已验证。详细状态full_pipeline_admission.json、real_train_future_isolation.json及stdout保留全部源变更/meshgrid警告。

工程适配：SMP0.1.0在新venv安装通过；逐scene float32数组替代NumPy隐式ragged array；batch1 scene collate显式as_tensor；原旋转/翻转调用保留、Pandas append换独立concat适配器。官方源不变，未改模型forward/loss。原26scene cap也不删；提供明确无cap的独立数据迭代入口供未来方案选择，**未建立或运行新训练runner**。timm0.1.20因torch._six缺失仍导入失败，但已确认本ResNet101分割不走timm分支；不安装其他骨干/不盲目修复无关分支。
'''
    docs['06_SPLIT_AND_PROTOCOL.md']=f'''# 作者协议与HTraj开发候选

作者train_trajnet.pkl：30录像、8,494目标、169,880行。test_trajnet.pkl：17录像、2,829目标、56,580行。不是全原始SDD60录像。逐metaId检查20行、单录像/track、有限坐标、严格frame差12通过；没有凭名字把test认作val。官方notebook却确实用test选最低ADE，本轮拒绝该入口，不计算测试误差。

作者train/test录像交集、完整窗口身份交集均0。因此同录像同track的20帧支持不会跨两个split；track数值跨录像重复不代表同一人。站点背景交集为coupa/gates/hyang/nexus，不能称未知场景泛化。跨录像是否跟踪同一自然人没有标注，不能声称已全部身份去重。

每个(scene,trackId)在短期包里只有一段20帧，没有可观测相邻窗口stride，**不能据此编造stride20**。作者原短期TrajNet选窗/原标注到pkl的完整链缺失；原预处理源可查但不是这些pkl实际生成参数的证明。所有输入和GT按sample_id/source(scene)/person/frame/history_SHA/GT_SHA记录在author_target_manifest.csv。训练池有2个出图坐标点，保留，未clip或删难例；未来loader边界处理需声明。

独立开发候选：仅在作者训练录像池内，每个至少2录像的地点，以固定SHA256('HTraj-SDD-dev-v1:'+sceneId)最小者作val。仅有coupa_3的地点留在train。固定val为{', '.join(s['candidate_val_scenes'])}。得到train7,496目标/25录像、val998目标/5录像、sealed test2,829目标/17录像。录像隔离使完整228原frame时间支持不跨新split；不需用时间块隔离带弥补本已不同的录像。仍非未知地点；未根据预测误差调整划分。

已下载预测权重可能训练过这5个新val录像，也可能用官方test选模；**不能用它报告独立开发泛化成绩**。新开发协议只能用于之后单独批准的从头训练方案；作者原train/test清单单独保留，不混称论文数值复现。

时间：30fps依据作者downsample文档，包内帧差12→2.5Hz/0.4s；没有下载原视频实测fps。8历史首末2.8s，未来末观测至终点4.8s，20帧首末7.6s。像素单位，无米制假设。原pkl缺少lost/occluded/generated/label，不伪造这些字段或把缺失当0；数据状态与来源局限见protocol_candidate.json。

**NOT_READY**：数值身份/隔离通过，但SDD类别名称和ignore规则、分割预训练标签曝光、原短期目标选择/stride来源仍待核准，新训练日程也未冻结。正式协议不能由一个成功forward自动批准。默认test封存、不开训练、不开验证排名。
'''
    docs['07_NOVELTY_GAP_UPDATE.md']=f'''# 三篇近期工作的查重缺口更新

使用nature-academic-search的multi-source-search流程；未挂载学术MCP，CrossRef→OpenAlex→Semantic Scholar以及正式出版页/作者代码有限回退。9次metadata响应在literature_access_update.json及cache，网络失败/非JSON不包装成成功全文。搜索范围仅三篇指定文献，不换新模型。访问日期2026-10-20。

|论文|本轮实际增加的证据|仍未核实|
|---|---|---|
|[Completed Interaction Networks](https://doi.org/10.1109/TMM.2025.3542967)，TMM2025|核查正式元数据/摘要；审查自称作者实现的[CINet-main](https://github.com/Jayzhou3219/CINet-main)核心CINetr、CINetv、another_goal与STTL源码|完整论文数学方法未取得；repo README自称官方，独立出版方→源码归属未交叉确认|
|[McTrans](https://doi.org/10.1016/j.jvcir.2026.104723)，JVCIR2026|出版页引言/摘要/可见方法概述，CrossRef与Elsevier官方TDM响应|TDM200只含coredata、无方法正文；未找到经核实作者预印本/公开代码|
|[GDS-TGTF](https://doi.org/10.1016/j.eswa.2026.134731)，ESWA2026-10-14 pre-proof|出版摘要/亮点与官方TDM元数据|无完整公式、训练细节或可核实公开代码，不能据摘要断言无语义可靠性机制|

CINet源码冻结`{literature['commit']}`，49个小源码/README文件blob校验，静态审查未执行、未复制入Git。可见路径为6通道环境图+轨迹热图预测goal概率，结合时空分支与目标距离、当前时间等，loss含轨迹MSE与goal BCE。已覆盖场景/社会/短期goal联合建模。已读文件未能确认有校准的语义不确定性目标，但这**不是整篇论文没有该机制的证据**。可见代码还有缺失Ablation_Study导入、hard-coded cuda:1、forward内构造模块等审查风险；未运行，不能将其当作验证过的官方算法。

McTrans公开引言说明learnable goal token、环境CNN、层级cross-attention和逐步增强目标损失；这些不是待设计的新贡献。GDS公开摘要说明结构/运动双流、融合和目标逐级refinement。精确损失/可靠性权重/动态路径修正是否与拟议机制重合仍未知。

网络记录：GitHub API403限流后用git tree；首次代码过滤过宽下载到少量TXT数据，未解析、未预测/评价，随后明确排除data/output，仅小源码审查；中途异常和初始日志保留。Elsevier TDM返回HTTP200并不意味着全文可用，实际1840/1875字节只含coredata。没有使用第三方求助站/未经授权全文，没有绕过出版方访问限制，也未无限等待。

Stage11A已有Y-Net/SSU/UATP等重合结论保持不变。当前三篇全文缺口**没有全部关闭**。不能证明语义不确定性、目标候选可靠性尚未有人处理；不批准SCI新算法开发。本阶段没有提出新网络。
'''
    docs['08_RESOURCE_AND_BUDGET.md']=f'''# 实测资源与尚未确定的训练预算

允许上限1 GPU小时，只作工程准入。没有训练、loss、backward或optimizer更新，也未加载test进行预测。GPU为RTX3080 10GB。

|实际执行|成本与边界|
|---|---|
|作者包下载|{d['bytes']:,}字节；续传进程实测{d['elapsed_seconds']:.2f}s，首次中断前时间未完整测量，不把它算总下载时间|
|磁盘展开|{arc['uncompressed_bytes']:,}字节资源，另保留ZIP、环境和轻量缓存；未删除旧成果|
|六个训练输入完整K20前向|含加载/验证等程序墙钟{p['wall_seconds']:.3f}s；每例{min(x['seconds'] for x in p['cases']):.3f}–{max(x['seconds'] for x in p['cases']):.3f}s，第一例含冷启动；B=1|
|原GPU tensor峰值|allocated {p['gpu_peak_allocated']:,} bytes（{p['gpu_peak_allocated']/1024**2:.2f}MiB），reserved {p['gpu_peak_reserved']/1024**2:.2f}MiB|
|整卡nvidia-smi|{p['GPU_card_state']}；末尾读数不是高频采样峰值|
|未来GT隔离两次前向|整个进程{iso['wall_seconds']:.3f}s；无GT传入模型、无指标|

两次完整GPU测试进程合计墙钟约{(p['wall_seconds']+iso['wall_seconds']):.3f}s（约{(p['wall_seconds']+iso['wall_seconds'])/3600:.5f}小时），作为保守GPU占用上界记录，不声称实际GPU kernel时间。没有接近1小时预算，没有启动额外模型。首次独立分割load probe仅CPU。

**正式训练时间：未确定**。本轮B1无梯度时间不能推算300epoch训练；包括分割特征/BN、增强场景、batch4/8冲突、26scene cap及训练/val采样开销。SMP0.1.0 ResNet分支完整推理可运行，不等于大batch反向资源已验证。建议协议审查通过后另行批准一个完整训练遍历预算测量，再估算总训练和checkpoint开销；本阶段没有该训练许可。

当前新venv只装小依赖；Torch2.5.1/数值库通过只读pth复用已有stage4安装。无全局降级；timm旧版不兼容作为未用分支保留失败，不为此下载新大型模型。新代码/资源/配置均有SHA，所有原包和权重在忽略cache。
'''
    docs['STAGE11AP_REPORT.md']=f'''# Stage11A+总报告

**工程进展：正式checkpoint严格加载与真实训练输入完整推理通过。总体NO-GO：当前协议NOT_READY，不启动正式训练；SCI创新性也未获准。**

1. 作者指定资源下载成功：{d['bytes']:,}字节，SHA `{d['sha256']}`，ZIP/CRC/路径与空间检查通过；未下载69GB原视频。
2. 获得作者短期SDD池（8,494 train + 2,829 test目标）、47 RGB、60 masks、3 configs及5模型权重；原SDD完整视频/annotations仍不全。不能把子集称全原始数据。
3. 真实SDD单通道mask存在，ID1..5；分割模型6通道0..5。SDD ID名称/ignore无专用权威表，inD README仅作线索，不能假定第六类background。分割预训练用了哪些test/val标签未知。
4. 六个训练案例RGB/mask尺寸同网格、真实历史与train未来离线叠加已保存；图像边界粗粒度，原视频精确配准未证，不使用米制误差。
5. 776键/形状严格加载；参数{p['parameters']:,}（分割{p['segmentation_parameters']:,}、预测核心{p['core_parameters']:,}）。没有strict=False、没有修改冻结模型。
6. 真实输入完整链六例通过：[20,1,12,2]，TTST10000→20goal，CWS关，所有输入仅历史/RGB、无GT oracle，未来反事实差0、权重不变。无ADE/FDE、无训练。
7. 原包无独立val，notebook用test选模不能照搬。作者train/test录像不重合，地点共享4处。生成唯一source-train录像级候选7,496 train/998 val，原test2,829封存；只身份/支持核验，没有验证排名，协议保留NOT_READY。
8. 独立环境解决SMP、ragged、append所需数据兼容；旧式模块安全加载有专用受限fallback，保留旧库源码变更警告。timm未用分支仍不兼容；旧Torch数值等价未证明。训练26scene早停未静默删除，未来日程需单列。
9. CINet作者声明源码新增静态核查；McTrans/GDS正式接口只取到元数据，三篇完整方法仍有缺口。此前查重限制继续，不把工程适配称创新。
10. 实测GPU两个进程共{p['wall_seconds']+iso['wall_seconds']:.3f}s，allocated峰{p['gpu_peak_allocated']/1024**2:.2f}MiB；正式训练GPU小时未知，不能由B1推理计时填入训练预算。

Q1下载成功：是。Q2完整原始SDD：否，作者短期任务资源齐，但原生成来源部分未验证。Q3语义mask：有。Q4类别/坐标：同网格通过，名称/ignore和原视频精配准未全明。Q5严格加载：通过。Q6真实完整前向：通过。Q7正式split冻结：未通过。Q8是否仍有泄漏风险：原test选模、分割预训练标签曝光，以及将已有预测权重评估新val的风险均未被当作合法评价；本轮没有实际执行这些泄漏路径。Q9文献是否覆盖：广泛相关机制已覆盖，三篇完整细节仍未核实，不能宣布空白。Q10值得正式复现：Y-Net可条件性继续资源/协议准备，当前不批准正式训练。

下一步仅建议核实SDD类别与分割训练来源、审核source-train-only开发划分和训练预算定义，然后单独决定是否批准完整epoch测量。工程PASS≠论文性能复现PASS≠创新GO。

所有证据在results/tables/stage11ap；测试和历史SHA以full_unit_tests.xml、history_after.json、final_checks.json为准。分支feat/stage11ap-ynet-resource-admission，基于36e76a3；最终commit以Git交付为准。完成后停止，不进入正式训练。
'''
    # Explicitly distinguish engineering entry points from the mandatory legacy toy tests.
    note='\n完整单元测试范围说明：用户要求运行旧完整测试套件；其中既有frozen-ego/AgentFormer/NoSocial容量等测试会在临时人工模型上调用backward/optimizer.step以验证恢复或梯度。这不是Y-Net训练实验，不使用真实数据训练、不改旧权重。上述Y-Net入口的loss/backward/optimizer计数均为0，不能把它扩大成整个pytest进程没有任何优化器调用。原测试源码未修改。\n'
    for name in ['05_FULL_PIPELINE_ADMISSION.md','08_RESOURCE_AND_BUDGET.md','STAGE11AP_REPORT.md']:
        docs[name]+=note
    for name,text in docs.items():
        with (report/name).open('x') as f:f.write(text)
    print('Wrote',len(docs),'Stage11A+ reports')
if __name__=='__main__':main()
