"""Report actual train-only admission results, not prediction accuracy."""
from .common import *
import pandas as pd,xml.etree.ElementTree as ET
def main():
    folder=ROOT/'reports/stage11b0';folder.mkdir(exist_ok=True)
    s=json.loads((OUT/'split_audit.json').read_text())
    r=json.loads((OUT/'training_execution.json').read_text())
    b=json.loads((OUT/'budget_estimate.json').read_text())
    a=json.loads((OUT/'augmentation_audit.json').read_text())
    gate=json.loads((OUT/'cpu_test_gate.json').read_text())
    times=pd.read_csv(OUT/'training_timing.csv');measured=times[times.phase=='full_pass']
    success=r['status']=='ENGINEERING_TRAINING_PASS'
    docs={}
    docs['01_SPLIT_AUDIT.md']=f'''# 数据身份与时域隔离审核

**SPLIT_IDENTITY_PASS**，不是FULL_PROTOCOL_READY。沿用Stage11A+不重选：train7,496目标/25录像；val998/5，固定bookstore_2、deathCircle_3、gates_7、hyang_7、nexus_4；test2,829/17。

源manifest SHA `{s['source_manifest_SHA']}`。原作者train.pkl内8,494个目标的history/GT/reference SHA重新核对；每个目标20行、frame差12、单个scene/track，窗口为first..first+228原frame。train/val/test录像身份和窗口ID互斥，按录像定义的全部20帧支持不会跨集合。test.pkl仅作opaque SHA核对，未反序列化test未来，不做预测。

训练专用副本只含7,496目标/25录像，来自已核验作者train池，行值/排序不变。GPU入口只加载该副本，不打开含预留val的原train.pkl，更不打开test.pkl。对应哈希在training_data_artifact.json，实际访问ID在actual_target_visits.csv。

两个已知出图点都在nexus_0/track125，frame2904和2916，y1950.5/1956.5超过原H1947。原值保留，按作者0.25与round得到y488/489，在补边后H512范围内；没有clip、过滤或改GT。原图外点的监督不代表真实图像内标签，未来评价仍保留。out_of_image_points.csv可复查。

限制：同地点多录像共享背景；原train/test共享coupa/gates/hyang/nexus，新的train/val也不是未知地点。原始视频全局时间/自然人跨录像身份无对应证据；只证明录像ID级隔离，不能把该结果扩大为完全独立自然人确认。作者短期TrajNet选窗的原始生成manifest不全，原pkl没有类别/lost/occluded/generated，未伪造这些字段。

数据来源绑定作者Drive ZIP及SHA，仍不是完整原始SDD。未按误差删除任何目标；本轮训练实际完成范围由training_execution.json记录。语义不确定性不妨碍用户本轮明确授权的train-only工程测试，仍阻止完整科研协议自动冻结。
'''
    docs['02_SEMANTIC_AUDIT.md']='''# 语义类别与分割预训练来源

状态仍为**SEMANTIC_CLASS_MAPPING_UNVERIFIED**与**SEGMENTATION_EXPOSURE_UNVERIFIED**，但本轮新增了作者原文依据，不能再笼统说作者没有说明。

成功从[CVF正式论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html)确认并下载[官方补充ZIP](https://www.openaccess.thecvf.com/content/ICCV2021/supplemental/Mangalam_From_Goals_Waypoints_ICCV_2021_supplemental.zip)（约10.6MB）。补充PDF SHA和访问失败记录在semantic_supplement_addendum.json/semantic_audit.json。首个错误大小写链接404、canonical ZIP TLS失败均保留；没有绕过访问限制。

补充材料4.1（PDF第4页）明确五种stuff——pavement/tree/terrain/structure/road，以及用于padding和inD黑区的额外dummy；因此六通道的概念来源有直接原文支持。60张SDD mask只观测到ID1..5，真正SDD的编号→类别和ignore规则仍没有表格可核实。不能把inD README直接移植，不能给0命名background，也不把tree/terrain直接设障碍。

补充材料还明确声明分割预训练仅使用相应轨迹训练场景。应区分：**方法级声明有依据**，**当前公开checkpoint确切成员清单未取得**，不是已经证明它用了test，也不能声称文件级已证明完全没见过test。新HTraj val来自作者原训练池，其语义标签曝光仍可能存在；不能宣称独立未知场景语义泛化。

原文分割架构Unet+ResNet101、ImageNet骨干、Dice、lr1e-4/batch4、1500轮后lr1e-5、前200轮冻结骨干；这里只审查，不重新训练分割器。作者公开Git和ZIP没有提供可绑定checkpoint的完整分割训练manifest/脚本/日志。当前文件SHA沿用Stage11A+，原权重不变。

本轮仅将作者分割权重作**冻结eval特征提取器**，无梯度、BN不更新；手工mask不作为Y-Net输入，不进行语义违规评价。未来若要求严格未知场景语义，须提出单独批准、无评测标签曝光的分割来源/训练协议。
'''
    docs['03_TRAINING_CODE_AUDIT.md']=f'''# 训练代码、初始化与工程入口

作者版本`{OFFICIAL}`，源文件复制到新cache后逐SHA核验，不修改原源码。源码MIT、现有HTraj安全loader/数据adapter接口已核查，来源写入新脚本。环境继续使用隔离Stage11A+ venv（Torch2.5.1+cu124/SMP0.1.0），不装新库或降级旧环境。

作者论文/补充与YAML为batch8，短期notebook实际传batch4；本轮明确选**scene batch1、target microbatch4（notebook值）**，首个smoke target1，其后4/4。scene指一个录像下的轨迹集合，不是同时可见人数。没有把整个scene的几百目标当一个optimizer batch，也未按拥挤程度删样本。

原短期train.py在batch>25后break，增强后只访问26scene；原25录像池有200增强scene，作者30录像池则240，不能称完整epoch。本轮只做**无增强完整25scene一遍**，另用CPU核查8x增强访问数，未运行增强GPU epoch。

固定seed17、Adam lr1e-4、FP32、无新loss/clip/AMP/调度器。仅个体历史/静态场景；卷积编码器及goal/path解码器沿用原结构，没有新增模块。原YNet核心随机初始化，未读取作者sdd_trajnet预测权重；仅加载作者分割权重。3个smoke更新后重置到同一随机核心和空Adam，再开始完整测量，不能把smoke访问算进一次无重复遍历。

原损失严格为L=1000*mean_BCE(goal12,GT_gaussian12)+1000*mean_BCE(path12,GT_gaussian12)。所有像素/time/目标取mean，两个分支权重1；GT Gaussian沿用normalize=False、kernlen31、nsig4。path训练按GT第11索引目标距离图teacher forcing，合法训练监督，不是部署泄漏。推理函数没有future参数，改future后原history相同、预测相同；本轮不算ADE/FDE。

训练adapter唯一明确的运行状态修正：net.train后始终segmentation.eval且requires_grad=False；在no_grad下每录像计算一次分割特征。原作者在早期eval计算后切回train，晚期>=unfreeze会在train模式再调用，冻结参数不等于BN冻结；本轮不沿用该隐式BN漂移，也不冒充完全相同训练日程。

checkpoint含模型、Adam一/二阶矩、CPU/CUDA/NumPy/Python RNG、进度和配置；原方法无scheduler，记录None和原因，不丢失一个实际存在的调度器。本阶段无best选择，只有smoke与完整遍历last；恢复后相同下一步参数和Adam状态严格一致={r.get('checkpoint_restore_and_next_step_exact',False)}。不覆盖作者权重或旧checkpoint。

代码/配置训练前冻结commit `{r['started_code_commit']}`；运行源码版本及数据SHA均写日志。
'''
    table='\n'.join(f"|{x['epochs']}|{x['unaugmented_training_only_hours']:.2f}|{x['eightfold_augmented_linear_training_only_hours']:.2f}|" for x in b['estimates'])
    docs['04_TRAINING_RESOURCE_AUDIT.md']=f'''# 真实完整遍历与资源预算

运行状态：**{r['status']}**。完整遍历={r['complete_pass']}；实际{r.get('full_pass_recordings')}录像、{r.get('full_pass_target_visits')}目标访问、{r.get('full_pass_unique_targets')}独特目标，{r.get('full_pass_steps')}优化步。无增强，无26scene截断；所有原始目标ID逐一配对。3个smoke共9目标访问、两次恢复探针共8访问单列，不混入完整epoch。

完整无增强一遍实测 **{b['observed_full_pass_seconds']:.3f}s**（{b['observed_full_pass_seconds']/60:.2f}分钟），含scene预处理/分割和循环开销，不含最后checkpoint。训练step wall均值{b['average_step_wall_seconds']:.4f}s，P50/P90/P99={b['median_step_seconds']:.4f}/{b['p90_step_seconds']:.4f}/{b['p99_step_seconds']:.4f}s，剔除前5步均值{b['mean_excluding_first5_steps']:.4f}s。GPU event span总{b['all_train_GPU_event_span_seconds']:.3f}s是流上事件间隔，含可能的CPU提交空档，不称纯kernel计算时间。

加载/resize/pad合计{b['total_loading_seconds']:.3f}s；冻结分割合计{b['total_segmentation_seconds']:.3f}s。每录像尺寸、目标数与耗时见scene_timing.csv，scene目标数量主要影响更新步数，像素分辨率影响每步卷积，不把录像轨迹数误作交互密度。

峰值allocated={b['allocated_peak_bytes']/1024**3:.3f}GiB，reserved={b['reserved_peak_bytes']/1024**3:.3f}GiB；整卡离散采样在training_memory.csv，不冒充高频物理峰值。本机10GB在全部25scene、target batch4下无OOM。不能据此保证batch8或增强旋转后所有尺寸也必定通过。checkpoint约{b['checkpoint_bytes']/1024**2:.2f}MiB，save+SHA {b['checkpoint_save_seconds']:.3f}s；恢复正确性已验证，恢复单独耗时未分离测量。

CPU核验增强8倍：{a['augmented_targets']:,}目标访问、200录像变体、target batch4共{a['target_batch4_steps']}更新。未执行增强GPU训练，旋转尺寸补边可能改变吞吐。原作者坐标采用W/2,H/2连续边界中心，与数组像素中心W-1/H-1有1像素约定差；保持作者实现，不能称精确逐像素配准。原出界点及增强后出界统计保留，不删样本。

|完整epoch数|无增强训练GPU小时（按实测线性）|8倍增强训练GPU小时（外推，未实测）|
|---|---:|---:|
{table}

上述不含正式TTST validation、额外保存、温度/其他进程影响；不是统计置信区间。budget_estimate.json给出无增强0.8–1.3倍、增强0.8–1.5倍规划区间，只是显式假设。没有用Stage11A+六例推理计时外推训练。

26scene在本200变体池可能访问{b['26_scene_cap_target_range']}目标、{b['26_scene_cap_step_range']}更新，随机scene等概率下期望{b['26_scene_cap_expected_target_visits_under_uniform_scene_shuffle']:.2f}目标；这既非每次固定量，也非完整59,968访问。作者原30录像池另为240变体，不可把其300名义epoch与此300完整epoch直接画等号。

短期loss可下降只证明可优化：未做独立验证、未证明收敛或预测性能。任何30/50/100/300轮都只是预算，不是本轮许可。本轮保守GPU额度见training_execution.json及final_gpu_budget.json，未超过3600s。
'''
    docs['05_EVALUATION_PROTOCOL.md']='''# 后续评价规范：公式冻结，正式实验仍未批准

8历史/12未来，0.4s，原SDD参考图像像素。公式实现evaluation.py只在人工数据上测试，不打开真实val/test评价。预测恢复原像素scale后统一计算：ADE1=首候选逐时L2均值；FDE1=该候选完整路径末点L2；minADE20/ minFDE20分别对20条完整路径取最小，各自可来自不同head。

GoalFDE单列为采样目标与GT末点误差，不能用Y-Net官方Goal FDE冒充完整路径FDE。K1首TTST候选对应softargmax预测goal后解码，不是分布数学期望。K20候选预算Ke20×Ka1；TTST10000是proposal，不是10000最终轨迹；CWS关、T1、rel_threshold0.01。未来所有对照使用同样对象、候选数和采样规则。

GT只用于离线oracle误差，不提供在线选模能力。minADE与minFDE不同head的人工反例已测试。推理无future参数，训练期GT teacher forcing不会流入该函数；本轮只执行工程K1依赖测试，无预测指标排名。

未来拟按稳定sample_id产生torch/NumPy采样seed（base17），单独保存随机状态。窗口加权和录像宏平均分开，不能假定共享录像/重叠支持为iid。预测出图不clip后再评价；非有限输出停止诊断，不删难例。语义合规指标待类别/ignore/边界规则可靠后再启用，不把合规等同安全。

未来选模建议：独立开发集最低minADE20、同值取早轮、配对FDE来自同checkpoint；只是方案，必须另行批准冻结训练预算和完整数据协议。本阶段不保存按真实精度选出的best，无测试选模，不将不同输入/单位/预测时长的表格混排。
'''
    matrix=[('A 数据身份与时间',True,'SPLIT_IDENTITY_PASS；录像ID级支持隔离，不等于未知地点'),
            ('B 原损失正确性',success,'原两个BCE×1000，teacher forcing只在训练'),
            ('C 随机预测核心更新',success,'未加载作者预测权重，encoder/goal/path变化'),
            ('D 真实反向',success,'真实训练批次loss/梯度有限'),
            ('E 完整遍历',r.get('complete_pass',False),'全部7,496目标一次，非26scene cap'),
            ('F 预算',r['conservative_total_GPU_charge_seconds']<3600,'10GB batch4实测；正式多轮未批准'),
            ('G 语义类别映射',False,'SEMANTIC_CLASS_MAPPING_UNVERIFIED'),
            ('H 分割预训练曝光',False,'作者声明train-only已取得；当前权重成员/新val曝光未核实'),
            ('I 正式协议冻结',False,'候选仍NOT_READY，训练预算/选模待批准'),
            ('J 独立创新',False,'没有新算法或独立性能证据')]
    cnew(OUT/'go_no_go_matrix.csv',[dict(item=x,passed=y,evidence=z) for x,y,z in matrix])
    tests=ET.parse(OUT/'full_unit_tests.xml').getroot().find('testsuite')
    docs['STAGE11B0_REPORT.md']=f'''# Stage11B-0总报告

**工程训练{'GO' if success else 'NO-GO/未完成'}；正式基线多轮训练仍未获批准；SCI新算法NO-GO。** 用户授权只覆盖本轮≤3600秒工程测量，完成后停止。

1. 从Stage11A+已推送fba0e46建立feat/stage11b0-ynet-training-admission；训练前代码配置冻结`{r['started_code_commit']}`。官方源`{OFFICIAL}`不变，继承隔离venv，没有修改Stage1—11A+。
2. 数据身份审核通过，train7,496/25、val998/5、test2,829/17不变；测试pkl不反序列化。分割语义/生成来源限制继续单列，不冒充完整协议冻结。
3. 官方补充材料新证据确认五stuff+dummy及作者所述train-only分割预训练；SDD编号/ignore和当前checkpoint成员仍未核实，原训练池新val可能有语义曝光。
4. 新adapter沿用原Y-Net随机核心、goal/path损失、GT训练teacher forcing、Adam1e-4/seed17；冻结分割器eval，参数及BN不变；明确目标batch4与scene1，无26scene cap。
5. CPU前置12项通过；首轮临时checkpoint相对路径错误已修正，原始7个setup错误完整gzip保留，未跳过。全仓库{tests.get('tests')}项最终通过，日志见full_unit_tests.xml。
6. 真实smoke3步，完整遍历{r.get('full_pass_steps')}步/{r.get('full_pass_target_visits')}目标，恢复探针2步，总真实optimizer调用{r['total_optimizer_steps']}。完整遍历从与smoke相同随机初态重置开始，单列重复访问，不把探针当新epoch。
7. 整轮实测{b['observed_full_pass_seconds']:.3f}s；参数更新、分割冻结、无GT推理及Adam/RNG恢复后的同一步一致性通过。没有val/test成绩、没有训练ADE/FDE、没有收敛声明。
8. allocated峰{b['allocated_peak_bytes']/1024**3:.3f}GiB/reserved {b['reserved_peak_bytes']/1024**3:.3f}GiB，本次25场景target4均无OOM；不外推为任何batch/增强必可运行。
9. 30/50/100/300完整遍历预算见04与budget_estimate.json。八倍增强访问59,968、200变体的GPU耗时只是由真实训练外推，未在本轮运行；正式验证与额外存储开销未测。
10. 工程GO的证据矩阵A–J见go_no_go_matrix.csv。正式基线计划需审核开发划分、语义曝光声明、选模规则和足额预算。若坚持未知场景语义，需另行无评测标签曝光的分割方案。不能将工程成功称创新。

建议下一阶段先审查上述正式方案和八倍增强训练成本，再决定是否批准有限正式复现；当前不自动启动Stage11B、多seed或300轮。完整数据遍历证明路径可训练，不证明预测能力或论文新颖性。

历史SHA、真实资源/权重/config与最终测试证据在results/tables/stage11b0。最终提交SHA以Git交付为准；原checkpoint与历史结果不覆盖。
'''
    for name,text in docs.items():
        with (folder/name).open('x') as f:f.write(text)
    print('Wrote',len(docs),'reports; engineering',success,'formal multi-epoch not approved')
if __name__=='__main__':main()
