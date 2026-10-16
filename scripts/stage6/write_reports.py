"""Evidence-led research reassessment; no invented performance or new method."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,os,xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

ROOT=Path('results/tables/stage6');D=ROOT/'diagnosis';R=Path(os.environ.get('STAGE6_REPORT_OUTPUT','reports/stage6'));R.mkdir(parents=True,exist_ok=True)
rows=read_csv(D/'per_target_motion_diagnostics.csv.gz');groups=read_csv(D/'all_group_diagnostics.csv');horizons=read_csv(D/'per_horizon_aggregates.csv')
distribution=read_csv(D/'history_distribution_comparison.csv');concentration=read_csv(D/'error_concentration.csv');people=read_csv(D/'per_person_contributions.csv')
execution=json.load(open(D/'execution.json'));sg=json.load(open(ROOT/'public/SGCN_functional_execution.json'));axes=json.load(open(ROOT/'public/structural_supplement.json'))
sing=json.load(open(ROOT/'public/SingularTrajectory_git_fallback.json'));protected=json.load(open('configs/stage6/protected_artifacts.json'))
testroot=ET.parse(ROOT/'full_tests.xml').getroot();suites=testroot.findall('testsuite') if testroot.tag!='testsuite' else [testroot]
tests=sum(int(t.attrib['tests']) for t in suites)

def f(x):return '—' if x is None or x=='' else f'{float(x):.6f}'
def table(head,rr):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rr)+'\n'
def save(name,text):
 path=R/name
 if path.exists():
  assert path.read_text()==text, f'Existing report differs: {path}'
 else:
  with path.open('x') as out:out.write(text)
def choose(rec,seed,model,group='all',weight='target-window'):
 return next((r for r in groups if (r['recording'],r['seed'],r['model'],r['group'],r['weighting'])==(rec,str(seed),model,group,weight)),None)
def values(rec,seed,model,group='all'):
 return [r for r in rows if r['recording']==rec and r['seed']==str(seed) and r['model']==model and group in r['groups'].split('|')]

scope=f'''本阶段新训练run=0，优化器更新=0，ETH评价=0。恢复Stage5.5索引中12组best K1 latent-prior预测，未重新选epoch；逐目标ADE/FDE相对原结果最大差{execution['maximum_error_match_difference_m']:.3e}m（float64诊断与原float32的舍入差，容许2e-6m）。不是重训、不是新增预测成绩。原始完整坐标在被Git忽略的`{execution['trajectory_packet_path']}`，SHA `{execution['trajectory_packet_sha256']}`；Git保存逐目标/逐时间步派生误差、ID、分组与分片索引，不提交原始轨迹数据。

协议仍为HTraj-SocialVAE-Causal8x12-v1：8帧历史、12帧未来、0.4秒、米制。K1 latent-prior不是预测分布数学期望；VAE重建+MC KL不是精确边缘NLL。GT只用于监督目标身份核对及事后误差诊断，`model(x,n)`没有GT参数。不能把未来定义的诊断标签变成在线分组、输入、邻居选择或门控。全部阈值与定义见`configs/stage6/diagnostic_protocol.json`，它们是事后描述规则，不是新独立确认或因果识别。

'''
summary=table(['录像','seed','权重','Full ADE/FDE','Placebo ADE/FDE','ΔADE/ΔFDE'],[
 [r['recording'],r['seed'],r['weighting'],f(r['Full_ADE_m'])+' / '+f(r['Full_FDE_m']),f(r['Placebo_ADE_m'])+' / '+f(r['Placebo_FDE_m']),f(r['delta_ADE_m'])+' / '+f(r['delta_FDE_m'])]
 for r in read_csv('results/tables/stage5p5/analysis/paired_overall_metrics.csv') if r['selection']=='best' and r['metric_family']=='K1-prior'])
whole=table(['权重','seed','模型','N窗口/ID','ADE/FDE','速度预测/真实(m/s)'],[
 [w,s,m,q['targets']+'/'+q['people'],f(q['ADE_m'])+' / '+f(q['FDE_m']),f(q['predicted_mean_speed_mps'])+' / '+f(q['truth_mean_speed_mps'])]
 for w in ['target-window','equal-person-ID'] for s,m in [('analytic','CV-Last2'),('analytic','CV-OLS4'),('analytic','CV-OLS8'),('17','Full'),('42','Full'),('2026','Full')]
 for q in [choose('biwi_hotel',s,m,weight=w)]])
motion=table(['历史组','seed','模型','N/ID','ADE/FDE','预测/真实速度','速度偏差'],[
 [g,s,m,q['targets']+'/'+q['people'],f(q['ADE_m'])+' / '+f(q['FDE_m']),f(q['predicted_mean_speed_mps'])+' / '+f(q['truth_mean_speed_mps']),f(q['speed_bias_mps'])]
 for g in ['speed_stationary','speed_low','speed_normal','speed_fast','turning'] for s,m in [('analytic','CV-OLS4'),('analytic','CV-OLS8'),('17','Full'),('42','Full'),('2026','Full')]
 for q in [choose('biwi_hotel',s,m,g)] if q is not None])
tags=[]
for s in [17,42,2026]:
 for m in ['Full','PlaceboSocial']:
  rr=values('biwi_hotel',s,m);n=values('biwi_hotel',s,m,'speed_normal')
  count=lambda k:sum(r[k]=='True' for r in rr)
  tags.append([s,m,len(rr),count('tag_sustained_future'),count('tag_sustained_but_predicted_deceleration'),count('tag_history_stationary_predicted_moving'),count('tag_future_stationary_predicted_moving'),count('tag_direction_error'),count('tag_displacement_undershoot'),count('tag_displacement_overshoot')])
tagtable=table(['seed','模型','N','未来持续','持续却预测减速','历史静止→预测动','未来静止→预测动','端点偏向≥30°','位移<0.5GT','位移>2GT'],tags)
time_table=table(['seed','模型','时间(s)','N','位置误差','预测速度','真实速度','向量速度误差'],[
 [r['seed'],r['model'],f(r['horizon_seconds']),r['targets'],f(r['error_m']),f(r['predicted_speed_mps']),f(r['truth_speed_mps']),f(r['velocity_vector_error_mps'])]
 for r in horizons if r['recording']=='biwi_hotel' and r['weighting']=='target-window' and r['model'] in ['Full','CV-OLS4','CV-OLS8'] and r['step'] in ['1','6','12']])
id_table=table(['seed','模型','Top3误差ID','Top3占ADE总误差','Top3正超额误差ID(对OLS4)','正超额份额'],[
 [r['seed'],r['model'],r['top_ADE_IDs'],f(r['ADE_share']),r['positive_excess_OLS4_IDs'],f(r['positive_excess_OLS4_share'])]
 for r in concentration if r['recording']=='biwi_hotel' and r['model'] in ['Full','PlaceboSocial'] and r['top_n']=='3'])
id_detail=table(['seed','ID','窗口','ADE/FDE','预测/真实速度','对OLS4有符号ADE误差总量'],[
 [r['seed'],r['target_id'],r['windows'],f(r['ADE_m'])+' / '+f(r['FDE_m']),f(r['predicted_mean_speed_mps'])+' / '+f(r['truth_mean_speed_mps']),f(r['signed_excess_OLS4_m'])]
 for r in people if r['recording']=='biwi_hotel' and r['model']=='Full' and r['target_id'] in ['356','361','369']])
orientation=table(['seed','历史朝向','窗口/ID','ADE/FDE','ΔADE对OLS4','小样本'],[
 [s,d,'0/0' if q is None else q['targets']+'/'+q['people'],'—' if q is None else f(q['ADE_m'])+' / '+f(q['FDE_m']),'—' if q is None else f(q['delta_ADE_vs_CV-OLS4_m']),'True' if q is None else q['small_group']]
 for s in [17,42,2026] for d in ['E','NE','N','NW','W','SW','S','SE','unreliable'] for q in [choose('biwi_hotel',s,'Full','heading_'+d)]])
distro=table(['留出录像','split','目标窗口/人ID','历史均速','历史静止比例','正常行走比例','朝向不可可靠判定'],[
 [r['fold'],r['split'],r['targets']+'/'+r['people'],f(r['speed_mean']),f(r['stationary_fraction']),f(r['normal_fraction']),f(r['heading_unreliable_fraction'])] for r in distribution if r['source']=='all'])
save('01_STAGE5P5_FAILURE_ANALYSIS.md','# Stage5.5最终负结果与HOTEL误差诊断\n\n'+scope+'''## 已结束的旧社会内容路线

Full相对Placebo的K1优势未通过跨录像、跨seed、两种权重的冻结规则。ZARA2两权重均3/3改善；HOTEL窗口加权2/3、行人等权1/3，ADE/FDE方向都反转。保留ZARA2条件性收益，不以均值或单组收益挽救跨录像失败。终止在TOHI/G0/G1/SocialVAE旧社会内容路线增加GNN、社会注意力、关系GRU、可靠性门控与高阶模块。

'''+summary+'## 同目标强CV比较：HOTEL\n\n'+whole+'## 历史运动状态\n\n'+motion+'''历史静止<0.05、低速<0.4、正常<1.8m/s，其余高速，沿用Stage5.2。HOTEL历史正常行走91窗口/24ID中，GT未来平均1.213391m/s，Full三seed分别约0.952812/0.907931/0.837950m/s；OLS4约1.213988m/s。直接支持“该群体K1速度偏低”，不是证明其训练机理。历史静止106窗口只有4ID；Full平均预测速度也低于该组GT均速，因此不能把全部失败概括成“静止误判运动”。

## 事后轨迹形态标签：与历史分组分开

'''+tagtable+'''未来持续标签：GT平均速度≥0.4m/s且末3步均速≥前3步的75%；预测减速同时需前3步预测≥0.2、末3步低于其70%且低于GT末3步70%。HOTEL有78个持续目标，Full符合复合减速条件34/34/71；这些目标不能用来训练任何选择器。0/1/0个“历史静止却预测均速≥0.2m/s”说明阈值下的严重假运动并非主导现象；小幅漂移仍能显著高于强CV。未来静止标签只用于事后结果分析，不能代替历史静止标签。

端点角误差只在预测和GT净位移均≥0.2m时定义；其余不可判定，不填0。幅度异常只在GT位移≥0.5m后比较0.5倍/2倍阈值，是严重幅度标签，未命中不代表不存在温和收缩。标签可重叠，不能相加当独立原因占比。

## 预测时域与速度误差

'''+time_table+'''Full在0.4秒误差约0.07–0.08m，到4.8秒约0.75–0.88m；OLS4/8相同终点误差约0.44/0.41m。时间增长与末端速度偏差共同支持累积运动幅度/方向误差描述，不证明所有误差都由减速造成。每目标12步数据完整保存于`diagnosis/step_shards/`，不只选择终点。

## 初始历史方向与终点分解

'''+orientation+'''这里的初始方向是预测起点处可观测历史净位移方向x7−x0；净位移<0.2m或历史均速<0.2m/s标不可判定，不用未来方向。八扇区含空组完整列出；全局朝向与录像几何、速度和ID共变，不能直接当作旋转不等变的因果证据。可靠目标保存端点沿历史方向/垂直方向有符号误差，二者平方和等于FDE平方；没有可靠历史方向的目标保留缺失。`per_target_motion_diagnostics.csv.gz`还保留预测与GT的端点角和净位移。

## 按ID的误差集中度

'''+id_table+id_detail+'''Full误差最大的3个ID三seed均为356/361/369，占窗口加权总ADE约24.6%–33.3%；不是“全部误差来自极少数人”。对OLS4正超额误差的Top3份额仅约22.4%–28.5%，且ID随seed变动；有符号正负差另存，没有删掉改善目标。按人等权后Full仍差，故不能通过剔除难ID解决主结论。ID窗口不独立，报告贡献而非大量独立显著性。

## 训练域与验证域

'''+distro+'''HOTEL训练来源历史静止约17.1%、正常约60.2%，验证约33.3%/28.6%；历史方向可靠性也明显不同。相同录像内的窗口构成影响均值，原始源录像逐一统计见`history_distribution_comparison.csv`。这是分布差异证据，不是“它导致速度收缩”的证明：训练整体均速反而更高，不能用简单均速迁移解释所有正常行走的偏慢预测。未执行同模型分布匹配反事实、目标域干预或重训练。

## 证据等级

直接观测：时间误差增长、正常行走预测速度偏低、条件性方向偏移、小幅静止漂移、有限ID集中、来源运动构成变化。相关/并存：早期best和后期验证退化、分布差异与HOTEL泛化失败。待验证：训练混合分布、VAE后验训练与prior K1统计不一致、场景几何与方向表征各自的因果作用。没有证据指定唯一失败机制。

全部神经值是固定K1轨迹诊断，不等价于整个VAE预测分布无价值；K20不能替代K1解释。Stage5.5完整曲线及其曝光/28和35个ID限制继续有效。源：Stage5.5配对表/冻结权重，Stage6的diagnosis/execution.json与全量派生表。
''')

sgcommit=sg['repository_commit'];singcommit=sing['commit']
sgurl=f'https://github.com/shuaishiliu/SGCN/tree/{sgcommit}';sturl=f'https://github.com/InhwanBae/SingularTrajectory/tree/{singcommit}'
sgblob=sgurl.replace('/tree/','/blob/');stblob=sturl.replace('/tree/','/blob/')
cases=table(['权重','N','forward','高斯参数最大差','均值路径最大差(m)','空间邻接恢复差','单位图跨人Jacobian','错误'],[
 [r['fold'],r['people'],r['forward_success'],f(r.get('raw_gaussian_parameter_max_difference')),f(r.get('mean_trajectory_max_difference_m')),f(r.get('learned_spatial_adjacency_max_difference')),f(r.get('other_node_Jacobian_L2_with_identity_graph')),r.get('error_type','')]
 for r in sg['cases']])
save('02_PUBLIC_GNN_BASELINE_REVIEW.md',f'''# 公开GNN与补充模型复核

## SGCN：版本与正式来源

[CVPR2021正式论文](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html)，Liushuai Shi等，8994–9003，[arXiv2104.01528](https://arxiv.org/abs/2104.01528)。作者仓库[冻结完整历史发布]({sgurl})，当前HEAD `0d4ac44f44a0688a839f25070be1a4faeed5ade1`已移除主要代码，不能把当前main称为可直接训练。MIT许可证；固定Git blob与下载字节SHA逐文件核验。未修改作者源码。

25,369参数，两份HOTEL/ZARA2 val_best各97个state键严格加载。只在RTX3080进行人工N=1/2/3/5/8/12前向，没有真实场景ADE、没有ETH评价、没有训练。官方非ETH权重可能使用ETH作为上游训练来源，不能用于HTraj严格非ETH公平性能结论。

SGCN有显式稀疏有向空间与时间关系，输入8帧逐步位移加时序位置通道，输出12帧每人5个二维高斯参数。空间图[8,4,N,N]，时间图[N,4,8,8]，图卷积两条空间→时间/时间→空间路径；最终共享时序预测。复杂度主要含O(TN²d+NT²d)，图关系/掩码存储O(H(TN²+NT²))，不是稀疏参数就自动线性复杂度。来源：[model.py]({sgblob}/model.py)。

## 实际有限测试：成功与失败都保留

'''+cases+f'''原始高斯参数差是参数单位；人工历史以米构造，均值位移累加路径差才以米表示，但它不是实测场景误差。部分HOTEL人数例子差为0，不能掩盖N=2反例或ZARA2多个N反例。N=1由`squeeze()`引起rank断言；官方loader本来要求人数大于min_ped=1，因此它是单人接口限制，不自动等于论文宣称支持单人却失败。

**修正Stage5静态推测：SGCN的TXP输入实际为[N,8,4,16]，节点在batch轴。** 独立TXP换序恢复差和其他节点Jacobian均为0。它不沿节点索引直接卷积；不得把Social-STGCNN的问题原样套在SGCN上。空间mask的输入[8,4,N,N]确实沿N×N执行3×1/1×3卷积，学出的邻接不满足同步置换，完整网络也出现换序差。旧报告不改写，新增`structural_supplement.json`记录纠正。单位空间/时间图干预下N=3跨人Jacobian为0，只是结构检查，不是已训练NoSocial性能。

## 数据与训练准入问题

`utils.py`要求整20帧完整行人才进入窗口和上下文，人数≥2；它不是Stage5.1只按历史选择邻居的因果协议。位置读入再差分，输入只用历史，但未来存在性筛选上下文仍是协议差异。官方loader以frame索引序列取样，原始0.4秒/米制需要文件级审计，不能仅凭8/12认定相同。

`train.py`有完整从头训练入口，默认300epoch、batch128窗口累积，实际使用Adam(lr=0.01, weight_decay=0.0001)、MultiStepLR([50,100], gamma=0.5)、Gaussian NLL；解析器另有momentum参数但未用于实际优化器、按val_loss选权重；test使用Gaussian采样，不等于K1 latent-prior。**源码边界累积分支风险：else分支没有把当前窗口的graph_loss加入累积loss，却直接除batch_size并更新；边界窗口可能漏监督。** 这是可定位静态问题，未在本阶段运行训练重现；不能据此宣布所有公开训练都无效。尾批缩放、单人squeeze与NumPy/旧依赖均须未来单测。引用：[train.py]({sgblob}/train.py)、[utils.py]({sgblob}/utils.py)、[test.py]({sgblob}/test.py)。

作者README列Python3.6.2/Torch1.2.0/NumPy1.19.0/2080Ti。当前Torch2.5.1+CUDA12.4、RTX3080严格加载及人工前向可行，完整旧训练环境、收敛、吞吐未验证。不能用“前向成功”代替论文复现。由于排列与训练边界问题，本轮不把SGCN升级为即用的第四章基础，不自动修模型。

## Social-STGCNN旧实验证据复核

CVPR2020论文与本地维护版commit `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`分开。Stage4.1固定官方维护权重在64个非ETH窗口/235目标，置换后恢复的平均位置差0.266736m，最大1.917565m，ADE0.521069→0.567688；差异首次出现在TXP3×3节点轴卷积，非ST-GCN。原Full与NoSocial还曾有BN/PTN归一化差异，不能单因素归因。

Stage4.2受控G0/G1改为相同归一化、1553参数、共享逐人decoder与同初始化；最终同预算到100轮，G1−G0 ADE ZARA2 +0.091252、HOTEL −0.046452m，两者均弱于同目标三CV。PTN毫米扰动风险有固定权重证据，未作移除PTN因果训练，不能指定唯一失败原因。来源：`reports/stage4p1/03_PERMUTATION_AUDIT.md`、`reports/stage4p2_final/05_FINAL_RESEARCH_DECISION.md`及各原始CSV。本轮没有重训或篡改旧指标。

## 唯一补充候选：SingularTrajectory（源码为主）

[CVPR2024正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.html)与[作者仓库]({sturl})对应，固定commit `{singcommit}`，27个源码/配置文件Git blob及SHA核查通过。它比SGCN更新，提供8/12配置、训练入口、公开release下载脚本，运动/静止两套Singular表示、历史原点/方向/尺度、原型与扩散已是原方法。**这些内容不能作为HTraj重新命名的创新。**

源码是Transformer场景mask交互，不是替换SGCN的简单显式空间GNN。默认样例K20、batch512、256epoch、AdamW、10步去噪，按验证FDE选择；SVD/原型以train的obs与future初始化，未来标签仅训练侧拟合本身允许，但必须审计只用source train。官方完整上下文仍离线筛选。额外依赖场景图、homography与vector_field，`script/generate_vector_field.py`使用地图图像；与HTraj纯历史坐标协议不等价。domain adaptation任务不能冒充未接触目标域的domain generalization。许可为CC BY-NC4.0，不是MIT。

只做了零速/低速/正常速度的归一化组件检查，没有完整权重加载/完整模型前向/数据评价。裸尺度归一化零速会Inf，但实际静止分支关闭尺度，三个人工样本均有限并能恢复；不能据裸子模块Inf声称整网失稳。CUDA硬编码、依赖与场景资产、数据未来筛选、不同任务配置完整性和RTX3080完整训练显存/耗时均未验证；示例配置可见不代表release全部可复现。公开权重仅定位URL，未下载/核验，不报ADE/FDE。

来源：[model.py]({stblob}/SingularTrajectory/model.py)、[trainer.py]({stblob}/utils/trainer.py)、[dataloader.py]({stblob}/utils/dataloader.py)、[config]({stblob}/config/config_example.json)、[LICENSE]({stblob}/LICENSE)。由于地图侧信息、扩散成本和当前协议不匹配，不作为本轮主基础，也不盲目调试。

## 实际状态对照

|对象|显式图|8/12|从头入口|权重|本轮实际执行|公平消融与协议风险|
|---|---|---|---|---|---|---|
|SGCN历史版|空间/时间有向图|是|有；边界累积待验证|2份strict成功|人工前向/排列/Jacobian|空间mask索引敏感、完整20帧上下文、有效容量|
|Social-STGCNN维护版|社会图+节点卷积|是|旧阶段真训练|旧阶段strict/评价|本轮复核冻结记录|旧Full/NoSocial多因子；G0/G1受控失败不可直接当官方|
|SingularTrajectory|Transformer场景关系|示例是|有|release可定位，未验证|仅归一化组件|地图与train-only表示拟合、K1/K20、计算预算|
|冻结SocialVAE-NoSocial|完全无真实邻居|是|Stage5.3真训练|自训练best/last冻结|本轮引用旧结果|仅seed17；有效容量与Full不同，不用来证明社会收益|

检索MCP未挂载；OpenAlex429、GitHub REST403均保留日志，使用CVF/arXiv/CrossRef及作者Git原始文件降级核查。一次本地目录重复创建的工程失败日志保留，修正后独立v2日志成功；没有隐去或填造模型结果。聚焦审查不是穷尽综述，也不提供新颖性认证。
''')

save('03_RESEARCH_PROBLEM_COMPARISON.md','''# 研究问题比较：证据与覆盖程度

## 全阶段证据链及适用边界

|阶段|可核查事实|当前影响|限制/原始来源|
|---|---|---|---|
|Stage1|HighGraph/UniEdge等已研究高阶与社会结构|高阶/时间记忆不是空白命题|docs/literature/；不把结构不同自动当创新|
|Stage2.6|H3对H2三seed ADE都差，H2独立稳定收益不足|旧高阶路线停|reports/stage2p6/STAGE2P6_REPORT.md；partial小开发协议|
|Stage3|Receiver-Fix/关系记忆未通过预定筛选|有效接收不等预测价值|reports/stage3/06_RESEARCH_DECISION.md；不外推所有seed|
|Stage3.5|完整/部分协议与录像隔离明确|先对齐任务而非跨协议排名|reports/stage3p5/STAGE3P5_REPORT.md|
|Stage3.6|冻结ego后社会支路仍未胜同容量个体修正|不能只归因联合训练破坏ego|reports/stage3p6/06_RESEARCH_DECISION.md；非完全同结构联合反事实|
|Stage3.7|原主干已平移等变；方向归一化只改善HOTEL|不能再用去绝对坐标包装新消融|reports/stage3p7/05_RESEARCH_DECISION.md|
|Stage3.8|OLS4/8/Median只在HOTEL改善，其他4录像退化|停止修CV估计器和可靠性gate|reports/stage3p8/05_RESEARCH_DIRECTION_DECISION.md|
|Stage4.1/4.2|节点索引伪关系、归一化混杂被定位，公平图对照跨录像失败|代码可靠性必要但不保证社会收益|原离线完整上下文不能与本轮因果协议混排|
|Stage5–5.4|公开VAE可真训练、因果loader、NoSocial与容量/Placebo对照|成熟基线比原型可靠；单seed迹象非定论|各冻结报告；不同控制函数类/分布不等价|
|Stage5.5|Full vs Placebo HOTEL跨seed/权重反转，ZARA2重复|结束由毫米收益驱动的复杂社会改造|3seed不是3个新数据集，两曝光录像|
|Stage6|冻结K1出现HOTEL运动速度收缩、方向/ID/时间误差可分解|B问题的直接可观测对象更明确|仅描述，不指定唯一原因；不训练新模型|

不能把不同时期的partial、A_offline_complete20、Causal8x12数值串成同一纵向排行榜；它们贡献的是不同层面的证据。当前结论也不是“所有社会信息无价值”：ZARA2有条件收益，但旧路线要求的跨录像稳定性未成立。

## A与B逐项判断

|维度|A：图神经网络社会交互|B：跨场景运动状态与预测泛化|
|---|---|---|
|本项目正向证据|有计算依赖和ZARA2条件收益；无稳定全局增益|多阶段神经HOTEL弱于CV、best很早、固定输出速度偏差明确|
|技术问题可观测性|需先修合法关系、排列、容量和负对照；可辨识性成本高|可量化逐步速度、沿/横误差、状态比例和泛化gap|
|现有文献覆盖|稀疏图/方向图/高阶/社会注意力已广泛研究|运动先验、域对齐、归一化、静动分支也已有研究，绝非空白|
|尚可证伪的不足|在排列正确、预算和容量控制后，真实关系何时有独立收益|只用source train历史定义的状态分布控制，能否减少稳定速度收缩且保留其它录像性能|
|硕士题目适配|符合复杂场景，但旧实验不足支撑主章贡献|跨录像速度/方向与运动状态变化属于复杂场景，符合总题目|
|有限时间的小论文机会|当前低：需重建可靠图基础，且旧收益未重复|相对更可行：已有可审计诊断，但仍缺新方法/独立验证，不能承诺可投稿|
|主要反证|动态图、关系记忆、高阶与公平G0/G1多次不稳定|归一化和长历史CV也已失败，不能把B改名为旧速度gate继续试错|

**优先B，A仅保留为备用科研问题。不是直接批准两个方法开发。** B的实验支持是“存在必须解释的泛化问题”，并不是“某个适应模块已有效”。不要把描述性诊断提升为方法创新或独立因果证据。

## 文献占位与新颖性风险

- [SGCN，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html)：稀疏空间/时间图已存在；本轮源码还暴露排列和训练边界问题。
- [Adaptive Trajectory Prediction via Transferable GNN，CVPR2022](https://arxiv.org/abs/2203.05046v2)：跨域图迁移与域对齐已有研究；应区分是否访问目标域资料。
- [Towards Single Source Domain Generalisation...，CoLLAs2023](https://proceedings.mlr.press/v232/huang23a.html)：CV运动先验加神经残差、source-domain泛化已存在；普通CV残差不能作为HTraj创新。
- [SingularTrajectory，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.html)：统一运动表示、静动分支、尺度方向处理、adaptive anchor已有实现。domain adaptation任务与无目标域资料的泛化须分开。

上述为聚焦检索，不足以量化哪一方向“论文数量更多”；没有穷尽检索就不编造覆盖率或新颖性分数。A/B均拥挤，当前只是B有更明确本地失效测量。后续若提出具体方法，必须进一步对该技术命题检索，并通过额外非测试录像/数据确认。当前没有一个已完成的、具备可信新贡献和独立验证的小论文方法。
''')

direction=dict(primary=dict(name='B: source-only motion-state distribution and cross-recording prediction generalization',
 baseline='Frozen Stage5.3 SocialVAE-NoSocial implementation, not another social redesign',
 inputs='Eight observed target positions in meters; history-only velocity/mask; no real/synthetic neighbors or future',
 outputs='Twelve future 2D positions; deterministic latent-prior K1 primary, fixed-noise K20 separate',
 hypothesis='Source-history state composition contributes to speed shrinkage; a source-only sampling-factor counterfactual may reduce bias without architecture/loss/gate change. Unverified.',
 first_factor='Uniform original source sampling versus pre-frozen history-motion-stratified source sampling, same total sample visits and optimizer steps; report repeat/coverage; never use future errors',
 future_minimum_runs=4,future_seed=17,future_max_epochs_per_run=30,smoke_first_epochs=5,
 future_validation=['biwi_hotel','crowds_zara02'],requires_separate_authorization=True,
 stop='Stop if both-recording window/person K1 criterion or paired FDE safety fails; never restart high-order/gate search'),
 backup=dict(name='A: explicit social graph attribution on a reliable public baseline',baseline='SGCN5153702d8ae32a2b0586b093d4e094aee08b0f9b, audit reference only until permutation/training gates repaired and verified',
 immediate_eligibility=False,training_authorized=False,stop='Do not start unless node-order, gradient, loader, loss accumulation, capacity and same-target protocol gates pass; no old graph/module stacking'),
 rejected_as_immediate_primary=['current TOHI/G0/G1 structural variants','SocialVAE social architecture redesign','SingularTrajectory without map/protocol/compute audit'],
 current_stage_new_training_runs=0,ETH_evaluated=False,claimed_new_method_improvements=False)
direction_path=ROOT/'research_direction.json'
if direction_path.exists():
 assert json.load(open(direction_path))==direction, 'Existing direction metadata differs; preserve and investigate'
else:
 json_new(direction_path,direction)
save('04_CHAPTER4_METHOD_DIRECTION.md','''# 第四章技术方向：唯一主线与唯一备用

## 主要方向B：仅来源训练的运动状态分布控制与预测泛化

**研究问题：** 在训练采样和梯度不接触验证/测试未来、不加社会模块、不换decoder/loss的条件下，来源训练运动状态的构成与优化选择，是否与HOTEL持续运动预测的速度收缩有关？是否能在保留ZARA2性能的同时减轻该误差？这比“设计一个可靠性门控”更可证伪，目前没有答案。

**基础模型：** Stage5.3严格SocialVAE-NoSocial冻结实现，公共SocialVAE源码仍为eb23cbae562bac2c2dcf0d81529e6ab2ac280962。它是已真训练、隔离邻居路径的个体VAE对照，不是新的论文网络；不再改SocialVAE社会模块。现成NoSocial仅seed17，不能冒充三seed成熟泛化主干。Full/Placebo三seed和三个CV全部保留为冻结诊断参照，CV-Last2为固定主基线、OLS4/8逐一展示，禁止事后按录像挑速度估计器部署。

**输入输出：** 目标8帧米制历史及可观测运动状态，邻居完全关闭；预测12帧2D位置。主指标K1 latent-prior ADE与同checkpoint FDE，固定ID采样的首样本与K20另外报告，不把确定性轨迹说成数学期望。

**第一项待验证假设与最低预算：** 不先新造模型。若另行授权，只比较原均匀来源训练与一个预冻结的“历史运动状态分层来源采样”策略；不使用录像验证误差、未来标签或GT事件给采样器信息。相同源码/损失/参数/初始化/总训练访问次数/优化步数，记录重复和独特目标覆盖，不能把更多有效训练量冒充新方法。历史分组保持固定阈值，不能因为HOTEL结果改分箱。

两个固定LORO开发录像×两个训练采样条件×seed17，共4run，每run最多30轮，前5轮为该run自身smoke而非另加5轮；新旧两个条件都从同seed随机初始化，不将旧100轮best直接与新30轮作公平归因。若采样改变epoch含义，应以相同优化步和样本访问数为预算定义，并同时记录原始训练池覆盖。其余原验证区域仍不进训练。验证GT仅依预设规则选checkpoint和报告；两个开发录像既已曝光又参与选择，故本方案不是严格从未接触目标域的最终泛化证明。用现有SocialVAE旧100轮约35–66分钟的实耗粗估，4×30轮约42–79分钟量级，含I/O/评价保守安排1–3 GPU小时；是历史推算，不是新模型实测。开始前用5轮实测校准，单阶段硬预算3 GPU小时，不无限延长。当前未授权或执行这些训练。

分层采样会改变有效训练分布、经验风险权重和优化噪声；即使损失函数代码不变，也不称“完全相同训练分布”。这个反事实测试的是整个采样策略，不会独立证明某一个分布统计量或运动偏差是唯一因果中介。

**正式停止规则（作为下一阶段建议，需下一指令冻结）：**

1. 任一非有限值、目标/GT不一致、邻居/未来泄漏或预算不匹配，停止并诊断，不调参搜索。
2. 两录像、窗口与行人权重的ADE均需改善至少max(0.005m,基线ADE的1%)；FDE退化不超过max(0.005m,1%)。同一规则选择最低窗口K1 ADE，配对同epoch FDE。不能用宏平均覆盖某录像失败。
3. HOTEL正常行走的平均速度偏差绝对值须下降至少10%，且两录像历史静止组FDE退化不超过max(0.01m,5%)；全部运动/方向组仍报告。阈值是未来预注册开发门槛，非显著性。
4. 不满足即停止该采样解释，不转去加gate/关系GRU/GNN。即使通过，也仅支持一个训练因素；若仍显著弱于强CV，不能宣布已得到可信第四章主干。
5. 通过后最多建议另阶段多seed及额外非ETH资料确认，不在此自动追加。所有现有五非ETH录像均有历史曝光，不能重新命名为独立确认；ETH继续封存。

**可能的论文贡献与当前差距：** 可争取可复核的跨录像失效诊断、运动状态条件的泛化机制证据和真正超过强基线的简洁训练/表示方法。当前只有诊断，均匀/分层采样、CV残差、归一化、GRU或VAE本身都不是创新。若没有新颖技术和独立复现证据，最多形成内部技术报告，不能承诺可投稿。训练构成差异不等于已证明收缩的原因，源均速更高这一反证必须保留。

## 备用方向A：显式社会关系的可辨识价值

基础审计对象只保留SGCN历史5153702d8ae32a2b0586b093d4e094aee08b0f9b；输入同8帧观测、历史图，输出12帧二维高斯及K1均值/K20，未来若比较应重新以相同监督集合训练。**当前尚不具备启动资格**：空间mask排列失败、loss累计边界与离线上下文筛选待修复/验证。修复属于基线正确性，不能作论文创新。不得将备选写成已胜过CV的模型。

待验证问题只限“在排列正确、合法历史图、匹配自身传播和有效容量后，真实关系是否优于无社会和合法内容负对照”。最低先做0训练结构准入；准入通过且另行授权后，才提2录像×2严格匹配图条件×seed17、各5轮smoke/最多30轮的预算。任一准入失败、不胜强个体/容量控制、跨录像方向不一致则停止，不再叠高阶/注意力/GRU/可靠性gate。旧TOHI/G0/G1/SocialVAE社会改造仍终止，不因“备用”重新开启。

## 章节安排与风险

大论文总题目《面向复杂场景的行人轨迹预测关键技术研究》可容纳跨场景运动状态与泛化。建议第四章暂改为“面向跨场景运动分布变化的行人轨迹预测方法”，只作研究计划，不改写被冻结的大纲或声称已有正式新方法；第五章不启动。A符合原社会主题，但证据不足所以仅备用。

当前风险：两曝光录像/少ID、未来事件只能用于诊断、VAE主K1与训练目标不一致、NoSocial只有单seed、采样平衡是常规方法、源/目标运动类型与场景几何混杂、地图条件公开模型无法直接适配纯轨迹协议、补充数据尚未取得。B更有本地问题证据，A/B均无可承诺小论文的新方法。本阶段0训练、0新增网络、0ETH评价，到此停止。
''')

save('STAGE6_REPORT.md',f'''# Stage6：研究方向重审

**主要方向B：跨场景运动状态与预测泛化；备用方向A：严格结构准入后的显式社会图归因。** 两者是科研问题选择，不是批准新模块或宣布性能提升。旧TOHI/G0/G1/SocialVAE社会内容结构迭代正式终止；不再增加GNN、社会注意力、关系GRU、可靠性门控或高阶模块。

Stage5.5 Full对Placebo：HOTEL窗口ADE/FDE仅2/3 seed改善、行人权重仅1/3；ZARA2两权重均3/3。故毫米级优势未通过跨录像稳定性要求，不能凭均值继续加模块。完整数值保留于01及原Stage5.5报告，未重写旧结果。

'''+scope+whole+'''## 失败描述与不能证明的事

HOTEL正常行走91窗口/24ID，Full预测速度0.84–0.95m/s，真实约1.21m/s，持续运动78窗口中预测复合减速34/34/71个。严重历史静止→运动仅0/1/0，不能把整个失败归为假运动。Full最坏三个ID占总ADE24.6%–33.3%，而非全部误差。错误随预测时域增大，部分端点方向偏移与运动幅度收缩并存。训练/验证历史状态和方向构成不同，但尚无控制训练反事实，无法证明唯一原因；源平均速度更高而正常组预测仍慢，也反对简单均速解释。01报告保留全部ID、八朝向含空组、各时域和三CV。

## 公开基础核查

'''+f'''SGCN冻结 `{sgcommit}`，25,369参数、两份非ETH权重97键strict成功；人工N=2–12可前向，但部分换序后预测改变，单人rank失败。**空间mask索引问题成立，旧静态报告关于SGCN TXP也沿节点索引卷积的猜测被纠正**：其输入[N,8,4,16]，节点是batch轴，独立TXP排列/跨人Jacobian均为0。源train.py还有边界窗口loss未纳入累计的待复现风险。因结构/训练/上下文协议问题，不立即采用SGCN开发新方法。

Social-STGCNN维护版的节点卷积问题和100轮受控G0/G1双录像失败继续有效，不能用它证明所有图网络无价值。补充SingularTrajectory冻结 `{singcommit}`，仅源码和归一化组件检查，未做全模型复现；静动表示/归一化已是已有思想，地图vector field、额外依赖及K20/验证FDE选择与当前协议不一致，不贸然替换基线。

## 唯一推荐与实验边界

主线先以已冻结SocialVAE-NoSocial作为个体VAE基础，三CV及Full/Placebo为诊断参照；问题是来源历史状态构成是否影响运动持续性与泛化，不是继续修社会支路。下一阶段若批准，仅建议单一采样因素的4组seed17、最多30轮公平对照，5轮smoke包含在预算内；阈值、强CV与停止规则见04，当前未训练。备用SGCN需先结构/训练准入，当前不满足。没有已具备新颖方法和独立验证的小论文，B只是比A有更充分的可测失效证据。

普通平衡采样、静动分支、CV残差、归一化和常规GNN不构成新创新。已有运动先验域泛化与迁移图工作见03和literature/sources.json。两个方向都必须区分仅来源梯度训练、目标开发集选择与真正未见目标域泛化，以及访问目标场景图/数据的适应；不能把旧曝光录像叫独立确认，也不把未来定义的失败标签给在线模型。

## 交付与验收

五份报告01失败诊断、02公开基线、03问题比较、04主/备方向和本总报告。派生结果、source commit/SHA、数据manifest SHA、strict加载、全部人工测试（含单人失败）、网络失败/降级记录与分片逐步误差位于results/tables/stage6/；完整坐标和外部源码/权重在忽略cache，未上传数据/权重。

当前完整单元测试{tests}项通过；保护清单覆盖{len(protected['files'])}个历史文件，最终SHA复核记录见delivery_verification.json。本阶段实际新训练0、优化器更新0、ETH评价0，旧源码/协议/指标/checkpoint均未覆盖。仅本地commit，不自动push。

用户自行：`git push -u origin feat/stage6-research-reassessment`。本阶段完成后停止。
''')
print('Five Stage6 evidence reports generated')
