"""Freeze protocol first; write evidence reports only after four actual runs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

ROOT=Path('results/tables/stage6p1');REPORT=Path('reports/stage6p1')


def number(value):return '—' if value is None or value=='' else f'{float(value):.6f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'


def save(name,text):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/name).open('x') as out:out.write(text)


def freeze():
    plan=json.load(open('configs/stage6p1/protocol.json'));init=json.load(open(ROOT/'initialization_check.json'))
    cohorts=json.load(open(ROOT/'cohort_verification.json'))
    save('01_FROZEN_PROTOCOL.md',f'''# Stage6.1冻结训练协议

仅检验一个来源训练采样因素，无新模型、无新loss/decoder/归一化/GNN/社会注意力/门控/运动残差。基础提交af7da3ffda0d65ec8ff1e05a5d0235d62cd53dad，新分支feat/stage6p1-motion-sampling-pilot。冻结源码/配置先本地提交再训练；精确训练代码commit由各run/provenance.json记录，避免循环自SHA。

Stage5.3 SocialVAE-NoSocial实现原样复用，官方源码commit eb23cbae562bac2c2dcf0d81529e6ab2ac280962不改动。名义参数{init['parameter_count']}，结构有效上界{init['structurally_effective_scalar_upper_bound']}；两条件完全相同架构、权重初始化及初始化后Torch RNG。保留原GRU、潜变量、decoder和重建+MC KL目标；不能把该loss称为精确边缘NLL。全部run从随机权重开始，不用任何历史checkpoint初始化。无邻居输入路径的既有隔离测试继续适用。

数据HTraj-SocialVAE-Causal8x12-v1：8帧历史、12帧未来、0.4秒、米制。HOTEL与ZARA2分别LORO；训练只使用原manifest的来源train区域，既不包含留出录像，也不提升旧validation。监督目标资格沿用冻结8/12完整性，类别计算只接收8帧历史位置。ETH封存；两个录像均已曝光且用于选择，不是独立确认或严格未见目标域结论。

'''+table(['留出录像','split','目标窗口/行人ID','录像来源','manifest SHA'],[[r['recording'],r['split'],str(r['target_windows'])+'/'+str(r['person_IDs']),','.join(r['records']),r['manifest_sha256']] for r in cohorts])+f'''
四run：HOTEL Uniform、HOTEL Stratified、ZARA2 Uniform、ZARA2 Stratified；seed17，Adam lr0.0005、wd0、batch128、无scheduler/clip/augmentation。各30轮，前5轮是本run内smoke，绝非额外5轮。每轮N次访问、ceil(N/128)次优化，保留尾批。最多四个独立进程并发，单RTX3080资源共享，实际wall时间包含争用，不把并发耗时当独占吞吐。

Uniform=default_rng(17+epoch).permutation(N)，与原Stage5.3规则相同；Stratified用同一独立numpy种子按P有放回抽N个。只改变采样；重复曝光、独特覆盖与有效训练分布确实不同，不能称完全相同训练分布/经验风险，亦不等价于独立证明某个统计量造成速度偏差。参数量/访问/优化步/初始化匹配不排除优化噪声差异。

选择最低窗口加权K1 latent-prior ADE，精确相同取早epoch，同checkpoint FDE。固定保留1/5/10/20/30轮immutable checkpoint和独立best，所有epoch原始曲线均保留，last=30，best/last只作索引。辅助K1首样本、K20分别最小ADE/FDE；固定Stage5.2 target-ID噪声种子与预算，FPC关闭。latent-prior轨迹不是预测分布数学期望。评价用验证GT；GT不能进入采样、输入或图。

正式继续门槛全部同时通过：两录像×窗口/行人权重ADE增益≥max(0.005m,1%Uniform ADE)；同两权重FDE退化≤max(0.005m,1%Uniform FDE)；HOTEL正常行走窗口权重绝对组均值速度偏差下降≥10%；两录像×两权重历史静止组FDE退化≤max(0.01m,5%)；所有数据/预算/ID/GT/配置核验。主速度门槛为|mean(v_pred−v_GT)|，同时报告mean(|每目标速度偏差|)及逐步速度误差，避免均值抵消误读。历史分类阈值0.05/0.4/1.8m/s原样保留。未来持续运动标签仅为事后诊断。

任何门槛不通过则停止该采样假设，不搜索比例、阈值、结构、seed或训练长度；best在30标预算截断，不续训。任何NaN/Inf停止该run并保留失败现场，不重试调参。其余事先冻结run仍可完成。旧100轮NoSocial不替代新30轮Uniform。即便通过也只建议另阶段验证，普通重采样不是论文创新。

协议SHA：`{sha256('configs/stage6p1/protocol.json')}`。完整公式、随机数规则、源数据/manifest SHA和停止门槛在configs/stage6p1/protocol.json；旧文件保护清单独立记录。只本地commit，不push；四run完成后停止。
''')
    rows=read_csv(ROOT/'source_class_distribution.csv')
    save('02_SAMPLING_DISTRIBUTION.md','''# 历史速度类别与唯一采样分布

每目标速度=sqrt(dx²+dy²)/0.4的7段均值，float64，只读取8帧目标位置。stationary<0.05、low[0.05,0.4)、normal[0.4,1.8)、fast≥1.8m/s。类别与概率API不接受未来、邻居或录像身份；来源目标池已经按冻结监督资格确定，不因采样改变。

唯一S：P(i)=0.5/N+0.5/(G*N_g)，G是非空速度类别数，因此总和0.5+0.5=1；空类概率0、不造样本。U按完整置换一轮一次，S每轮有放回N次。非空类期望占比=0.5*N_g/N+0.5/G。下表训练占比为训练前期望，验证只描述原有构成，不用来计算P或修改比例。

'''+table(['留出录像','split','类别','类别目标数/N','原比例','S期望比例'],[[r['fold'],r['split'],r['category'],r['targets']+'/'+r['total'],number(r['original_fraction']),number(r['expected_Stratified_fraction'])] for r in rows])+'''
实际每轮四类visits、unique_targets、repeated_visits、repeated_target_ID_windows、coverage、sampled_fraction保存各run/epochs/epochXXX_sampling.csv；完整每目标访问计数含0访问、累积计数在epochXXX_target_visits.csv.gz。这里“目标”指recording/start_frame/person-ID窗口键，不把重叠窗口当独立行人。repeated_visits=总访问−独特目标窗口；repeated_target_ID_windows=访问≥2次的独特目标窗口数，两者不同。保存order SHA和固定生成代码以复算顺序，累计覆盖率另报。结果文件最终由真实训练产生，本报告不填未执行指标。
''')


def results():
    paired=read_csv(ROOT/'analysis/paired_overall_metrics.csv');runs=read_csv(ROOT/'analysis/run_summary.csv')
    motion=read_csv(ROOT/'analysis/all_motion_group_diagnostics.csv');horizon=read_csv(ROOT/'analysis/horizon_group_metrics.csv')
    decision=json.load(open(ROOT/'analysis/research_decision.json'))
    tests=ET.parse(ROOT/'full_tests.xml').getroot().findall('testsuite')
    ntests=sum(int(t.attrib['tests']) for t in tests)
    results_table=table(['录像','选取','评价','权重','N/ID','U ADE/FDE','S ADE/FDE','S−U ADE/FDE'],[
        [r['recording'],r['selection'],r['metric_family'],r['weighting'],r['targets']+'/'+r['people'],
         number(r['Uniform_ADE_m'])+' / '+number(r['Uniform_FDE_m']),number(r['Stratified_ADE_m'])+' / '+number(r['Stratified_FDE_m']),
         number(r['delta_ADE_m'])+' / '+number(r['delta_FDE_m'])] for r in paired])
    training=table(['录像','条件','best/last','参数/有效上界','访问/优化步','wall秒','峰值显存MiB','train loss首/末','best train ADE'],[
        [r['recording'],r['condition'],r['best_epoch']+'/30',r['parameters']+'/'+r['effective_parameters'],r['total_target_visits']+'/'+r['total_optimizer_steps'],
         number(r['run_wall_seconds']),number(float(r['peak_allocated_bytes'])/2**20),number(r['train_loss_first'])+' / '+number(r['train_loss_last']),number(r['best_train_ADE'])] for r in runs])
    sampling=[]
    for r in runs:
        curve=read_csv(ROOT/'runs'/r['recording']/r['condition']/'training_curve.csv')
        sampling.append([r['recording'],r['condition'],number(sum(float(c['coverage']) for c in curve)/30),
            number(min(float(c['coverage']) for c in curve)),number(max(float(c['coverage']) for c in curve)),
            sum(int(c['repeated_visits']) for c in curve)])
    save('03_TRAINING_RESULTS.md','# 四组实际训练结果\n\n'+training+'''
四组为独立进程/同seed随机初始化、无热启动；各30完整预算epoch，smoke包含1–5轮，未增加训练。S的“完整”指N次预算访问，不是全体独特目标均被访问。并发导致wall时间包含资源争用，GPU显存是各进程PyTorch峰值而非整卡独占使用。参数与数据SHA、全部30曲线/逐步梯度/loss原始记录、定期权重和best/last索引保留。

'''+results_table+'''所有FDE配对同checkpoint；不拼接独立最优ADE/FDE。best选择只依据窗口K1 ADE，ID权重/K20没有重新选权。原始训练目标不是精确边缘NLL，K1 latent-prior不是分布期望。K20的最小ADE与最小FDE可以来自不同候选。

'''+table(['录像','条件','平均独特覆盖率','最小','最大','30轮重复访问'],sampling)+'''
best_at_budget_boundary见run_summary.csv：若best=30则预算截断，不宣称充分收敛，不自动延长。训练loss持续下降也不等于验证泛化改善。best_train与验证差距应在相同K1口径下解释，而非把重建loss和ADE直接相减。
''')
    selected=[r for r in motion if r['selection']=='best' and r['group'] in ['speed_stationary','speed_low','speed_normal','speed_fast','turning','direction_change']]
    mt=table(['录像','条件','组','权重','N/ID','ADE/FDE','预测/真实均速','有符号偏差','mean|目标偏差|','沿/横终点误差','可靠朝向N'],[
        [r['recording'],r['condition'],r['group'],r['weighting'],r['targets']+'/'+r['people'],number(r['ADE_m'])+' / '+number(r['FDE_m']),
         number(r['predicted_mean_speed_mps'])+' / '+number(r['truth_mean_speed_mps']),number(r['speed_bias_mps']),number(r['absolute_target_speed_bias_mps']),
         number(r['endpoint_along_error_m'])+' / '+number(r['endpoint_cross_error_m']),r['reliable_heading_targets']] for r in selected])
    ht=table(['条件','时间s','权重','N/ID','位置误差m','预测/真实速度m/s','偏差m/s'],[
        [r['condition'],number(r['seconds']),r['weighting'],r['targets']+'/'+r['people'],number(r['error_m']),number(r['predicted_speed_mps'])+' / '+number(r['truth_speed_mps']),number(r['speed_bias_mps'])]
        for r in horizon if r['recording']=='biwi_hotel' and r['selection']=='best' and r['group']=='speed_normal'])
    save('04_MOTION_ERROR_ANALYSIS.md','# 历史运动分组和速度诊断\n\n'+mt+'''速度由x_last到预测首帧再到后续11段位置除0.4计算，GT用同一起点；目标速度偏差是12段速度均值之差。有符号组偏差、绝对组偏差、逐目标绝对偏差与逐步绝对速度误差是不同统计量，不能互换。

终点沿/横轴由历史净位移定义，净位移≥0.2m且历史均速≥0.2m/s才可靠；否则保留空值，仍计算ADE/FDE/速度。沿轴负值可描述落后，但小组均值可能抵消，不能指定唯一因果。无可靠方向样本没有伪填0。所有33个冻结历史分组包括空组在all_history_group_metrics.csv和all_motion_group_diagnostics.csv，均未删难例。

## HOTEL正常行走：全部12步

'''+ht+'''
每目标/每步配对原始派生误差在motion_diagnostics.csv.gz与per_step_motion.csv.gz；完整XY只在被忽略cache并记录SHA。未来持续/减速标签是Stage6事后定义，未用于采样、模型输入或在线gate。当前实验只改变整个采样策略，不能独立证明状态构成或速度偏差是唯一因果中介。
''')
    primary=[r for r in paired if r['selection']=='best' and r['metric_family']=='K1-prior']
    pt=table(['录像','权重','U ADE/FDE','S ADE/FDE','ΔADE/ΔFDE'],[[r['recording'],r['weighting'],number(r['Uniform_ADE_m'])+' / '+number(r['Uniform_FDE_m']),
         number(r['Stratified_ADE_m'])+' / '+number(r['Stratified_FDE_m']),number(r['delta_ADE_m'])+' / '+number(r['delta_FDE_m'])] for r in primary])
    save('05_PAIRED_COMPARISON.md','# 配对比较与证据边界\n\n'+pt+'''
同录像同一窗口/ID、同真实未来SHA、同历史SHA、同采样噪声ID/预算，逐目标S−U保存paired_target_errors.csv.gz；窗口加权、按录像内行人等权分别计算，两录像宏平均是两录像等权，不是把所有目标窗口混在一起。重叠窗口不独立，没有用窗口数量冒充独立行人显著性。只有一个训练seed和两个曝光开发录像，不作确认性或跨所有场景结论。

源池/监督资格相同，但有放回采样改变实际类别比例、独特覆盖、重复曝光及优化路径；这是采样策略整体对照，不是模型结构收益，也不是已完全隔离某个统计因果。两个30轮Uniform均真实重新训练，旧100轮NoSocial未替代、也未与新30轮混排。三个CV冻结同目标参考文件同时保留，不事后逐录像选最弱或最好CV作为部署模型。
''')
    gates=table(['门槛','录像','权重','通过','完整检查值'],[[r['gate'],r.get('recording','HOTEL' if 'speed' in r['gate'] else 'all'),r.get('weighting','target-window' if 'speed' in r['gate'] else 'all'),r['passed'],
        '; '.join(k+'='+str(v) for k,v in r.items() if k not in ['gate','recording','weighting','passed'])] for r in decision['checks']])
    boundary=[r['recording']+'/'+r['condition'] for r in runs if r['best_at_budget_boundary']=='True']
    verdict='全部预设门槛通过，仅建议另阶段稳定性验证，不能称论文创新或确认性收益。' if decision['all_passed'] else '未通过全部预设门槛，停止当前采样假设；不再搜索比例、速度阈值、seed、结构或训练长度。'
    save('06_RESEARCH_DECISION.md','# 冻结规则下的研究决策\n\n**'+verdict+'**\n\n'+gates+'\n预算边界best：'+(', '.join(boundary) if boundary else '无')+'。到30轮即停止，不自动延长。\n\n'+'''
如果速度偏差改善但ADE/FDE门槛不通过，只是运动学诊断结果；若HOTEL改善、ZARA2退化，只是条件性现象。一个子组改善不能覆盖整体/另一录像失败。未来误差不能反过来定义采样类别。即便全部通过，常规类别重采样也不是新论文方法，仍需更多未曝光资料、seed和针对具体命题的文献核查；本轮不自动追加。

下一研究方向交由用户决定，不重启旧社会模块或可靠性gate。当前不训练创新模型、不读取ETH、不上传checkpoint/数据、不自动push。
''')
    save('STAGE6P1_REPORT.md','# Stage6.1：运动历史采样可行性试验\n\n**'+verdict+'**\n\n'+pt+'## 实验预算\n\n'+training+f'''
四组实际从头训练各30轮，前5轮包含在内，统一冻结Stage5.3 NoSocial、seed17/Adam0.0005/b128，模型/参数/损失相同。没有额外种子、额外epoch或网络模块。完整单元测试{ntests}项通过；历史文件最终SHA验收见results/tables/stage6p1/delivery_verification.json。所有指标来自真实训练与配对评价，ETH未评价。best在30的run见06，标预算截断。

01冻结协议；02训练前类别期望与实际计数说明；03完整best/last、K1/K20与预算；04正常行走/静止和全部历史组、每步/终点速度诊断；05两权重与宏平均配对；06逐条停止规则。数据池/manifest/原始来源SHA、初始化、采样噪声、逐epoch访问/重复/覆盖、曲线、checkpoint SHA、GPU与实耗均保存results/tables/stage6p1/。原始XY与checkpoint仅本地忽略目录。

主速度门槛是HOTEL正常组绝对组均值偏差下降≥10%，并附逐目标绝对偏差避免抵消。精确结果和每个判定值见04/06及research_decision.json，不能以旧Full预测速度替代新NoSocial结果。两个曝光开发录像+单seed不构成独立泛化确认。采样改变训练分布和优化噪声，不能宣称唯一失败原因已被证明。

训练前源码/协议commit由run/provenance.json记录；本报告所在最终本地提交由Git追溯，不制造循环自SHA。不自动push。用户自行：`git push -u origin feat/stage6p1-motion-sampling-pilot`。完成后停止。
''')
    print('Seven Stage6.1 reports present; actual results and fixed decision only')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else results()
