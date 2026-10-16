"""Evidence-only delivery generated after all eight preregistered runs finish."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,os
from src.socialvae_causal.io import read_csv
ROOT=Path('results/tables/stage5p5');A=ROOT/'analysis';R=Path(os.environ.get('STAGE5P5_REPORT_OUTPUT','reports/stage5p5'))
index=read_csv(A/'all_12_run_index.csv');effects=read_csv(A/'all_group_paired_effects.csv');stats=read_csv(A/'multiseed_all_group_statistics.csv')
macros=read_csv(A/'recording_macro_paired_metrics.csv');ci=read_csv(ROOT/'bootstrap/all_bootstrap_intervals.csv')
convergence=json.load(open(A/'convergence_audit.json'));decision=json.load(open(ROOT/'stability_decision.json'));checks=json.load(open(ROOT/'delivery_verification.json'))
refs=read_csv(A/'historical_reference_only.csv');plan=json.load(open('configs/stage5p5/protocol.json'))
scheduler=json.load(open(ROOT/'scheduler_completion.json'))
new_code_commits={r['code_commit'] for r in index if r['reference_or_new']=='new formal run'}
assert len(new_code_commits)==1
frozen_training_commit=next(iter(new_code_commits))

def f(v):return '—' if v is None or v=='' else f'{float(v):.6f}'
def pair(r,prefix):return f(r[prefix+'_ADE_m'])+' / '+f(r[prefix+'_FDE_m'])
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def save(name,text):
 with (R/name).open('x') as stream:stream.write(text)
def one(seed,rec,weight='target-window',group='all',family='K1-prior',selection='best'):
 return next(r for r in effects if (int(r['seed']),r['recording'],r['weighting'],r['group'],r['metric_family'],r['selection'])==(seed,rec,weight,group,family,selection))
def summary(rec,weight='target-window',group='all',family='K1-prior'):
 return next(r for r in stats if (r['recording'],r['weighting'],r['group'],r['metric_family'],r['selection'])==(rec,weight,group,family,'best'))
base='单位米；Delta=Full−Placebo，负值表示Full好。各模型按最低窗口加权验证K1 latent-prior ADE选checkpoint，FDE及所有辅助指标均配对同一checkpoint。行人权重分析仍使用该checkpoint，不另选最低行人ADE。K1 prior不是预测分布数学期望，训练目标为原重建+MC KL估计，不是精确边缘NLL。\n\n'
raw=''
for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
 raw+='\n## '+family+'：全部seed原始指标\n\n'+table(['录像','权重','seed','Full ADE/FDE','Placebo ADE/FDE','Delta ADE/FDE'],[
  [rec,w,seed,pair(q,'Full'),pair(q,'Placebo'),f(q['delta_ADE_m'])+' / '+f(q['delta_FDE_m'])]
  for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for seed in [17,42,2026] for q in [one(seed,rec,w,family=family)]])
training=table(['录像','seed','模型','best轮','best K1 ADE/FDE','last K1 ADE/FDE','实耗min','进程内PyTorch峰值分配GiB'],[
 [c['recording'],c['seed'],c['model'],c['best_epoch'],f(c['best_validation']['K1_prior_ADE_m'])+' / '+f(c['best_validation']['K1_prior_FDE_m']),
 f(c['last_epoch_curve']['K1_prior_ADE_m'])+' / '+f(c['last_epoch_curve']['K1_prior_FDE_m']),f(c['actual_run_wall_seconds']/60),f(c['peak_allocated_bytes']/2**30)] for c in convergence])
objective=table(['录像','seed','模型','best训练loss/重建/KL','last训练loss/重建/KL','最大梯度L2'],[
 [c['recording'],c['seed'],c['model'],' / '.join(f(c['best_epoch_curve'][k]) for k in ['train_loss','train_rec','train_kl']),
 ' / '.join(f(c['last_epoch_curve'][k]) for k in ['train_loss','train_rec','train_kl']),f(c['gradient_L2_max'])] for c in convergence])
wall_note=f"训练前冻结协议与代码提交：`{frozen_training_commit}`；8组新增run均记录该提交。新增8组调度总墙钟实耗：{f(scheduler['actual_wall_seconds']/60)}分钟。单run实耗和总墙钟分别记录，并发实耗不可简单相加当作独占GPU训练时间。两模型名义参数量均2,144,002，结构有效上界均2,045,698；各参数实际梯度并集及无梯度计数见每组lifetime_parameter_gradients.csv，参数数目相同不意味着输入分布或函数类相同。\n\n"
save('02_MULTISEED_TRAINING.md','# 多随机种子正式训练\n\n'+base+wall_note+training+'\n'+objective+'''
12组完整索引见analysis/all_12_run_index.csv，4组seed17为历史冻结正式结果，8组42/2026为本阶段新run。每组均100完整epoch，保留尾batch，未热启动。各seed公共初始参数与构造后RNG一致，初始SHA见initialization_checks.json与各run/provenance.json。

所有每轮曲线整合在analysis/all_12_training_curves.csv.gz；新run每epoch保留步骤、逐目标误差、梯度标量并集、参数SHA/范数/有限性、显存和耗时。best/last/固定周期权重留在experiments/stage5p5/checkpoints并被Git忽略；源路径和SHA见各run/checkpoint_inventory.csv。best候选序列化到RAM后一次写出，不覆盖旧epoch文件；周期与best均做严格恢复/RNG检查。

新run在同一3080最多2Full+2Placebo进程并发，墙钟实耗包含竞争；seed17运行历史调度不同，不能据实耗直接比较网络效率。全部8组事先冻结，无seed42后调节seed2026、额外seed、200epoch延长、数据增强或新模型。执行日志和失败状态保留；未执行数据不得填成0。
''')
aggregate=''
for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
 aggregate+='\n## '+family+'：跨seed与两录像宏平均\n\n'+table(['录像/宏平均','权重','ΔADE均值±样本SD','ΔFDE均值±样本SD','Full/Placebo ADE的跨seed SD','ADE/FDE改善seed数'],[
 [r['recording'],r['weighting'],f(r['delta_ADE_m_mean'])+' ± '+f(r['delta_ADE_m_sample_SD']),f(r['delta_FDE_m_mean'])+' ± '+f(r['delta_FDE_m_sample_SD']),
 f(r['Full_ADE_m_sample_SD'])+' / '+f(r['Placebo_ADE_m_sample_SD']),r['ADE_Full_better_seeds']+'/'+r['available_seeds']+' ; '+r['FDE_Full_better_seeds']+'/'+r['available_seeds']]
 for r in stats if r['group']=='all' and r['selection']=='best' and r['metric_family']==family])
bootstrap=table(['录像','seed','权重','指标','点差值','95%簇bootstrap区间','含0'],[
 [r['recording'],r['training_seed'],r['weighting'],r['metric'],f(r['paired_point_delta_m']),f(r['percentile_95_lower_m'])+' ~ '+f(r['percentile_95_upper_m']),r['interval_contains_zero']]
 for r in ci if r['metric_family']=='K1-prior'])
save('03_PAIRED_STATISTICS.md','# 配对统计与不确定性\n\n'+base+raw+aggregate+'\n## K1簇bootstrap\n\n'+bootstrap+'''
每个固定seed/录像真实执行2000次ID整簇有放回抽样，保留每个ID所有重叠窗口及重复次数。窗口加权按抽样窗口总数加权，行人权重先在ID内部平均；同一抽样矩阵用于两指标、两权重和三个指标族。原始抽样ID、全部分布、稳定种子、72个区间记录见bootstrap/。CI为percentile 2.5%/97.5%，不是p值或确认性显著性检验。

HOTEL28 ID、ZARA2 35 ID，同行人窗口强相关，不同行人亦可能互动相关；该cluster方法只部分处理相关性。三次训练seed不是三个新数据集，两个录像不是大量独立场景。区间条件于已选择best的模型与固定评价噪声，不重做训练/checkpoint选择，不能修正验证选模或历史开发曝光偏差。

样本SD用ddof1，n=3估计本身不稳定。独立模型误差跨seed SD与配对差值SD同时展示：前者含共同初始化难易波动，后者针对配对效应。训练seed同时决定初始化、预注册样本顺序及训练潜变量噪声；这是整体随机训练稳定性验证，不能归因为只改变初始权重的单因素实验。主K1确定性无需评价采样；K1首样本/K20跨seed还包含预注册评价噪声变化，不能将其SD全解释为初始化影响。

两录像宏平均先分别计算录像内权重指标，再各占1/2；不是把1410窗口混池。原始三seed分别列出，不以均值隐藏方向反转。NoSocial/EgoCapacity仅seed17历史参考，未纳入这些均值、SD或bootstrap。
''')
allgroups=''
for rec in ['biwi_hotel','crowds_zara02']:
 for weight in ['target-window','equal-person-ID']:
  allgroups+='\n## '+rec+' / '+weight+'：全部历史组\n\n'+table(['组','窗口/ID','Full平均ADE','Placebo平均ADE','ΔADE(17/42/2026)','ΔFDE均值','改善ADE/FDE seed数','少样本'],[
   [r['group'],r['target_windows_per_seed']+'/'+r['people_per_seed'],f(r['Full_ADE_m_mean']),f(r['Placebo_ADE_m_mean']),
    ' / '.join(f(one(seed,rec,weight,r['group'])['delta_ADE_m']) for seed in [17,42,2026]),f(r['delta_FDE_m_mean']),
    r['ADE_Full_better_seeds']+'/'+r['available_seeds']+' ; '+r['FDE_Full_better_seeds']+'/'+r['available_seeds'],r['small_group']]
   for r in stats if r['recording']==rec and r['weighting']==weight and r['selection']=='best' and r['metric_family']=='K1-prior'])
focus=table(['录像','权重','历史组','ΔADE三seed','ΔFDE三seed'],[
 [rec,w,g,' / '.join(f(one(s,rec,w,g)['delta_ADE_m']) for s in [17,42,2026]),' / '.join(f(one(s,rec,w,g)['delta_FDE_m']) for s in [17,42,2026])]
 for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for g in ['speed_stationary','speed_normal','last_neighbors_6plus','density_3plus']])
condition_notes=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for w in ['target-window','equal-person-ID']:
  normal_s=summary(rec,w,'speed_normal');stationary_s=summary(rec,w,'speed_stationary')
  normal_repeat=normal_s['ADE_Full_better_seeds']=='3' and normal_s['FDE_Full_better_seeds']=='3'
  stationary_repeat=stationary_s['ADE_Placebo_better_seeds']=='3' and stationary_s['FDE_Placebo_better_seeds']=='3'
  condition_notes.append(f"- {rec}/{w}正常行走：Full的ADE改善{normal_s['ADE_Full_better_seeds']}/{normal_s['available_seeds']}，FDE改善{normal_s['FDE_Full_better_seeds']}/{normal_s['available_seeds']}；{'该历史条件内三seed收益方向重复' if normal_repeat else '该历史条件内收益方向未完全重复'}，不推广为独立录像确认。")
  condition_notes.append(f"- {rec}/{w}历史静止：Full的ADE退化{stationary_s['ADE_Placebo_better_seeds']}/{stationary_s['available_seeds']}，FDE退化{stationary_s['FDE_Placebo_better_seeds']}/{stationary_s['available_seeds']}；{'相对Placebo的退化三seed重复' if stationary_repeat else '未发现ADE/FDE三seed同步退化'}，ID数与少样本限制仍须保留。")
  for g in ['last_neighbors_6plus','density_3plus']:
   r=summary(rec,w,g)
   condition_notes.append(f"- {rec}/{w}/{g}：{r['target_windows_per_seed']}窗口/{r['people_per_seed']}ID，ADE改善{r['ADE_Full_better_seeds']}/{r['available_seeds']}，FDE改善{r['FDE_Full_better_seeds']}/{r['available_seeds']}，少样本={r['small_group']}；{'无法支持高人数/高密度稳定收益判断' if r['small_group']=='True' or r['available_seeds']=='0' else '符号一致性仅针对已曝光开发组'}。")
save('04_MOTION_AND_INTERACTION.md','# 历史运动与交互条件\n\n'+base+focus+'\n## 条件问题逐项判断\n\n'+'\n'.join(condition_notes)+'\n'+allgroups+'''
分组直接复用Stage5.2历史特征及SHA：静止<0.05m/s，低速<0.4、正常<1.8，其余高速；转弯/方向变化和相对方向需历史可靠速度。不用未来GT或未来误差挑输入/组。邻居人数指最后历史帧2m内具有可观测速度的合法邻居；密度为历史后7帧合法邻居均数，不是完整20帧离线上下文人数。

上表保留全部组，含新进入、未知速度、同向/相向/交叉及不可判定组；多标签组有重叠，不能相加当独立人数。ID<10或窗口<30标记少样本；空组指标缺失，不填零。各seed独立ADE/FDE、K1首样本/K20和last结果均见analysis/all_group_model_metrics.csv及all_group_paired_effects.csv，不能只根据上表的均值宣布组内稳定收益。

正常行走优势、静止退化是否在3seed重复，直接查上方各自三项差值；负值Full更好。高人数/高密度价值必须同时检查人数、符号和少样本标记；空组或少数ID不足支持高密度结论。分组为探索性条件分析，不作多重子组显著性筛选。
''')
normal=table(['录像','权重','seed','Full正常行走ADE/FDE','Placebo正常行走ADE/FDE'],[
 [rec,w,seed,pair(q,'Full'),pair(q,'Placebo')]
 for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for seed in [17,42,2026] for q in [one(seed,rec,w,'speed_normal')]])
cv_normal=table(['录像','权重','参考模型','正常行走ADE/FDE','窗口/ID'],[
 [r['recording'],r['weighting'],r['model'],f(r['ADE_m'])+' / '+f(r['FDE_m']),r['target_windows']+'/'+r['people']]
 for r in refs if r['model'].startswith('CV-') and r['group']=='speed_normal'])
late=table(['录像','seed','模型','best≤5','best≥91','训练K1(best/last)','验证K1(best/last)','后20轮训练loss均值变化','后20轮验证ADE均值变化'],[
 [c['recording'],c['seed'],c['model'],c['best_within_first5'],c['best_within_last10'],
 f(c['best_train_K1']['train_K1_prior_ADE_m'])+' / '+f(c['last_epoch_curve']['train_K1_prior_ADE_m']),
 f(c['best_validation']['K1_prior_ADE_m'])+' / '+f(c['last_epoch_curve']['K1_prior_ADE_m']),
 f(c['last20']['train_loss']['change']),f(c['last20']['K1_prior_ADE_m']['change'])] for c in convergence])
epoch_notes=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for model in ['Full','PlaceboSocial']:
  cc=sorted([c for c in convergence if c['recording']==rec and c['model']==model],key=lambda c:c['seed'])
  epoch_notes.append(f"- {rec}/{model}：seed17/42/2026的best分别为{' / '.join(str(c['best_epoch']) for c in cc)}轮；前5轮最优{sum(c['best_within_first5'] for c in cc)}/3，最后10轮最优{sum(c['best_within_last10'] for c in cc)}/3。")
ctx=read_csv(ROOT/'context_distribution.csv')
distribution=table(['录像','split','上下文','统计量','均值','P95'],[
 [r['recording'],r['split'],r['context'],r['statistic'],f(r['mean']),f(r['P95'])] for r in ctx if r['statistic'] in ['count','distance','speed']])
normal_notes=[]
for w in ['target-window','equal-person-ID']:
 cv=next(r for r in refs if r['recording']=='biwi_hotel' and r['model']=='CV-Last2' and r['group']=='speed_normal' and r['weighting']==w)
 poorer=sum(float(one(seed,'biwi_hotel',w,'speed_normal')['Full_ADE_m'])>float(cv['ADE_m']) for seed in [17,42,2026])
 normal_notes.append(f"HOTEL/{w}正常行走：Full在{poorer}/3 seed的ADE高于同目标CV-Last2（{f(cv['ADE_m'])}m）；其它强CV逐项列出。")
stationary=table(['录像','权重','seed','Full历史静止ADE/FDE','Placebo历史静止ADE/FDE'],[
 [rec,w,seed,pair(q,'Full'),pair(q,'Placebo')]
 for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for seed in [17,42,2026] for q in [one(seed,rec,w,'speed_stationary')]])
cv_stationary=table(['录像','权重','参考模型','历史静止ADE/FDE','窗口/ID'],[
 [r['recording'],r['weighting'],r['model'],f(r['ADE_m'])+' / '+f(r['FDE_m']),r['target_windows']+'/'+r['people']]
 for r in refs if r['model'].startswith('CV-') and r['group']=='speed_stationary'])
stationary_notes=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for w in ['target-window','equal-person-ID']:
  cv=next(r for r in refs if r['recording']==rec and r['model']=='CV-Last2' and r['group']=='speed_stationary' and r['weighting']==w)
  available=summary(rec,w,'speed_stationary')
  if available['available_seeds']=='0':
   stationary_notes.append(f"{rec}/{w}历史静止为空组，不判断CV退化。")
  else:
   poorer=sum(float(one(seed,rec,w,'speed_stationary')['Full_ADE_m'])>float(cv['ADE_m']) for seed in [17,42,2026])
   stationary_notes.append(f"{rec}/{w}历史静止：Full在{poorer}/3 seed的ADE高于固定CV-Last2（{f(cv['ADE_m'])}m）；少样本={available['small_group']}。这是相对CV的退化，与04报告相对Placebo的差值分别解释。")
full_cv=table(['录像','权重','seed/参考','Full或CV ADE/FDE'],[
 [r['recording'],r['weighting'],r['model']+' (解析参考)',f(r['ADE_m'])+' / '+f(r['FDE_m'])]
 for r in refs if r['model'].startswith('CV-') and r['group']=='all'])
full_cv+='\n'+table(['录像','权重','seed','Full ADE/FDE'],[[rec,w,s,pair(one(s,rec,w),'Full')] for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for s in [17,42,2026]])
save('05_GENERALIZATION_AUDIT.md','# 泛化、训练预算与输入分布核查\n\n'+base+'## 强CV与Full同时报告\n\n'+full_cv+'\n## 正常行走目标\n\n'+normal+'\n'+cv_normal+'\n'+'\n'.join(normal_notes)+'\n\n## 历史静止目标的绝对预测能力\n\n'+stationary+'\n'+cv_stationary+'\n'+'\n'.join(stationary_notes)+'\n\n## 训练稳定性与后20轮\n\n'+'\n'.join(epoch_notes)+'\n\n'+late+'\n## 冻结合成分布差异\n\n'+distribution+'''
后20轮变化指91–100均值减81–90均值。best很早且后期训练loss下降、验证ADE上升是经验泛化退化迹象，不等于证明具体因果原因。训练与验证目标运动构成不同，训练K1与验证K1差距不能脱离录像分布解释；原始VAE目标也不保证确定性K1单调改善。best接近100需记预算截断，绝不因此延长。

三种CV均在同一冻结目标集合上重新核对过GT和误差，主解析参考固定CV-Last2。事后逐录像/权重挑选最低CV仅是“解析基线上界参考”，不是可部署新模型。社会归因与绝对预测能力是两个问题：即使Full优于Placebo，HOTEL如果仍弱于CV必须保留该事实。K20与确定性CV预测预算不同，不混为K1比较。

上下文分布为Stage5.4全冻结训练/验证目标的历史观测统计，SHA逐值复用；输入本身不随训练seed改变，不是新增分布匹配实验。Placebo始终4个近邻、完整8帧、恒定初始偏移；真实邻居数量、距离、进入和速度可观测性不同。context_mask_summary.csv记录实际历史mask，synthetic_future_slots_all_padding列只针对合成上下文（real行为不由该列False推断）。不能因三seed一致便消除这些结构/输入分布混杂。

上下文表count为每历史帧有效邻居数，distance单位米，speed单位米/秒（官方逐步位移除以0.4秒）；count、distance、speed按各自有效观测集合汇总，不能把这些观测条目当作独立行人。

ETH未评价；两个录像为历史多轮曝光的开发数据。录像隔离保护每个run的训练/验证支持，不意味着新的独立科研确认。
''')
verdict=decision['verdict']
if verdict=='stronger_cross_initialization_development_evidence':
 conclusion='Full在3seed、2录像、2权重的K1 ADE及配对FDE全部方向一致。真实邻居内容具有更强的跨初始化开发证据；仍未排除合成分布不匹配、选模和开发曝光偏差，不能宣布严格因果贡献或独立录像确认。'
elif verdict=='ADE_only_evidence_with_FDE_conflict':
 conclusion='Full的K1 ADE方向在3seed、2录像、2权重下一致，但配对FDE不完全同步。只能记录ADE单项开发证据，不支持联合ADE/FDE稳定改善；停止据此直接开发复杂社会模块。'
else:
 conclusion='seed17毫米级优势未在所有seed、录像和权重下重复。当前社会内容收益不稳定或不普遍，停止继续基于该收益开发复杂社会交互模块；不能用三个seed均值或某个改善分组挽救失败的稳定性结论。'
main=table(['录像','权重','seed','Full K1 ADE/FDE','Placebo K1 ADE/FDE','Delta ADE/FDE'],[
 [rec,w,s,pair(q,'Full'),pair(q,'Placebo'),f(q['delta_ADE_m'])+' / '+f(q['delta_FDE_m'])]
 for rec in ['biwi_hotel','crowds_zara02'] for w in ['target-window','equal-person-ID'] for s in [17,42,2026] for q in [one(s,rec,w)]])
consistency=table(['录像','权重','3seed ADE差值','3seed FDE差值','ADE反转','FDE反转'],[
 [r['recording'],r['weighting'],' / '.join(f(x) for x in r['ADE_deltas_m']),' / '.join(f(x) for x in r['FDE_deltas_m']),r['ADE_direction_reversal'],r['FDE_direction_reversal']] for r in decision['cells']])
variation=table(['录像','权重','平均ΔADE±样本SD','Full ADE跨seed SD','Placebo ADE跨seed SD','平均ΔFDE±样本SD'],[
 [r['recording'],r['weighting'],f(r['delta_ADE_m_mean'])+' ± '+f(r['delta_ADE_m_sample_SD']),f(r['Full_ADE_m_sample_SD']),f(r['Placebo_ADE_m_sample_SD']),
 f(r['delta_FDE_m_mean'])+' ± '+f(r['delta_FDE_m_sample_SD'])]
 for r in stats if r['selection']=='best' and r['metric_family']=='K1-prior' and r['group']=='all'])
magnitude_notes=[]
for rec in ['biwi_hotel','crowds_zara02','two-recording-macro']:
 for w in ['target-window','equal-person-ID']:
  r=summary(rec,w)
  magnitude_notes.append(f"- {rec}/{w}：|平均ΔADE|={f(abs(float(r['delta_ADE_m_mean'])))}m，配对差值样本SD={f(r['delta_ADE_m_sample_SD'])}m，Full/Placebo原始ADE跨seed SD={f(r['Full_ADE_m_sample_SD'])}/{f(r['Placebo_ADE_m_sample_SD'])}m。n=3且无预设显著性门槛，只能描述相对幅度，不能凭该比值宣称明显超出自然波动。")
auxiliary=table(['指标族','录像','权重','平均ΔADE/ΔFDE','ADE/FDE改善seed数'],[
 [family,rec,w,f(r['delta_ADE_m_mean'])+' / '+f(r['delta_FDE_m_mean']),r['ADE_Full_better_seeds']+'/3 ; '+r['FDE_Full_better_seeds']+'/3']
 for family in ['K1-first-sample','K20-independent-minima'] for rec in ['biwi_hotel','crowds_zara02','two-recording-macro']
 for w in ['target-window','equal-person-ID'] for r in [summary(rec,w,family=family)]])
mainci=[r for r in ci if r['metric_family']=='K1-prior']
recording_notes=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for w in ['target-window','equal-person-ID']:
  r=summary(rec,w)
  recording_notes.append(f"- {rec}/{w}：K1 ADE改善{r['ADE_Full_better_seeds']}/3 seed，FDE改善{r['FDE_Full_better_seeds']}/3 seed。{'该录像与权重内方向重复，但不能代替两个录像同时满足门槛' if r['ADE_Full_better_seeds']=='3' and r['FDE_Full_better_seeds']=='3' else '该录像与权重内未达到ADE/FDE三seed全部改善'}。")
ci_note=f"K1条件bootstrap的{len(mainci)}个录像/seed/权重/指标区间中，{sum(r['interval_contains_zero']=='True' for r in mainci)}个包含0。该计数只作不确定性描述，不是预设稳定性门槛或多重显著性结论。"
focus_notes=[]
for rec in ['biwi_hotel','crowds_zara02']:
 for w in ['target-window','equal-person-ID']:
  for g in ['speed_normal','speed_stationary','last_neighbors_6plus','density_3plus']:
   r=summary(rec,w,g)
   focus_notes.append(f"- {rec}/{w}/{g}：ADE改善{r['ADE_Full_better_seeds']}/{r['available_seeds']}，FDE改善{r['FDE_Full_better_seeds']}/{r['available_seeds']}；窗口{r['target_windows_per_seed']}，ID{r['people_per_seed']}，少样本标记{r['small_group']}。")
notes='''
本阶段只允许并实际执行8个新增100epoch run，未训练NoSocial/EgoCapacity，未改变Placebo、attention、结构、loss、数据协议或评价batch。seed42成绩未改变seed2026配置，未加seed或200epoch延长。4个seed17来源冻结而非重复训练，全部12组合单独列出。

Stage5.4原始单seed结果和报告保持不变。本阶段新增证据用于更新其跨随机训练可推广性解释，不回写旧指标，也不把旧阶段的初步迹象升级为未经验证的独立因果结论。

平均配对差值必须与配对SD、各模型跨seed SD及每个seed原始差值一同解释。共同难易波动可能被配对抵消，因此不能只凭独立模型SD较大否定一致的配对方向；也不能凭n=3的平均小差值宣布统计确认。K1首样本/K20的评价噪声随训练seed按预注册键变化，其波动不是纯初始化波动。

bootstrap固定最佳checkpoint和评价噪声，只重采样人ID，未重做训练或checkpoint选择；不修复选模偏差、历史开发曝光、行人间相互依赖或独立录像不足。HOTEL28、ZARA2 35 ID，不能把1410重叠窗口或三次训练当大量独立样本。

HOTEL与三种强CV、正常行走绝对误差、后20轮泛化退化在05报告完整展示。K1与K20分开解释，不以K20替代社会K1稳定性。普通VAE、伪邻居和多seed统计本身不是论文新方法。

本阶段结束，暂不设计第四章新网络，不自行开展更多训练。若稳定性未获支持，明确停止由这项毫米差驱动的复杂社会模块路线；成熟SocialVAE可保留为冻结公开基线，后续研究问题需新的独立指令。即使稳定性通过，也只支持进一步研究设计，不等于已获得方法创新。
'''
save('06_RESEARCH_DECISION.md','# 研究判断\n\n'+conclusion+'\n\n'+'\n'.join(recording_notes)+'\n\n'+base+main+'\n'+consistency+'\n'+variation+'\n\n## 收益幅度与自然波动\n\n'+'\n'.join(magnitude_notes)+'\n\n'+ci_note+'\n\n## 辅助采样指标单独判断\n\n'+auxiliary+'\n仅当同一指标族各seed方向均一致时才记录该辅助指标重复；不将主K1结论自动推广到K20。\n\n## 四项历史条件问题\n\n'+'\n'.join(focus_notes)+'\n'+notes)
save('STAGE5P5_REPORT.md','# Stage5.5：社会内容收益的跨初始化核查\n\n'+conclusion+'\n\n'+'\n'.join(recording_notes)+'\n\n'+base+main+'\n'+variation+'\n'+consistency+'\n'+ci_note+'\n\n'+training+'\n\n## 辅助采样指标\n\n'+auxiliary+'\n'+notes+f'''
验收：{checks['completed_new_runs']}组新增正式run均100完整epoch；12组合完整索引；{checks['unit_tests']}项完整单元测试通过；{checks['protected_files_verified']}个历史文件SHA未变；{checks['new_checkpoints_verified']}个新增checkpoint核验；ETH未评价，官方源代码和Stage5.4合成规则未改。

训练前冻结协议/代码提交：`{frozen_training_commit}`，全部8组新增run采用该代码版本。最终交付提交由本报告所在Git历史核查；不虚构包含自身报告的循环SHA。

结果入口：results/tables/stage5p5/analysis/；bootstrap原始分布、抽样ID和区间在bootstrap/；每轮原始日志在runs/。代码/数据/配置/模型SHA见frozen_source_sha.json、各run/provenance.json、analysis/all_12_run_index.csv及checkpoint_inventory.csv。NoSocial/EgoCapacity历史结果只在historical_reference_only.csv，不参与多seed汇总。

详细报告01冻结协议、02训练、03配对与bootstrap、04全部历史组、05泛化、06科研决策均已保存。本地commit后停止，不自动push。用户自行：git push -u origin feat/stage5p5-social-stability。
''')
print('Seven-report package generated from actual complete multiseed measurements')
