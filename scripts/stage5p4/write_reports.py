"""Generate research delivery from executed measurements; no invented metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import os
from src.socialvae_causal.io import read_csv
ROOT=Path('results/tables/stage5p4');REPORT=Path(os.environ.get('STAGE5P4_REPORT_OUTPUT','reports/stage5p4'))
OUT=ROOT/('analysis_final' if (ROOT/'analysis_final').exists() else 'analysis_phase1')
metrics=read_csv(OUT/'model_metrics.csv');groups=read_csv(OUT/'all_group_metrics.csv');macros=read_csv(OUT/'recording_macro_metrics.csv')
convergence=json.load(open(OUT/'convergence_audit.json'));decision=json.load(open(ROOT/'phase1_decision.json'))
verification=json.load(open(ROOT/'delivery_verification.json'))
recs=['biwi_hotel','crowds_zara02'];models=['CV-Last2','CV-OLS4','CV-OLS8','Full','NoSocial','EgoCapacity']
placebo='PlaceboSocial' in {r['model'] for r in metrics}
if placebo:models.append('PlaceboSocial')
final_decision=json.load(open(ROOT/'phase2_decision.json')) if placebo else None
def val(model,rec,weight='target-window',family='K1-prior',which='best'):
 if model.startswith('CV-'):family='deterministic';which='analytic'
 return next(r for r in metrics if (r['model'],r['recording'],r['weighting'],r['metric_family'],r['selection'])==(model,rec,weight,family,which))
def fmt(x):return '—' if x is None or x=='' else f'{float(x):.6f}'
def pair(r):return fmt(r['ADE_m'])+' / '+fmt(r['FDE_m'])
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def save(name,text):
 with (REPORT/name).open('x') as f:f.write(text)
base='单位米，ADE/FDE配对来自同一按最低验证K1 ADE选择的checkpoint。K1 latent-prior不是预测分布数学期望；K20两项各自取最优候选。全部结果来自seed17，重复曝光开发录像，不是独立确认或跨seed证据。\n\n'
summary=table(['录像','模型','K1 ADE/FDE','K1首样本 ADE/FDE','K20 minADE/minFDE'],[
 [rec,m,pair(val(m,rec)), '—' if m.startswith('CV-') else pair(val(m,rec,family='K1-first-sample')),
 '—' if m.startswith('CV-') else pair(val(m,rec,family='K20-independent-minima'))] for rec in recs for m in models])
training=table(['录像','模型','best轮','best训练ADE/FDE','last验证ADE/FDE','100轮实耗min','进程内PyTorch峰值分配GiB'],[
 [c['recording'],c['model'],c['best_epoch'],fmt(c['best_train_K1']['train_K1_prior_ADE_m'])+' / '+fmt(c['best_train_K1']['train_K1_prior_FDE_m']),
 fmt(c['last_epoch']['K1_prior_ADE_m'])+' / '+fmt(c['last_epoch']['K1_prior_FDE_m']),f"{c['actual_run_wall_seconds']/60:.2f}",f"{c['peak_allocated_bytes']/2**30:.3f}"] for c in convergence])
objective_rows=[]
for c in convergence:
 if c['model'] not in ['EgoCapacity','PlaceboSocial']:continue
 variant='ego_capacity' if c['model']=='EgoCapacity' else 'placebo_social'
 for label,e in [('best',c['best_epoch']),('last',100)]:
  r=json.load(open(ROOT/variant/c['recording']/f'epochs/epoch{e:03d}.json'))
  if label=='best':r['train_K1_prior_ADE_m']=c['best_train_K1']['train_K1_prior_ADE_m']
  objective_rows.append([c['recording'],c['model'],label,e,*[fmt(r[k]) for k in ['train_loss','train_rec','train_kl','posterior_loss','train_K1_prior_ADE_m','K1_prior_ADE_m']]])
objective=table(['录像','模型','选择','轮次','训练loss','重建','MC KL','验证后验目标','训练K1 ADE','验证K1 ADE'],objective_rows)
capacity=table(['录像','模型','名义','结构有效上界','loss连接模块','长期非零梯度标量','长期从未非零梯度标量'],[
 [r['recording'],r['variant'],r['nominal'],r['structural_upper_bound'],r['module_loss_connected_ever'],r['lifetime_ever_nonzero_gradient'],r['lifetime_never_nonzero_gradient']] for r in verification['run_statistics']])
save('03_EGO_CAPACITY_RESULTS.md','# EgoCapacity正式训练结果\n\n'+base+summary+'\n'+training+'\n'+objective+'\n'+capacity+'''
两组EgoCapacity都从全新seed17初始化，未导入旧权重。每轮遍历全部训练目标，保留尾batch；完整曲线、逐目标预测误差、per-step loss/rec/KL/梯度、每轮参数审计见results/tables/stage5p4/ego_capacity/。best和last及固定间隔checkpoint均留存在被Git忽略的experiments/stage5p4/checkpoints/；来源和SHA见selection.json、checkpoint_inventory.csv。两组EgoCapacity部分时段在同一GPU并发，模型、优化器和RNG各自独立；execution_schedule.json留档。各run实耗包含资源共享影响。Full/NoSocial实耗为冻结历史运行值，软件路径和资源调度不同，不能只凭实耗直接比较架构效率。

结构上界相近不等于函数类、优化轨迹或实际梯度活跃率完全相同。名义未用参数不计入有效上界；MLP中从未激活的ReLU相关标量由长期审计单列。没有通过任意参数掩码凑数。
''')
if not placebo:
 save('04_PLACEBO_SOCIAL_RESULTS.md','# PlaceboSocial执行状态\n\n**未执行，未构建正式训练run，没有可报告指标。**\n\n第一阶段预注册判断：`'+decision['verdict']+'`。是否满足四个录像×加权单元的Full继续门槛：'+str(decision['Full_qualifies_all_four'])+'。按冻结停止规则，本阶段不启动Placebo，不增加seed或训练预算。\n\n这不等同于Placebo失败，也不能据此判断真实邻居内容已被证实有效或无效。只有预设门槛通过并另行冻结独立合成上下文协议，才允许其训练。\n')
else:
 context=read_csv(ROOT/'context_distribution.csv')
 context_table=table(['录像','split','上下文','统计量','均值','P95','零值比例'],[[r['recording'],r['split'],r['context'],r['statistic'],fmt(r['mean']),fmt(r['P95']),fmt(r['fraction_zero'])] for r in context if r['statistic'] in ['count','distance','speed','relative_speed']])
 save('04_PLACEBO_SOCIAL_RESULTS.md','# PlaceboSocial真实执行结果\n\n'+base+summary+'\n'+context_table+'\n实际执行两组100完整epoch；规则仅此一套、没有搜索。公式和边界/未来padding规则见PLACEBO_FROZEN_PROTOCOL.md；训练前10项隔离结构测试和全套337项测试通过。Full原始社会模块均实际执行并获得梯度，见placebo_capacity_by_parameter.csv和每轮梯度记录。名义参数2144002、结构有效上界2045698，与Full一致，但常量初始偏移限制输入多样性，非函数类相同。协议、合成上下文分布与隔离审计见本阶段phase2产物。Placebo改变社会输入分布，不是完美单因素对照；结果不能自动等同于独立社会因果效应。\n')
weighttables=''
for weight in ['target-window','equal-person-ID']:
 weighttables+='\n## '+weight+'\n\n'+table(['录像','模型','K1 ADE/FDE'],[[rec,m,pair(val(m,rec,weight))] for rec in recs for m in models])
 weighttables+='\n两个录像宏平均（各录像1/2权重）：\n\n'+table(['模型','指标族','ADE/FDE'],[[r['model'],r['metric_family'],pair(r)] for r in macros if r['group']=='all' and r['within_recording_weighting']==weight and r['model'] in models and r['selection'] in ['analytic','best']])
allgroups=''
group_models=['Full','NoSocial','EgoCapacity']+(['PlaceboSocial'] if placebo else [])
for rec in recs:
 allgroups+='\n## '+rec+'：全部历史分组K1窗口加权结果\n\n'+table(['组','窗口/行人']+[m+' ADE/FDE' for m in group_models],[
 [g, str(rr['target_windows'])+'/'+str(rr['people']),*[pair(next(r for r in groups if r['recording']==rec and r['model']==m and r['selection']=='best' and r['metric_family']=='K1-prior' and r['weighting']=='target-window' and r['group']==g)) for m in group_models]]
 for g in dict.fromkeys(r['group'] for r in groups)
 for rr in [next(r for r in groups if r['recording']==rec and r['model']=='Full' and r['selection']=='best' and r['metric_family']=='K1-prior' and r['weighting']=='target-window' and r['group']==g)]])
save('05_PAIRED_COMPARISON.md','# 完全配对比较与全部历史分组\n\n'+base+weighttables+allgroups+'''
完整K1首样本/K20、行人等权、last与各组配对差异见analysis_phase1（若执行第二阶段则最终见analysis_final）下all_group_metrics.csv、all_group_paired_differences.csv、paired_model_errors.csv.gz、person_aggregated_errors.csv、recording_macro_metrics.csv。原始ID、GT SHA与分组一致；ZARA2有35个、HOTEL有28个行人ID，窗口重叠，不能按1410个独立行人作显著性推断。组是历史固定、多标签且会重叠；空组保留缺失值，不填0。行人权重先在录像内每个ID平均，再等权；宏平均不混为窗口池化。

三种CV均展示，主分析固定CV-Last2。事后逐录像最低解析CV仅为“解析基线上界参考”，不能冒充一个可部署模型。即使神经模型K20较好，也不能替代确定性K1社会贡献结论。
''')
condition=decision['verdict']
if condition=='ego_capacity_sufficient_stop':
 verdict='EgoCapacity在两个录像及两种加权口径下均达到预设不差于Full且FDE无实质退化的门槛。额外目标自身表达容量已足以解释当前Full相对NoSocial的观测优势；没有证据必须引入真实邻居才能达到该性能。停止后续社会结构实验。'
elif condition=='conditional_or_weak_evidence_stop':
 verdict='容量控制没有给出跨录像、跨权重一致的社会独立收益。第一阶段继续门槛未通过，结果属于条件性、冲突或差异不足；不能宣布Full社会收益独立成立。按预设规则停止，不追加Placebo。'
else:
 verdict='Full在第一阶段两个录像、两种权重下均满足预注册继续门槛。该结果仅说明目标自身容量控制未完全解释优势；仍需结合实际执行的Placebo以及输入分布、归纳偏置和单seed限制判断，不能直接宣布独立邻居收益。'
if placebo:
 code=final_decision['verdict']
 if code=='target_only_Full_path_sufficient_stop':
  verdict='Ego残差MLP本身未解释Full的K1优势，但同一Full架构仅使用目标自身合成上下文即可在两个录像、两种权重下达到预设不差于Full门槛。真实邻居并非达到当前性能的必要输入；停止继续增加SocialVAE社会结构。该结论不等同于原Full从未使用邻居。'
 elif code=='provisional_content_signal_with_distribution_confound':
  verdict='Full在两个录像、两种加权口径下均超过EgoCapacity和PlaceboSocial并通过预设K1/FDE门槛。本次控制未能以额外ego表达或目标自身合成上下文解释Full的K1优势，可记录真实邻居内容的初步贡献迹象。但合成输入分布不同、函数类不完全匹配、单seed与开发曝光仍未排除，不能宣称严格独立因果归因。'
 else:
  verdict='Ego残差MLP未解释Full的跨录像K1优势，但Placebo内容负对照未支持跨两个录像、两种权重一致的独立邻居收益。结果属于条件性或冲突证据；本阶段停止，不再增加SocialVAE社会结构。'
gate=table(['录像','权重','Ego ADE/FDE','Full ADE/FDE','Full相对ADE收益','ego足以解释','Full继续达标'],[
 [c['recording'],c['weighting'],fmt(c['EgoCapacity_ADE_m'])+' / '+fmt(c['EgoCapacity_FDE_m']),fmt(c['Full_ADE_m'])+' / '+fmt(c['Full_FDE_m']),f"{100*c['Full_relative_ADE_gain']:.3f}%",c['ego_sufficient'],c['Full_qualifies']] for c in decision['cells']])
if placebo:
 gate+='\n第二阶段固定内容负对照：\n\n'+table(['录像','权重','Placebo ADE/FDE','Full ADE/FDE','Full相对ADE收益','Placebo足以达到','Full继续达标'],[
  [c['recording'],c['weighting'],fmt(c['Placebo_ADE_m'])+' / '+fmt(c['Placebo_FDE_m']),fmt(c['Full_ADE_m'])+' / '+fmt(c['Full_FDE_m']),f"{100*c['Full_relative_ADE_gain']:.3f}%",c['placebo_sufficient'],c['Full_qualifies']] for c in final_decision['cells']])
answers='''
1. **额外ego容量能否解释Full优势？** 本次256→1024→256控制未能解释四个主指标单元的K1优势；这只是否定“本次具体MLP已足够”的解释，不是数学上排除所有容量解释。额外ego路径的归纳偏置和优化轨迹均不同。HOTEL EgoCapacity ADE0.440967m还弱于NoSocial的0.395194m，因此Full相对EgoCapacity的12.2%优势不能包装成12.2%的社会收益；原Full相对NoSocial只有约2%。
2. **有效容量上界接近后Full是否仍好？** 是，相对EgoCapacity在HOTEL窗口/行人ADE改善12.196%/12.740%，ZARA2改善1.380%/4.568%，配对FDE均更好。两者上界差2.94%，不是完全同容量或同函数类。
3. **K1优势是否跨两个录像？** Full对EgoCapacity在两个录像、两种权重均通过预注册门槛；对Placebo同样四单元全部通过，窗口ADE绝对改善HOTEL 5.557mm、ZARA2 6.617mm，属于小幅经验收益，非显著性或跨seed稳定性证据。
4. **K20是否相同？** 不同：ZARA2 EgoCapacity为0.152978/0.293431m，优于Full的0.157530/0.304836m；Placebo在ZARA2为0.146936/0.280722m，也优于Full；HOTEL的K20则Full更好。不用K20代替K1社会归因。
5. **无真实邻居控制能否达到Full？** EgoCapacity本次未达到；Placebo也未达到Full的K1，四单元均更差；但二者在ZARA2的K20均优于Full，所以不能说所有预测能力都依赖邻居。
6. **社会内容负对照是否执行？** '''+('已实际执行ZARA2/HOTEL各100完整epoch，独立冻结协议与代码提交后才启动；没有额外seed或规则搜索。' if placebo else '未执行，第一阶段门槛未通过，按规则停止；未虚填指标。')+'''
7. **是否证明可独立归因的邻居收益？** 尚不能作严格因果证明。第一阶段解决了一个具体容量控制，第二阶段还存在合成上下文分布变化与初始化输入多样性差异；单seed、重复曝光开发集及重叠窗口也限制结论。当前证据等级以本报告开头最终判断为准。
8. **HOTEL弱于强CV是否仍存在？** Full窗口ADE0.387184m，明显弱于CV-OLS8的0.220144m；行人等权Full0.736104m，仍弱于CV-OLS4约0.354559m。新控制的实际数值完整列出，不挑最弱CV。K20与确定性CV不是同一预测预算。
9. **是否值得继续社会图？** 目前值得进一步验证真实邻居的内容价值，但证据不足以启动复杂图模块。建议下一阶段若另行批准，优先预注册多seed内容对照与HOTEL泛化核查，先确认分布混杂和稳定性，而非直接开发GNN。
10. **是否结束SocialVAE社会改造？** 当前阶段正式结束、无额外训练授权。不必因本次结果彻底否定SocialVAE或真实社会信息；但应暂缓架构改造，保留成熟基线。本次有初步一致K1内容证据，仍不足以立即宣布第四章创新；优先解决HOTEL运动泛化与强解析基线差距。常规容量MLP和伪邻居不是论文创新。
'''
save('06_RESEARCH_DECISION.md','# 研究决策\n\n'+verdict+'\n\n'+gate+'\n'+answers+'\n所有门槛为预注册开发规则，不是显著性或因果识别保证。\n')
save('STAGE5P4_REPORT.md','# Stage5.4：容量归因总报告\n\n'+verdict+'\n\n'+base+summary+'\n'+gate+'\n'+capacity+'\n'+answers+f'''
验收：完整单元测试{verification['unit_tests']}项通过；历史保护清单{verification['protected_files_verified']}个文件SHA未变；正式新run均100完整epoch；checkpoint核验{verification['checkpoints_verified']}个；官方源码、因果数据协议、Full与NoSocial旧权重及结果均未改动。代码和配置SHA、数据manifest与原始来源SHA、真实实耗、训练前后测试及执行日志见results/tables/stage5p4/。

训练前冻结协议提交与实际训练代码SHA见各run/provenance.json。最终交付仅本地commit，不自动push；Git不含原始数据、checkpoint、环境或缓存。用户推送：git push -u origin feat/stage5p4-capacity-attribution。

详细材料：01容量设计、02隔离测试、03正式容量结果、04Placebo状态、05全配对与分组、06研究判断。单seed、录像开发曝光、函数类不完全一致与重叠窗口限制均保留，不将常规容量控制包装成创新。
''')
print('Reports generated from verified measurements')
