"""Generate research delivery from executed measurements; no invented metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
ROOT=Path('results/tables/stage5p4');REPORT=Path('reports/stage5p4')
OUT=ROOT/('analysis_final' if (ROOT/'analysis_final').exists() else 'analysis_phase1')
metrics=read_csv(OUT/'model_metrics.csv');groups=read_csv(OUT/'all_group_metrics.csv');macros=read_csv(OUT/'recording_macro_metrics.csv')
convergence=json.load(open(OUT/'convergence_audit.json'));decision=json.load(open(ROOT/'phase1_decision.json'))
verification=json.load(open(ROOT/'delivery_verification.json'))
recs=['biwi_hotel','crowds_zara02'];models=['CV-Last2','CV-OLS4','CV-OLS8','Full','NoSocial','EgoCapacity']
placebo='PlaceboSocial' in {r['model'] for r in metrics}
if placebo:models.append('PlaceboSocial')
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
training=table(['录像','模型','best轮','best训练ADE/FDE','last验证ADE/FDE','100轮实耗min','峰值显存GiB'],[
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
 save('04_PLACEBO_SOCIAL_RESULTS.md','# PlaceboSocial真实执行结果\n\n'+base+summary+'\n协议、合成上下文分布与隔离审计见本阶段phase2产物。Placebo改变社会输入分布，不是完美单因素对照；结果不能自动等同于独立社会因果效应。\n')
weighttables=''
for weight in ['target-window','equal-person-ID']:
 weighttables+='\n## '+weight+'\n\n'+table(['录像','模型','K1 ADE/FDE'],[[rec,m,pair(val(m,rec,weight))] for rec in recs for m in models])
 weighttables+='\n两个录像宏平均（各录像1/2权重）：\n\n'+table(['模型','指标族','ADE/FDE'],[[r['model'],r['metric_family'],pair(r)] for r in macros if r['group']=='all' and r['within_recording_weighting']==weight and r['model'] in models and r['selection'] in ['analytic','best']])
allgroups=''
for rec in recs:
 allgroups+='\n## '+rec+'：全部历史分组K1窗口加权结果\n\n'+table(['组','窗口/行人','Full ADE/FDE','NoSocial ADE/FDE','EgoCapacity ADE/FDE'],[
 [g, str(rr['target_windows'])+'/'+str(rr['people']),*[pair(next(r for r in groups if r['recording']==rec and r['model']==m and r['selection']=='best' and r['metric_family']=='K1-prior' and r['weighting']=='target-window' and r['group']==g)) for m in ['Full','NoSocial','EgoCapacity']]]
 for g in dict.fromkeys(r['group'] for r in groups)
 for rr in [next(r for r in groups if r['recording']==rec and r['model']=='Full' and r['selection']=='best' and r['metric_family']=='K1-prior' and r['weighting']=='target-window' and r['group']==g)]])
save('05_PAIRED_COMPARISON.md','# 完全配对比较与全部历史分组\n\n'+base+weighttables+allgroups+'''
完整K1首样本/K20、行人等权、last与各组配对差异见analysis_phase1（若执行第二阶段则最终见analysis_final）下all_group_metrics.csv、all_group_paired_differences.csv、paired_model_errors.csv.gz、person_aggregated_errors.csv、recording_macro_metrics.csv。原始ID、GT SHA与分组一致；两个录像分别35与28个行人ID，窗口重叠，不能按1410个独立行人作显著性推断。组是历史固定、多标签且会重叠；空组保留缺失值，不填0。行人权重先在录像内每个ID平均，再等权；宏平均不混为窗口池化。

三种CV均展示，主分析固定CV-Last2。事后逐录像最低解析CV仅为“解析基线上界参考”，不能冒充一个可部署模型。即使神经模型K20较好，也不能替代确定性K1社会贡献结论。
''')
condition=decision['verdict']
if condition=='ego_capacity_sufficient_stop':
 verdict='EgoCapacity在两个录像及两种加权口径下均达到预设不差于Full且FDE无实质退化的门槛。额外目标自身表达容量已足以解释当前Full相对NoSocial的观测优势；没有证据必须引入真实邻居才能达到该性能。停止后续社会结构实验。'
elif condition=='conditional_or_weak_evidence_stop':
 verdict='容量控制没有给出跨录像、跨权重一致的社会独立收益。第一阶段继续门槛未通过，结果属于条件性、冲突或差异不足；不能宣布Full社会收益独立成立。按预设规则停止，不追加Placebo。'
else:
 verdict='Full在第一阶段两个录像、两种权重下均满足预注册继续门槛。该结果仅说明目标自身容量控制未完全解释优势；仍需结合实际执行的Placebo以及输入分布、归纳偏置和单seed限制判断，不能直接宣布独立邻居收益。'
gate=table(['录像','权重','Ego ADE/FDE','Full ADE/FDE','Full相对ADE收益','ego足以解释','Full继续达标'],[
 [c['recording'],c['weighting'],fmt(c['EgoCapacity_ADE_m'])+' / '+fmt(c['EgoCapacity_FDE_m']),fmt(c['Full_ADE_m'])+' / '+fmt(c['Full_FDE_m']),f"{100*c['Full_relative_ADE_gain']:.3f}%",c['ego_sufficient'],c['Full_qualifies']] for c in decision['cells']])
answers='''
1. Full优势能否由额外ego容量解释：以上四单元结果决定，不以名义参数量替代容量审计；控制引入不同归纳偏置，不能宣称容量完全控制。
2. 容量上界匹配后Full是否更好：见四单元门槛及全部配对表，不合并口径掩盖冲突。
3. K1优势跨录像情况：逐录像、窗口和行人等权同时列出；不能由宏平均替代两个录像分别成立。
4. K20是否同结论：见独立K20表；与K1是不同问题，不用K20替代K1归因。
5. 不含真实邻居的控制是否可达到Full：EgoCapacity列与Full直接配对；只报告实际结果。
6. Placebo执行状态：'''+('已执行，见04报告及独立冻结协议。' if placebo else '未执行，第一阶段停止规则不授权启动；没有虚填结果。')+'''
7. 可归因邻居收益：Full优于严格NoSocial本身不足以证明；容量、归纳偏置、随机优化、单seed和开发集曝光仍限制因果解释。
8. HOTEL强CV问题：同一目标CV-OLS4/8及窗口/行人两种权重完整展示；不因只比较最弱CV而忽略明显差距。
9. 是否继续社会图：本阶段不开发图模块；未获跨录像清晰证据时停止增加复杂社会结构。
10. 后续路线：优先完成公开基线/任务适配与泛化归因，保留现有Full/NoSocial作为冻结对照；普通MLP或伪邻居不是新论文方法。不以一个改善分组发起新网络研究。本阶段结束，不追加训练。
'''
save('06_RESEARCH_DECISION.md','# 研究决策\n\n'+verdict+'\n\n'+gate+'\n'+answers+'\n所有门槛为预注册开发规则，不是显著性或因果识别保证。\n')
save('STAGE5P4_REPORT.md','# Stage5.4：容量归因总报告\n\n'+verdict+'\n\n'+base+summary+'\n'+gate+'\n'+capacity+'\n'+answers+f'''
验收：完整单元测试{verification['unit_tests']}项通过；历史保护清单{verification['protected_files_verified']}个文件SHA未变；正式新run均100完整epoch；checkpoint核验{verification['checkpoints_verified']}个；官方源码、因果数据协议、Full与NoSocial旧权重及结果均未改动。代码和配置SHA、数据manifest与原始来源SHA、真实实耗、训练前后测试及执行日志见results/tables/stage5p4/。

训练前冻结协议提交与实际训练代码SHA见各run/provenance.json。最终交付仅本地commit，不自动push；Git不含原始数据、checkpoint、环境或缓存。用户推送：git push -u origin feat/stage5p4-capacity-attribution。

详细材料：01容量设计、02隔离测试、03正式容量结果、04Placebo状态、05全配对与分组、06研究判断。单seed、录像开发曝光、函数类不完全一致与重叠窗口限制均保留，不将常规容量控制包装成创新。
''')
print('Reports generated from verified measurements')
