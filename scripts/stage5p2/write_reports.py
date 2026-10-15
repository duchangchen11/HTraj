"""Render actual results only after both preregistered runs complete."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new

R=Path('results/tables/stage5p2');D=Path('reports/stage5p2')
RECS=['biwi_hotel','crowds_zara02']
LABEL={'biwi_hotel':'HOTEL','crowds_zara02':'ZARA2'}


def number(x,digits=4):
    return '无目标/未评价' if x is None or x=='' else f'{float(x):.{digits}f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def write(name,text):
    with (D/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    audit=json.load(open(R/'convergence_audit.json'));metrics=read_csv(R/'model_metrics.csv')
    groups=read_csv(R/'all_group_metrics.csv');macro=read_csv(R/'recording_macro_metrics.csv')
    oracle={r['recording']:r for r in json.load(open(R/'analytic_oracle_reference.json'))}
    selects={rec:json.load(open(R/rec/'selection.json')) for rec in RECS}
    def metric(rec,model='SocialVAE',selection='best',family='K1-prior',weight='target-window'):
        return next(r for r in metrics if r['recording']==rec and r['model']==model and r['selection']==selection and r['metric_family']==family and r['weighting']==weight)
    def cv(rec,model='CV-Last2',weight='target-window'):return metric(rec,model,'analytic','deterministic',weight)
    for rec,filename in [('crowds_zara02','02_ZARA2_TRAINING_RESULTS.md'),('biwi_hotel','03_HOTEL_TRAINING_RESULTS.md')]:
        a=audit[rec];s=selects[rec];b=a['best_validation'];last=a['last_epoch'];bestepoch=a['best_epoch']
        rowtable=[]
        for sel in ['best','last']:
            for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
                m=metric(rec,selection=sel,family=family)
                rowtable.append([sel,bestepoch if sel=='best' else 100,family,number(m['ADE_m']),number(m['FDE_m'])])
        trends=table(['指标','81–90轮均值','91–100轮均值','后十轮−前十轮','末20轮线性斜率/轮'],[
            [field,*[number(values[k],6) for k in ['epoch81_90','epoch91_100','difference','slope_per_epoch']]] for field,values in a['last20'].items()])
        text=f'''# {LABEL[rec]} 正式训练结果

已真实完成 seed17 的100个完整epoch；每轮 {s['training_targets_per_epoch']} 个训练目标窗口，累计 {s['total_optimizer_steps']} 次 optimizer 更新。仅训练集更新参数，验证录像整体排除。模型独立随机初始化，不加载任何checkpoint。训练代码提交 `{s['code_commit']}`。

{table(['选择','epoch','评价口径','ADE/m','FDE/m'],rowtable)}

best 按验证 K1 latent-prior ADE 唯一选择，第 {bestepoch} 轮；FDE、首样本与 K20 都来自同一个 checkpoint，不按各自最低值拼接。last 为第100轮。K20 两项分别按候选取最小值，可能来自不同候选；FPC关闭。

## 训练与泛化诊断

- 参数量：2,144,002；未引入新参数。原始训练目标仍为 reconstruction + MC latent KL，不是精确边缘NLL。
- 第1轮训练总loss {number(a['first_epoch']['train_loss'],6)}，第100轮 {number(last['train_loss'],6)}（rec {number(last['train_rec'],6)} + KL {number(last['train_kl'],6)}）。训练曲线是更新中各batch的目标数加权均值。
- best 固定权重完整训练集 K1 ADE/FDE：{number(a['best_train_K1']['train_K1_prior_ADE_m'])}/{number(a['best_train_K1']['train_K1_prior_FDE_m'])}m；验证：{number(b['K1_prior_ADE_m'])}/{number(b['K1_prior_FDE_m'])}m。训练集也含重叠窗口，不能当独立个体样本。
- best 验证 posterior loss/rec/KL：{number(b['posterior_loss'],6)}/{number(b['posterior_rec'],6)}/{number(b['posterior_kl'],6)}。这是使用未来GT的监督posterior诊断；预测与分组不读取未来GT。
- 完整run墙钟 {a['actual_run_wall_seconds']:.2f}s（{a['actual_run_wall_seconds']/60:.2f}min），训练GPU计算合计 {a['sum_training_compute_seconds']:.2f}s；峰值allocated {a['peak_allocated_bytes']/2**30:.3f}GiB，reserved {a['peak_reserved_bytes']/2**30:.3f}GiB；梯度L2最大值 {a['gradient_L2_max']:.4f}；全部逐轮参数有限性检查通过。
- 记录了100份不可覆盖checkpoint及SHA；第1、10……100轮和best的模型/optimizer/RNG恢复测试通过。该run未发生NaN/Inf导致的停止或重试。

{trends}

最优点{'位于最后10轮，存在预算边界影响，不能宣布充分收敛' if a['best_in_last10'] else '不在最后10轮，但这不足以证明所有指标已收敛'}。训练loss下降不保证验证ADE下降；随机posterior目标与确定性latent-prior轨迹ADE不同。末段斜率仅描述这一固定预算，未据此调参或延长训练。

## 可追溯文件

- `results/tables/stage5p2/{rec}/provenance.json`：来源、manifest SHA、初始权重fingerprint、环境、code commit。
- `training_curve.csv` 与 `epochs/epochNNN.json`：全部100轮；训练集K1仅在预注册轮次计算，其他轮留空。
- `epochs/epochNNN_steps.csv.gz`：每个batch的原始loss分项、梯度和耗时；`epochNNN_targets.csv.gz`：完整验证集逐目标误差。
- `checkpoint_inventory.csv`、`selection.json`、`checkpoint_restore_checks.json`：权重位置/SHA、配对选择和恢复。
- best `{s['best']['path']}`，SHA `{s['best']['sha256']}`；last `{s['last']['path']}`，SHA `{s['last']['sha256']}`。权重仅本地保存且被Git忽略。
'''
        write(filename,text)
    cvtable=[]
    for rec in RECS:
        for model in ['CV-Last2','CV-OLS4','CV-OLS8']:
            m=cv(rec,model);p=cv(rec,model,'equal-person-ID')
            cvtable.append([LABEL[rec],model,m['target_windows'],m['people'],number(m['ADE_m']),number(m['FDE_m']),number(p['ADE_m']),number(p['FDE_m'])])
        for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
            m=metric(rec,family=family);p=metric(rec,family=family,weight='equal-person-ID')
            cvtable.append([LABEL[rec],'SocialVAE-best '+family,m['target_windows'],m['people'],number(m['ADE_m']),number(m['FDE_m']),number(p['ADE_m']),number(p['FDE_m'])])
    macrotable=[]
    for m in macro:
        if m['group']=='all' and m['selection']!='last':
            macrotable.append([m['model'],m['metric_family'],m['within_recording_weighting'],number(m['ADE_m']),number(m['FDE_m'])])
    write('04_CV_COMPARISON.md',f'''# 完全配对CV比较

所有三种CV均重新运行；两个录像合计1410目标窗口、63个录像-行人ID单位。目标ID、历史坐标SHA和未来GT SHA与Stage5.1逐项一致，三种CV既有逐目标误差也逐项复核。未借用旧聚合数值代替重算。

{table(['录像','模型/口径','目标窗口','行人ID','窗口ADE','窗口FDE','行人等权ADE','行人等权FDE'],cvtable)}

主对照固定CV-Last2，不能按录像更换主基线。OLS4/8通过历史位置与物理秒OLS估计速度，预测均锚定最后历史位置。各录像最低解析误差：

{table(['录像','ADE最优CV','最低ADE','FDE最优CV','最低FDE'],[[LABEL[r],oracle[r]['minimum_ADE_model'],number(oracle[r]['minimum_ADE_m']),oracle[r]['minimum_FDE_model'],number(oracle[r]['minimum_FDE_m'])] for r in RECS])}

上表只是**解析基线上界参考**（事后逐录像选择最强解析基线），不是一个可部署的新模型。更低误差在指标意义上是误差下界；“上界参考”指该有限CV集合可达到的性能参考，不能作统计理论界限。

## 两录像宏平均（各录像等权）

{table(['模型','评价口径','录像内权重','宏ADE','宏FDE'],macrotable)}

窗口加权先在录像内按目标窗口平均；行人等权先在每个录像-行人ID内部平均，再平均该录像的行人。宏平均最后对两个录像各赋1/2权重。不能把按所有1410窗口直接合并的数值称为录像宏平均。

K1 latent-prior是预先指定的确定性评价；K1首样本是固定seed的第一条随机轨迹；K20利用20条候选和未来GT选择最优，是不同预测口径。K20优于CV不能推出单轨迹预测优于CV，也不能证明社会消息的独立贡献。

完整配对文件：`paired_model_errors.csv.gz`；完整汇总 `model_metrics.csv`、`person_aggregated_errors.csv`、`recording_macro_metrics.csv`；强CV参考 `analytic_oracle_reference.json`。
''')
    grouptext=[]
    for rec in RECS:
        rr=[r for r in groups if r['recording']==rec and r['model']=='SocialVAE' and r['selection']=='best' and r['metric_family']=='K1-prior' and r['weighting']=='target-window']
        grouptext.append('## '+LABEL[rec]+'：全部历史分组，best K1\n\n'+table(['组','窗口数','ID数','ADE','FDE','ΔADE−Last2','ΔFDE−Last2'],[
            [r['group'],r['target_windows'],r['people'],number(r['ADE_m']),number(r['FDE_m']),number(r['delta_ADE_vs_CV_Last2_m']),number(r['delta_FDE_vs_CV_Last2_m'])] for r in rr]))
    write('05_ERROR_AND_GENERALIZATION.md','''# 误差与泛化分析

全部分组阈值在训练前提交，定义见01报告与 `configs/stage5p2/protocol.json`。仅历史8帧决定组别，未用未来GT筛选输入或分组。静止指历史速度均值<0.05m/s，不代表未来静止。方向不可观测、空组和退化组均保留；空组没有指标，不填0。

下表Δ为SocialVAE减CV-Last2，负值表示改善。组标签可能重叠（多邻居运动关系尤其如此），不能相加为总体目标数。无合法邻居指初始化位置社会路径及七个运动消息时刻均没有可用2m邻居，不等于整个录像只有一个人。末帧人数和历史平均邻居密度分别记录，不能混用。

'''+ '\n'.join(grouptext)+'''

## 聚合和证据边界

全部模型、best/last、K1/K20、窗口/行人两种权重的所有组（包括0目标行）均见 `results/tables/stage5p2/all_group_metrics.csv`。逐目标配对差值不是独立行人检验；按行人聚合能降低长轨迹重复窗口的权重，但同录像不同个体也存在共享环境，不能由此声称独立统计显著性。这里只有一个训练seed和两个历史曝光的开发录像。

训练/验证K1差距、末20轮趋势见两个训练报告及 `convergence_audit.json`。验证posterior目标和确定性预测不同，KL估计单次/均值可以带随机误差，不据NLL术语或单点曲线宣称收敛。最佳轮由同一验证集反复选择，有开发选择偏差；本阶段不读取ETH。

历史运动组中某些收益只能作为条件性描述。即使“存在邻居”组改善，也不能排除ego运动建模或额外容量的贡献，仍需未来独立训练的完整NoSocial对照。
''')
    k1wins=[r for r in RECS if float(metric(r)['ADE_m'])<float(cv(r)['ADE_m'])]
    kwins=[r for r in RECS if float(metric(r,family='K20-independent-minima')['ADE_m'])<float(cv(r)['ADE_m'])]
    bestcvwins=[r for r in RECS if float(metric(r)['ADE_m'])<oracle[r]['minimum_ADE_m']]
    evidence=dict(K1_ADE_better_than_Last2_recordings=k1wins,K20_minADE_better_than_Last2_recordings=kwins,
        K1_ADE_better_than_strongest_analytic_recordings=bestcvwins,
        completed_100_epochs_both=True,ETH_evaluated=False,NoSocial_trained=False)
    json_new(R/'decision_evidence.json',evidence)
    decision=f'''# 研究决策

本阶段两组正式训练均真实完成100个完整epoch，无新增网络/损失，未加载smoke或官方训练权重。结论只针对两个历史曝光开发录像的seed17，不是独立确认或官方论文精确复现。

1. **能否完成训练？** 能。ZARA2 {selects['crowds_zara02']['total_optimizer_steps']}步、HOTEL {selects['biwi_hotel']['total_optimizer_steps']}步；梯度和逐轮参数均有限，严格checkpoint恢复通过。
2. **K1是否两个录像都超过CV-Last2？** K1 ADE改善的录像为：{', '.join(LABEL[r] for r in k1wins) or '无'}。两项配对ADE/FDE必须结合04报告，不用最低FDE轮替换。
3. **与各录像最强解析CV相比？** K1 ADE改善录像为：{', '.join(LABEL[r] for r in bestcvwins) or '无'}。HOTEL需同时看OLS8，ZARA2需同时看Last2；事后选择只作解析基线上界参考。
4. **K1和K20结论相同吗？** K20 minADE优于Last2的录像为：{', '.join(LABEL[r] for r in kwins) or '无'}。K20是一对多采样再用GT评分，与单轨迹K1不是同一任务。latent-prior轨迹不是数学期望，首随机样本也不代表平均采样质量。
5. **过拟合？** 按02/03报告同一best权重的完整训练与验证K1差距、末20轮训练loss和验证趋势判断。录像分布迁移、选择噪声与过拟合可能共存；不能只靠训练loss下降作因果解释。
6. **哪些运动类型获益？** 05报告完整列出静止/低速/正常/高速、转弯、方向变化及不可观测组的同目标Δ；不只展示正向案例。样本少的组仅描述，不作显著性结论。
7. **哪些交互场景退化？** 05报告列出所有有/无合法邻居、邻居数量/密度、新进入、未知速度与相对方向组；Δ>0即退化。这些是条件性误差差异，尚不能独立归因社会机制。
8. **100轮预算足够吗？** ZARA2 best第{audit['crowds_zara02']['best_epoch']}轮，HOTEL best第{audit['biwi_hotel']['best_epoch']}轮；最后10轮最优标记分别为{audit['crowds_zara02']['best_in_last10']}、{audit['biwi_hotel']['best_in_last10']}。固定预算内可判断开发表现，不能仅凭100轮证明收敛；末端仍改善须记录预算截断，不自动延长。
9. **是否值得完整NoSocial？** {'在相同任务/采样/容量规则下预注册完整NoSocial值得作为下一阶段候选，以检验已有多模态预测收益是否依赖社会信息。' if kwins else '现有预测能力证据不足，先审查基线/任务与采样，不立即堆叠图模块；NoSocial是否启动需下一阶段决策。'} 本阶段不运行它，也不以推理屏蔽代替从头训练。
10. **是否适合第四章？** {'可暂作为可运行的公开多模态开发基础，但尚无NoSocial独立归因、多种子或未曝光验证，不能据此确定第四章新方法成立。' if kwins else '目前不足以确认为第四章基础；保留可运行实现与失败证据，优先核查任务/基线。'} 无论结果如何，常规VAE、坐标处理、CV或采样控制均不是本项目的新创新点。

## 停止与下一阶段边界

本阶段到两个固定run和验收结束，不追加epoch、不换录像、不启动其他seed/NoSocial/新图网络或ETH测试。下一阶段若获授权，应先冻结NoSocial的两条社会信息路径（历史位置初始化和运动注意力）、相同参数与训练/评价口径，优先检验社会信息而非增加模块。任何第四章创新判断仍待独立消融与更广泛的稳定性证据。
'''
    write('06_RESEARCH_DECISION.md',decision)
    overview=[]
    for rec in RECS:
        for model in ['CV-Last2','CV-OLS4','CV-OLS8']:
            m=cv(rec,model);overview.append([LABEL[rec],model,'—',number(m['ADE_m']),number(m['FDE_m'])])
        for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
            m=metric(rec,family=family);overview.append([LABEL[rec],'SocialVAE '+family,audit[rec]['best_epoch'],number(m['ADE_m']),number(m['FDE_m'])])
    write('STAGE5P2_REPORT.md',f'''# Stage 5.2 正式SocialVAE基线总报告

已完成两个预注册独立seed17 run，各100个完整epoch；训练/验证精确沿用Stage5.1因果manifest。官方网络与原始重建+MC-KL损失保持冻结，FPC关闭，未读取ETH指标，未启动额外训练。训练配置与代码先提交，再执行正式参数更新；执行提交 `{selects[RECS[0]]['code_commit']}`。

{table(['录像','模型/指标','best epoch','ADE/m','FDE/m'],overview)}

上表SocialVAE三个口径共用该录像K1-prior ADE选定的best checkpoint；K20独立最小ADE/FDE。完整last及逐轮结果没有省略，见02/03报告。三个CV使用完全相同的窗口、目标和GT，重新计算并核对旧SHA；没有选择性只展示弱CV。

## 交付索引

- `01_FROZEN_TRAINING_PROTOCOL.md`：预注册配置、采样、阈值、选择与停止规则。
- `02_ZARA2_TRAINING_RESULTS.md`、`03_HOTEL_TRAINING_RESULTS.md`：best/last、完整预算、趋势、耗时/显存、checkpoint SHA。
- `04_CV_COMPARISON.md`：三个CV、K1/K20分离、窗口/行人等权与录像宏平均。
- `05_ERROR_AND_GENERALIZATION.md`：全部历史运动与交互分组，含退化/不可观测/空组。
- `06_RESEARCH_DECISION.md`：十项问题、证据边界及停止决定。
- `results/tables/stage5p2/`：全部真实配置来源、逐轮逐batch、逐目标配对/分组、采样seed/noise SHA、模型/数据SHA、完整执行日志和测试结果。

## 限制与验收

两个录像已有历史研究曝光，只有seed17；重叠窗口不是独立行人。K20相对CV的收益不是社会模块独立贡献，也不能替代确定性K1结论。VAE训练目标不称精确边缘NLL，latent-prior轨迹不称预测数学期望。

验收：完整单元测试见 `results/tables/stage5p2/final_tests.xml` 与 `final_tests.txt`；旧2569文件SHA和200份新checkpoint SHA复核见 `delivery_verification.json`、`checkpoint_sha_verification.csv`。保护结果以这些真实运行记录为准，不覆盖旧权重/指标。权重/原始数据/缓存不进入Git，只完成本地commit，不自动push。

用户手动推送：`git push -u origin feat/stage5p2-socialvae-baseline`。本阶段完成后停止，等待下一步科研决策。
''')


if __name__=='__main__':main()
