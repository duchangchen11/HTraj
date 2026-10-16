"""State preregistered verdict without promoting capacity-confounded gains."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from scripts.stage5p3.write_reports import write,table,n,ROOT,DOC,REC,LABEL

rule=json.load(open(ROOT/'stopping_decision.json'))
overall=json.load(open(ROOT/'social_effects_overall.json'))
rows=[]
for rec in REC:
    for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
        for weight in ['target-window','equal-person-ID']:
            r=next(r for r in overall if r['recording']==rec and r['selection']=='best' and r['metric_family']==family and r['weighting']==weight)
            rows.append([LABEL[rec],family,weight,n(r['Full_ADE_m'])+'/'+n(r['Full_FDE_m']),n(r['NoSocial_ADE_m'])+'/'+n(r['NoSocial_FDE_m']),n(r['Full_minus_NoSocial_ADE_m'])+'/'+n(r['Full_minus_NoSocial_FDE_m'])])
verdict={'worth_followup_capacity_unresolved':'两个录像在两种权重均达到预设K1门槛，值得进一步验证，但容量混杂未排除，不能宣称社会信息独立有效。',
    'conditional_or_weighting_conflict':'收益仅具条件性，或在录像/权重/误差指标之间冲突，未获得跨录像稳定社会收益证据。',
    'stop_complex_social_modules':'两个录像均未显示Full的K1净改善，按预注册规则停止基于SocialVAE增加复杂社会模块。'}[rule['verdict']]
text=f'''# 社会归因研究决策

**冻结规则判断：{verdict}**

{table(['录像','指标口径','权重','Full ADE/FDE','NoSocial ADE/FDE','Full−NoSocial ADE/FDE'],rows)}

主归因预先固定K1 latent-prior ADE，FDE配对。相对ADE改善≥1%、FDE退化≤1%，且两个录像及窗口/行人两权重均满足，才称值得进一步验证；更小差异或冲突只描述。K20单独分析，不能取代K1社会收益；首采样也全部报告。这个阈值是开发阶段决策标准，不是统计显著性。

## 容量与因果边界

严格NoSocial切断所有社会输入，保留原VAE和decoder。Full/NoSocial名义参数均2144002，但结构有效上界分别2045698/1459906，约28.6%差异。Full若更好，不能排除额外有效容量；NoSocial若更好，只表明此训练协议下Full没有净开发优势，不能证明社会信息在所有任务上无预测价值。

Ego-Capacity-Control只有03报告中的单独方案和拟定预算，**未实现、未训练、没有性能指标**。新增ego MLP也改变归纳偏置，本阶段不强行将其包装成严格单因素容量证明。容量差异审计已完成，性能层面的容量排除仍未完成；任何“独立社会贡献已证实”的说法都不受本阶段支持。

## 研究问题的回答

1. Full的社会入口是否完整识别？位置初始化、历史运动消息和后验邻居入口已核查。当前因果协议下未来邻居入口实际padding为0，后验仍受历史社会状态影响。
2. NoSocial是否真正隔离邻居？人工干预覆盖位置/速度/数量/排列/进入时间和非有限邻居，训练、后验、K1/K20均不变，原始输入Jacobian无依赖。Full的no_grad会让原始Jacobian也空，故另用真实干预确认其社会功能。
3. Full是否更好？见上表及全部逐目标配对文件；不能用不同目标、K20或最低FDE轮替代主K1比较。
4. ZARA2正常行走收益是否依赖社会？比较相同正常组Full−NoSocial，同时看两种权重，结果见05与最终验收补充。即使Full较好，容量未排除；不能由相对CV改善直接归因社会。
5. HOTEL正常行走退化是否来自社会？同样看Full与NoSocial以及两者相对CV，不能只看Full−CV。组样本有重复窗口，不能按窗口作独立显著性检验。
6. 是否能排除容量？不能，已明确区分有效上界、观察到的梯度和名义参数。没有伪造未运行控制。
7. 训练是否充分？两组NoSocial完成固定100完整epoch；各自best位置、训练/验证差距和末20轮趋势见04与`convergence_audit.json`。100轮不自动等于收敛，不追加训练。
8. 是否继续复杂交互改造？{verdict} 不新增GNN、高阶、关系GRU、attention或gate，不把普通VAE/消融称为新创新。
9. 是否已有独立确认？没有。两个录像均有历史研究曝光，只有seed17；窗口重叠且个体共享环境，ID聚合不等于独立样本。
10. 下一阶段做什么？先依据本阶段容量和条件性证据另行决策；若需要排除容量，独立冻结并实现/验证03中的目标专用容量控制，或重新选择任务证据。当前不自动追加任何训练或ETH评价。

## 执行停止

完成两个NoSocial run、完整测试、旧SHA及checkpoint验证和本地提交后停止。Full结果保持冻结，不根据NoSocial反向修改它。不追加seed42/2026，不自动push。
'''
write('06_RESEARCH_DECISION.md',text)
ns=json.load(open(ROOT/'convergence_audit.json'));full=json.load(open('results/tables/stage5p2/convergence_audit.json'))
mainrows=[]
for rec in REC:
    for label,source in [('Full',full),('NoSocial',ns)]:
        a=source[rec];b=a['best_validation']
        mainrows.append([LABEL[rec],label,a['best_epoch'],n(b['K1_prior_ADE_m'])+'/'+n(b['K1_prior_FDE_m']),n(b['minADE20_m'])+'/'+n(b['minFDE20_m']),n(a['actual_run_wall_seconds']/60,2)])
write('STAGE5P3_REPORT.md',f'''# Stage 5.3 社会信息归因总报告

{verdict}

两个NoSocial模型均以seed17全新初始化，分别完整训练100epoch；Full严格复用Stage5.2冻结结果，官方源码未改，未增加新图模块/损失。8/12帧、0.4s、米制因果协议、训练/目标/GT、选择与采样完全配对；ETH未评价。

{table(['录像','模型','best epoch','K1 ADE/FDE','minADE20/minFDE20','实际分钟'],mainrows)}

K1与K20不能混用；同模型各项来自同一K1 ADE选择的best。last和首采样均保留在04及真实结果文件。全部CV-Last2/OLS4/OLS8见04，不选弱基线突出神经网络。

## 交付索引

- `01_SOCIAL_INFORMATION_PATH_AUDIT.md`：源码路径、分层信息流、未来邻居实际mask和Jacobian detachment限制。
- `02_NOSOCIAL_IMPLEMENTATION.md`：最小功能修改、训练前冻结协议及停止门槛。
- `03_CAPACITY_AUDIT.md`：名义/有效/梯度容量，独立容量控制设计及未运行边界。
- `04_TRAINING_RESULTS.md`：两录像Full/NoSocial/CV的best/last、各预测口径、实际耗时、曲线/泛化。
- `05_INTERACTION_GROUP_ANALYSIS.md`：全部历史分组、窗口与行人两种权重，含空组和退化组。
- `06_RESEARCH_DECISION.md`：预设规则结论、容量混杂、下一步和停止。
- `results/tables/stage5p3/`：代码/数据SHA、逐轮逐batch/逐参数梯度、逐目标配对误差、全部组/宏平均、checkpoint SHA、测试和日志。

## 验收和限制

完整单元测试与所有原始失败记录按实际执行保存。`delivery_verification.json`给出旧文件SHA保护、完整训练预算、200份新checkpoint校验和测试结果；不得将结构测试等同预测性能验证。Stage5.2 CUDA跨batch拆分毫米级数值限制继续保留，评价固定相同batch128/目标顺序/目标epsilon，不修改容差或指标挑选。

没有独立确认集、额外seed或容量控制性能，不能宣布独立社会因果贡献或第四章新创新成立。本阶段无Ego-Capacity-Control训练，未运行项不填0。只完成本地commit，用户自行`git push -u origin feat/stage5p3-social-attribution`；完成后停止。
''')
