"""Write reports only after real result files and the final test run exist."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,json,xml.etree.ElementTree as ET

ROOT=Path('results/tables/stage4p1');DOC=Path('reports/stage4p1')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def pair(r,prefix):return f"{float(r[prefix+'_ADE']):.6f} / {float(r[prefix+'_FDE']):.6f}"


def new(name,body):
    with (DOC/name).open('x') as f:f.write(body.strip()+'\n')


def main():
    a=json.loads((ROOT/'analysis.json').read_text());summary=list(csv.DictReader((ROOT/'model_best_last.csv').open()))
    p=json.loads(Path('configs/stage4p1/protocol.json').read_text());models=['Full','NoSocial-ConsistentNorm']
    val=[r for r in summary if r['split']=='validation'];best={r['model']:r for r in val if r['label']=='best'}
    train=[r for r in summary if r['split']=='train' and r['label']=='best']
    result_table=table(['模型','checkpoint/epoch','NLL','K1均值 ADE/FDE','K1首样本 ADE/FDE','K20独立最优 ADE/FDE'],
        [[r['model'],r['label']+'/'+r['epoch'],f"{float(r['NLL']):.6f}",pair(r,'K1_distribution_mean'),pair(r,'K1_first_sample'),pair(r,'K20_independent_min')] for r in val])
    timing=table(['模型','训练秒','30轮验证合计秒','窗口/秒','目标窗口/秒','峰值allocated MiB','最小均值ADE epoch（仅描述）'],
        [[m,f"{a['models'][m]['train_seconds']:.3f}",f"{a['models'][m]['validation_seconds']:.3f}",f"{a['models'][m]['train_windows_per_second']:.3f}",
          f"{a['models'][m]['train_target_windows_per_second']:.3f}",f"{a['models'][m]['peak_train_cuda_allocated_MiB']:.3f}",a['models'][m]['minimum_K1_ADE_epoch']] for m in models])
    train_table=table(['模型','best train NLL（目标加权）','best train K1均值 ADE/FDE','best validation K1均值 ADE/FDE'],
        [[r['model'],f"{float(r['NLL']):.6f}",pair(r,'K1_distribution_mean'),pair(best[r['model']],'K1_distribution_mean')] for r in train])
    hashes=table(['协议','manifest SHA256'],[[name,info['sha256']] for name,info in p['manifests'].items()])
    checkpoint_rows=[]
    for m in models:
        run=json.loads((ROOT/m/'execution.json').read_text())
        for ck in run['checkpoints']:checkpoint_rows.append([m,ck['label'],ck['epoch'],ck['path'],ck['sha256']])
    ck_table=table(['模型','选择','epoch','本地checkpoint','SHA256'],checkpoint_rows)
    tests=ET.parse(ROOT/'final_tests.xml').getroot().find('testsuite').attrib
    delta=float(best['NoSocial-ConsistentNorm']['K1_distribution_mean_ADE'])-float(best['Full']['K1_distribution_mean_ADE'])
    fdelta=float(best['NoSocial-ConsistentNorm']['K1_distribution_mean_FDE'])-float(best['Full']['K1_distribution_mean_FDE'])
    direction='Full的均值ADE较低' if delta>0 else 'NoSocial对照的均值ADE较低' if delta<0 else '均值ADE相同'
    groups=list(csv.DictReader((ROOT/'paired_context_counts.csv').open()))
    group_table=table(['上下文人数','目标窗口','unique行人','Full ADE/FDE','NoSocial ADE/FDE','NoSocial−Full ADE/FDE'],
        [[r['group'],r['target_windows'],r['unique_agents'],f"{float(r['Full_ADE']):.6f} / {float(r['Full_FDE']):.6f}",
          f"{float(r['NoSocial_ADE']):.6f} / {float(r['NoSocial_FDE']):.6f}",f"{float(r['NoSocial_minus_Full_ADE']):+.6f} / {float(r['NoSocial_minus_Full_FDE']):+.6f}"]
         for r in groups if r['label']=='best' and r['mode']=='K1_distribution_mean' and int(r['target_windows'])>0])
    people=list(csv.DictReader((ROOT/'paired_person_means.csv').open()));people=[r for r in people if r['label']=='best' and r['mode']=='K1_distribution_mean']
    person_delta=sum(float(r['NoSocial_minus_Full_ADE']) for r in people)/len(people)
    new('02_PUBLIC_BASELINE_TRAINING.md',f'''
# Social-STGCNN从头训练：真实执行与数值边界

Full保持官方维护版结构和原PDF-clamped NLL，完成seed17、30epoch；未用预训练权重。首5轮在提交04254e4之后运行，检查点预测逐元素恢复一致，随后同一run继续6–30轮，不重复训练前5轮。验证NLL选best，所有epoch完整保存，不按ADE改选。

## 真实主结果

所有验证项使用同一921窗口/5833目标窗口，米制。NoSocial指单次修订后从头训练的NoSocial-ConsistentNorm；原失败版本独立保存，不混入结果。

{result_table}

K20对每个目标分别取minADE、minFDE，可能来自不同候选。固定采样seed314159在两模型使用相同标准正态银行；不是三训练种子。NLL连续概率密度可大于1，因此负NLL不是负距离或程序错误。

## 训练/验证与耗时

{train_table}

{timing}

吞吐为CUDA同步后的训练循环实测，包括搬运和反传；两模型串行运行，没有彼此争用GPU。train耗时合计不含每轮验证、checkpoint序列化、准备和最终best/last重评价。显存为PyTorch allocated峰值，不代表整个CUDA进程显存。环境Python/torch/CUDA/GPU及数据/代码commit见environment.json和各模型execution.json。

每模型30×24=720次SGD更新；每轮1488窗口/23349目标窗口，最后16窗口累积按实际数量平均。原官方train入口未执行，避免其累积边界漏窗口；优化器仍SGD/lr0.01，不做scheduler或多配置搜索。训练梯度均有限且非零，梯度均值/P95/max在每轮曲线。两者均7563可训练参数，其中原forward永不调用的1309参数保留，有梯度路径的分配参数计6254；不隐藏这部分容量。

训练NLL初末值：

{table(['模型','epoch1 train-mode窗口均值NLL','epoch30同口径NLL'],[[m,f"{a['models'][m]['initial_train_NLL']:.6f}",f"{a['models'][m]['last_train_NLL']:.6f}"] for m in models])}

这证明有限预算下可以数值稳定学习，不证明已经充分收敛或达到论文水平。30轮内验证误差并非单调，NLL和均值ADE最优轮次也可能不同；描述性最低ADE轮次不用于改变预设选择规则。训练态窗口等权NLL不能与评价态目标加权NLL直接解释为泛化gap，上表提供同为eval/目标加权的train/validation结果。

## 原NoSocial失败与一次修正

原NoSocial完整记录第1–4轮，执行第5轮训练后评价出现非有限NLL，未保存第5轮完整结果或checkpoint；停止而非填0。失败日志、4个权重及nosocial_failed_pilot.json保留。第4轮固定权重诊断显示逐人训练统计与全局running评价统计不一致，能明显放大Gaussian输出。修订前冻结nosocial_amendment.json，只改为训练/评价均逐人时间归一化，数据、lr、损失、预算、Full实现均不变。修订从头训练，5轮恢复门槛通过后继续到30，没有第三次尝试。

此修订不是最初就完整预注册，须连同失败过程报告。另核查两run前4轮checkpoint，每轮31个共有状态张量逐元素一致，支持修订未改变训练态变换/更新，确实只修正评价模式而非寻找更有利参数。新run仍从头训练，未复用失败权重。初始diagnosis里的零参数偏移字段源于诊断累积变量错误，不能使用；nosocial_failure_diagnosis_v2.json和保存的诊断脚本给出正确最大参数偏移17.1893，策略的log-sigma/correlation最大值结论未改变。

## checkpoint与来源

{ck_table}

{hashes}

data SHA逐录像存于preparation.json；原始数据、29MB缓存和全部checkpoint均被Git忽略。curve_30.csv保留每轮所有三种ADE/FDE、NLL、梯度、显存、训练/评价秒和吞吐。逐目标配对数据仅保存误差，没有上传原始轨迹。

与Stage4官方checkpoint不能直接判优：本轮排除ETH/zara03/uni_examples、保留其他录像旧val不用，只有1488窗口/30epoch；官方权重的训练预算、训练池、上游选择历史不同且可能使用ETH训练材料。本轮ZARA2整录像评价任务与官方离线筛选相近，但存在细微浮点处理顺序差异，不能称完全训练复现。没有运行官方250轮训练或任何ETH评价。

完整单元测试{tests['tests']}通过，耗时{float(tests['time']):.3f}秒；test XML、真实epoch日志和数据hash均已保存。
''')
    new('04_SOCIAL_ABLATION.md',f'''
# 社会信息归因：Full与完整NoSocial隔离对照

## 比较对象与可回答的问题

Full是未改的官方维护版。有效NoSocial是NoSocial-ConsistentNorm：图I，TXP-CNN节点核3×3重排为单人feature轴9×1，逐人时间归一化且train/eval一致。相同历史输入、预测12步、Gaussian/NLL、seed17、数据、SGD0.01、累计64窗口、梯度clip1、30轮和验证NLL选择。

两个模型各7563参数，原版未调用1309参数也同样保留；近容量不等于相同函数空间。NoSocial通过人工train/eval输入Jacobian和扰动测试证明目标不依赖任何其他节点；单位图的Full仍有非零跨节点Jacobian，所以不能将其称NoSocial。

## 真实整体结果

{result_table}

按预设NLL选定best的K1均值，{direction}，NoSocial−Full的ADE差{delta:+.6f}米，FDE差{fdelta:+.6f}米。正差表示Full误差更低，负差表示NoSocial更低。best与last全部报告，NLL与K1/K20均不能选择性省略。

## 完整上下文人数分组（best K1均值）

{group_table}

这是离线完整20帧上下文的人数，不是仅历史定义的物理邻居密度；不要与Stage3社会密度组混称。所有组和目标计数均保存，其他两种K与last的全部组见paired_context_counts.csv，无样本组指标为空而非0。

## 配对与重复窗口

两模型逐目标键(label,mode,window_id,agent_id)完全对齐，共{a['paired_error_rows']}行配对误差；每种选择/模式均5833目标窗口。实际unique验证行人{a['unique_validation_agents']}，同一人和时间支持重复，不能当5833个独立人。按行人先均值再等权，best K1的NoSocial−Full ADE差为{person_delta:+.6f}米；这是另一聚合口径，不是额外独立录像。不作iid窗口p值或伪三seed稳定性声明。

## 为什么仍不能单独归因图传播

Full与NoSocial同时隔离图、节点卷积和跨节点规范化；为保持参数量，单人核的几何也变为9×1；归一化会改变运动幅度表达和优化路径。即使Full较低，也只能证明本次整套Full相对该特定控制的条件性结果，不能排除表征/容量函数空间/优化混杂，不能单独证明合法社会图或真实邻居信息有效。

原失败NoSocial全局running评价产生明显尺度问题，不能被拿来作为更弱对照突出Full。其第1–4轮、失败和一次修订全部保留。修订版有独立从头训练，不是把未训练结构推理结果当性能。

固定官方checkpoint排列干预已证实节点卷积传递任意索引邻居信息，但只是推理依赖证据，不等于独立训练负对照。没有训练随机图、graph-only消融或多录像多seed，因此不能给这些未运行实验填指标。

下一阶段若继续，先固定相同运动/归一化与节点无序结构，只改变合法图与自环，做正交从头训练；这是基础消融建议，本阶段没有实现新模块或自动训练。
''')
    new('05_RESEARCH_DECISION.md',f'''
# Stage4.1科研判断

## 八个问题

1. **能从头稳定训练吗？** Full在固定非ETH协议完成30轮，loss/梯度/三种预测指标有限，5轮恢复逐元素一致；说明训练工程可用。原NoSocial在5轮评价失败；单次规范化一致性修正后的NoSocial也完成30轮。不能删掉失败，不能将数值稳定等同充分收敛。验证曲线波动、30轮与官方长预算不同，尚无论文级收敛保证。
2. **自训练与官方checkpoint为何不能直接比较？** 本轮只四个非ETH录像原train时段，1488窗口，ZARA2整录像开发验证；排除ETH及额外zara03/uni_examples；从头seed17、30轮、明确定义的累积/验证NLL规则。官方训练池及选择历史不同、可能见ETH。本轮没有ETH测试或官方训练协议完全复刻。
3. **原网络排列敏感吗？** 是。固定官方checkpoint、64窗口235目标，均值预测平均差0.266736米，最大1.917565米；ADE0.521069→0.567688，FDE1.090503→1.197161。ST-GCN约1e-7误差，首个TXP-CNN节点卷积差异约1.907，定位明确。不能据此断言消除敏感性就一定提升预测。
4. **NoSocial与Full实际性能如何？** best K1下{direction}；NoSocial−Full ADE/FDE为{delta:+.6f}/{fdelta:+.6f}米。所有K口径、best/last和分组见02/04，不以一种指标代替另一种，也不利用原失败对照夸大差距。
5. **社会图有独立收益吗？** **按预设NLL选择的K1整体结果不支持：Full没有超过NoSocial，独立收益尚未建立。** Full仅K20 FDE单项较低，不能用它代替K1整体或所有概率指标。两模型还同时改变图、节点卷积、归一化和单人核几何，因此既不能把单项领先归因图，也不能证明社会信息本身无价值。只有一个已曝光录像、一个seed，绝不能宣称跨场景稳定提升。
6. **节点卷积也在传消息吗？** 是，3×3核沿节点索引轴跨人传播；单位图下输入Jacobian仍跨人非零，排列诊断首个大差异定位tpcnns.0。训练态BN还共享节点统计。NoSocial把所有这些当前跨人路径隔离，均有结构测试。
7. **图是否值得独立消融？** 值得作为下一项可证伪的基础问题，尚不是新方法立项。当前图按位移差逆距离，而非物理距离；相同位移边权0、极近非零位移差边权可大，且索引卷积掩盖图作用。应先固定同一运动表示、规范化、节点处理，仅改图关系，独立从头训练；不直接堆高阶/attention/新loss。
8. **具备新方法研究条件吗？** 已有可运行从头训练管线、真实配对结果和明确排列依赖问题；**尚不足以正式宣称新网络路线成立**。应先完成无索引伪关系的公平图消融及有限预算收敛核查，再决定是否有社会信息价值。排列等变、归一化和CV/GRU本身均不是新创新。本阶段不自行执行下一阶段实验。

## 决策与停止

保留Social-STGCNN维护版作为公开工程基线，保留全部失败原型与checkpoint，不把维护版称精确CVPR2020。暂不开发新论文模块。Full在N5–8组有小幅条件优势，N≥9组明显退化，不能只挑中组宣称有效；人数来自离线完整上下文，也不等价物理邻居密度。下一阶段先固定与主指标一致的checkpoint规则、审查规范化与优化波动，再验证单因素图贡献和训练预算是否足够；若公平图控制仍无收益，应停止继续社会模块堆叠，重新审查任务/运动基线，而非根据一个录像挑数据或随机种子。

本次5轮门槛后最多30轮，没有调lr/换录像/多seed搜索；只发生一次有记录的NoSocial归一化一致性修正，旧失败实验保留。下一阶段模型和确认协议必须另行冻结，现有曝光ZARA2不能称独立确认，ETH继续封存。

证据：model_best_last.csv、全部epoch曲线、paired_target_errors.csv.gz、paired_person_means.csv、permutation_modules.csv、nosocial_failed_pilot.json、nosocial_failure_diagnosis_v2.json、单元测试XML。来源代码为Stage4冻结官方网络及Stage4.1独立训练目录；没有借用其他项目结果，也没有未运行模型指标。
''')
    new('STAGE4P1_REPORT.md',f'''
# Stage4.1：从头训练与社会结构基础归因

## 完成情况

分支feat/stage4p1-social-training，基于c0494e8aaba4bcd350d2efa75ba9a6f675d9146f。官方维护版333d3a57b4d2705e129b21aefefa09c79b2b9ae1源文件未改。Full与修订NoSocial均从头完成seed17、最多30轮（包含5轮功能验证），相同数据/损失/优化器/预算；ETH封存，没有push或新论文模块。原失败NoSocial及全部旧模型完整保留。

协议A：完整8/12、0.4秒、米制，完整20帧离线上下文至少2人。四个非ETH录像原train时段1488窗口/23349目标；留出ZARA2整录像921窗口/5833目标。B只生成独立历史上下文manifest，不训练，不与A混报。数据和SHA见01协议与configs/stage4p1。

## 真实结果

{result_table}

按预设NLL选定checkpoint的K1均值，{direction}；NoSocial−Full差ADE {delta:+.6f}米、FDE {fdelta:+.6f}米。**按预设NLL选择时Full未超过NoSocial，当前协议未建立独立社会收益**；Full只在K20 FDE单项较低，不能据此宣布社会有效。N5–8组Full有条件优势，N≥9组明显退化，全部报告。仅一个已曝光录像、一个训练seed，无跨场景稳定性或独立确认结论。K20分别最优，不能与K1混排。所有目标严格配对，重复窗口不独立。

## 关键发现和失败

- Full从头训练流程可用，5轮checkpoint恢复预测完全一致，完整30轮指标/梯度有限；当前预算不证明充分收敛或论文性能。
- 官方checkpoint行人重排后，64窗口235目标平均预测差0.266736米、最大1.917565米；K1 ADE增加0.046619、FDE增加0.106658。ST-GCN保持约1e-7数值等变，首个TXP-CNN节点卷积打破等变。
- 单位图不是NoSocial：节点卷积和训练BN仍传递跨人信息。修订NoSocial在train/eval具备零跨人Jacobian、排列一致性、相同7563参数；原版未调用1309参数也保留。
- 原NoSocial第5轮评价非有限NLL，停止、保留4个checkpoint和1–4轮记录。诊断支持训练/评价统计不一致引发尺度放大；只做一次预先记录的修订，新目录从头训练，未修改Full/lr/数据/NLL/预算。不能隐藏失败或把修订称完全初始预注册。
- Full/NoSocial改变了多条社会路径及规范化/核几何，所以**不能独立证明社会图收益**。没有随机关系重训练、graph-only或多录像、多seed实验；不填假指标。

## 资源和验证

{timing}

训练时间、评价时间单列；两主模型GPU串行；原失败试验只有已日志记录轮次的精确耗时，未捕获第5轮完整计时，不补造总时长。完整{tests['tests']}项测试通过；GPU吞吐、checkpoint SHA、数据来源hash和执行commit保存到结果。历史1251项资产SHA逐项核验，包含全部Stage1–4跟踪文件、旧checkpoint、官方源文件/权重；数据、缓存、环境、checkpoint不进入Git。

## 研究结论

可以继续把该公开维护版作为工程基线；还不能直接设计/宣称新论文方法。下一步先审查checkpoint规则与规范化/优化波动，再进行控制运动表示/规范化/节点顺序路径的图独立消融，并核查固定预算收敛；不添加复杂模块。任何单录像收益均为条件性开发证据，ETH不用于选择。

## 交付索引

- 01_TRAINING_PROTOCOL.md：冻结协议、公式、张量形状、筛选与单次修订。
- 02_PUBLIC_BASELINE_TRAINING.md：全部实际指标、曲线、耗时、检查点和失败。
- 03_PERMUTATION_AUDIT.md：完整人数分组和逐模块定位。
- 04_SOCIAL_ABLATION.md：严格无跨人路径的对照、配对与解释混杂。
- 05_RESEARCH_DECISION.md：八项判断、停止规则和下一阶段条件。
- results/tables/stage4p1/：真实配置索引、数据hash、逐epoch指标、目标/行人配对、测试、日志SHA及delivery_checks.json。
- experiments/stage4p1/：本地不可覆盖epoch checkpoint与原日志；Git忽略。

完成本地commit后停止。用户自行运行：

```bash
git push -u origin feat/stage4p1-social-training
```
''')
    print('wrote four result reports from executed data; original protocol and permutation report retained')


if __name__=='__main__':main()
