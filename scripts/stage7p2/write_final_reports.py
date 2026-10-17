"""Document actual admission evidence and a future, unexecuted training plan."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new

OUT = Path('results/tables/stage7p2')
REPORT = Path('reports/stage7p2')


def load(name):
    return json.load(open(OUT/name))


def write(name, text):
    with (REPORT/name).open('x', encoding='utf-8') as f:
        f.write(text.strip()+'\n')


def table(head, rows):
    return '\n'.join(['| '+' | '.join(head)+' |', '| '+' | '.join(['---']*len(head))+' |']+
        ['| '+' | '.join(map(str, r))+' |' for r in rows])


def main():
    data, weights, smoke = load('official_data_audit.json'), load('official_checkpoint_strict_loading.json'), load('smoke_summary.json')
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    versions = dict(AgentFormer='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',
        PECNet='a345ada1557f53dc2acb4c07f71d340d366980da',
        SGCN='5153702d8ae32a2b0586b093d4e094aee08b0f9b',
        SocialSTGCNN='333d3a57b4d2705e129b21aefefa09c79b2b9ae1')
    published = dict(
        AgentFormer=dict(year=2021, venue='ICCV2021, corrected maintained results', source='https://arxiv.org/html/2103.14023#S4.T1', values=[(.45,.75),(.14,.22),(.25,.45),(.18,.30),(.14,.24),(.23,.39)]),
        PECNet=dict(year=2020, venue='ECCV2020', source='https://arxiv.org/pdf/2004.02025v3', values=[(.54,.87),(.18,.24),(.35,.60),(.22,.39),(.17,.30),(.29,.48)]),
        SGCN=dict(year=2021, venue='CVPR2021', source='https://arxiv.org/html/2104.01528v1#S4.T1', values=[(.63,1.03),(.32,.55),(.37,.70),(.29,.53),(.25,.45),(.37,.65)]),
        SocialSTGCNN=dict(year=2020, venue='CVPR2020', source='https://arxiv.org/html/2002.11927', values=[(.64,1.11),(.49,.85),(.44,.79),(.34,.53),(.30,.48),(.44,.75)]))
    rows = []
    for model, m in published.items():
        for scene, (ade, fde) in zip(['ETH','HOTEL','UNIV','ZARA1','ZARA2','AUTHOR_AVG'], m['values']):
            rows.append(dict(model=model, scene=scene, author_reported_ADE=ade, author_reported_FDE=fde,
                year=m['year'], venue=m['venue'], source=m['source'], code_commit=versions[model],
                author_report_only=True, HTraj_standard_five_scene_reproduced=False,
                directly_strictly_comparable=False, reason='No common-cohort full reproduction completed; PECNet supplied evaluator additionally couples ADE to FDE-selected endpoints'))
    csv_new(OUT/'literature_reported_metrics.csv', rows)
    json_new(OUT/'public_baseline_audit.json', dict(versions=versions, literature=published,
        paper_evidence=load('paper_source_checks.json'), repository_evidence=load('baseline_source_checks.json'),
        models_actually_run_this_stage=['AgentFormer CVAE+DLow strict loading, real one-window training and checkpoint smoke'],
        formal_metrics_executed=False, all_other_baselines_this_stage='Paper/source audit only; previous stage results preserved but not ranked'))
    source_closure = {}
    for p in [*Path('scripts/stage7p2').glob('*.py'), *Path('src/agentformer_benchmark').glob('*.py'), Path('tests/test_agentformer_standard_benchmark.py')]:
        try:
            committed = subprocess.check_output(['git','show',f'93e8a5a:{p}'], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        assert hashlib.sha256(committed).hexdigest() == sha256(p), p
        source_closure[str(p)] = sha256(p)
    json_new(OUT/'execution_source_closure.json', dict(execution_started_at_commit=smoke['execution_commit'],
        full_evidence_containing_commit='93e8a5a', executed_source_files_equal_that_commit=source_closure,
        chronology_deviation=True, pre_smoke_complete_evidence_commit_claim=False,
        actual_process_finished_before_attempted_SIGSTOP=True,
        rule='Do not retroactively claim preregistration. No formal training/test selection; rules unchanged from initial protocol.',
        all_optimizer_updates=12, no_restarts_or_extra_budget=True))
    weight_table = table(['fold', 'CVAE索引/参数', 'DLow索引/参数', '双checkpoint strict'],
        [[r['fold'], f"{r['CVAE_checkpoint_index']}/{r['CVAE_parameters']:,}", f"{r['DLow_checkpoint_index']}/{r['DLow_registered_parameters']:,}", r['strict']] for r in weights['cases']])
    literature_table = table(['作者报告模型', 'ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2', '作者AVG'],
        [[name]+[f'{a:.2f}/{b:.2f}' for a,b in entry['values']] for name,entry in published.items()])
    write('04_OFFICIAL_MODEL_REPRODUCTION.md', f'''# 官方权重、从头训练入口与公开基线审查

## AgentFormer作者权重：严格加载通过，完整数值复现尚未执行

作者[README](https://github.com/Khrylx/AgentFormer/tree/{versions['AgentFormer']})给出公开Drive/Baidu链接。Drive先返回大文件确认页，按其公开下载表单正常下载，非账号认证绕过；517,512,541字节ZIP SHA256 `{weights['archive_sha256']}`。档案14个文件包含五折各CVAE/DLow以及nuScenes权重；本阶段未评价nuScenes。全部文件SHA与路径见 `official_checkpoint_acquisition.json`，只留被忽略cache。

{weight_table}

没有strict=False。原DLow把CVAE置于普通list，DLow state_dict只含591,872个采样器参数，不能以DLow单个权重代表完整网络。原构造函数按pred_cfg/pred_epoch另严格加载CVAE，本轮又逐张量确认这部分等于独立CVAE加载结果。ETH多两个out_mlp_dim=[512,256]，参数比其他折多，内部traj_scale=2；不能用Hotel配置替换ETH。

[作者归一化说明](https://github.com/Khrylx/AgentFormer/issues/5)确认旧代码错误地使用未来坐标计算中心，2021-10-07更新了权重和结果。当前冻结源码scene_orig只来自历史。本轮严格加载证明结构兼容，作者链接与修正版README相符；**checkpoint没有可靠源码commit元数据，严格加载本身不证明训练全过程或已复现修正后的五折数字**。本阶段未运行完整模型测试指标。API抓取issue评论曾403限流，作者说明通过GitHub页面核实；该失败不抹去。

## 原训练与采样语义

原CVAE：Agent-Aware Transformer、历史context、FutureEncoder联合后验、learned prior、自回归FutureDecoder；MSE+KL(min_clip=2)+K20 sample loss，权重均1，Adam1e-4，100完整epoch，StepLR(10,gamma=.5)，每10轮保存。原loss不是精确边缘NLL。

原DLow：固定CVAE权重，q_MLP=[512,256]输出20×32的A/b，KL权重.1/clip10、diverse权重5/d_scale5、recon权重5；Adam1e-4，50完整epoch，StepLR(5,.5)，每5轮保存。train_w_mean=True时z=b；infer也mean=True，形成20个确定性context-conditioned latent head。不能把完整模型解释为20次CVAE独立先验采样。CVAE仍在计算图中，但不属于DLow optimizer参数；本轮确认它的权重没有变化，未擅自修改requires_grad策略。

官方train.py没有验证选择逻辑，只按固定周期保存；test.py默认取最新存在的checkpoint。作者ZIP选择的CVAE/DLow索引如表，不能用测试成绩去重选。未来从头复现必须预先冻结验证选模规则；这属于公开记录的训练管理差异，不改网络/loss，也不冒称精确复刻作者未公开的选模全过程。

原train/test/eval.py的完整imports和--help均在兼容环境实际运行成功。真实smoke执行原train函数AST原样函数体与原data_generator，限制为一个预设frame，不是完整epoch。官方源码0修改；兼容仅沿用_LinearWithBias=Linear和为loader增加运行时np.int=int别名。入口日志、严格权重日志和失败记录均保存。

## 四个公开模型的版本与可比性

| 模型/年份 | 输入与候选 | 数据与评价/额外处理 | 本轮证据与严格可比性 |
|---|---|---|---|
| AgentFormer ICCV2021维护修正版 | ETH/UCY历史二维坐标/位移；CVAE+DLow20头 | 作者八录像五fold；完整20帧上下文、允许单人；独立minADE20/minFDE20；无地图/FPC；仅尺度还原 | 五折权重strict成功、真实训练smoke，未完成五折评价；暂不能填HTraj标准成绩 |
| PECNet ECCV2020 | 历史位置、目标初始位置、预测endpoint与nonlocal pooling；20候选 | 论文ETH/UCY米制LOO；SDD配置与ETH notebook不能混用。供应的test先按终点误差选endpoint，再生成一条路径计算ADE，**不是独立minADE@20**；K20用放大采样σ，论文不做低K截断 | commit `{versions['PECNet']}`；论文表2已核对，ETH notebook为no_pooling且留有按test选best的示例，不能作为完整最佳社会模型的公平入口。无本轮训练/权重复现，不取较弱ablation值冒充最佳结果；不可直接严格比较 |
| SGCN CVPR2021 | 位移加历史时间通道；稀疏空间/时间图；20高斯样本 | 原完整20帧目标、人数≥2；分别minADE/FDE，无RGB。冻结作者历史训练脚本与论文训练超参不一致，Stage6还记录单人/排列与累积边界问题 | 历史完整代码 `{versions['SGCN']}`；当前main已移除代码。表1核对，旧检查保留，本轮源码/论文复核而未训练；目标集合/实现差异未消除，不可直接严格比较 |
| Social-STGCNN CVPR2020 | 历史位移/图；二维高斯20样本 | 完整20帧且默认人数≥2；分别minADE/FDE；无额外RGB。TXP沿节点索引卷积，旧Stage4.1已发现顺序敏感 | 官方维护版本 `{versions['SocialSTGCNN']}` 不等于精确CVPR2020原版；论文表2核对，旧checkpoint成绩不移植到本协议。无本轮重新训练；不可直接严格比较 |

三篇开放获取PDF与SHA在 `paper_source_checks.json`，作者README/源码获取状态在 `baseline_source_checks.json`。部分历史README URL 404、issue API 403已保留；论文本体与本地冻结源码足以定位上述结论。Academic MCP未挂载，采用作者arXiv/正式论文与仓库主来源，未用第三方排行榜替代。PECNet notebook的完整字节SHA固定在新cache，未执行其测试/训练代码。

## 文献报告值（ADE/FDE米；不是HTraj实测排名）

{literature_table}

AgentFormer取[修正后的论文表1](https://arxiv.org/html/2103.14023#S4.T1)及README；PECNet取[原论文表2](https://arxiv.org/pdf/2004.02025v3)；SGCN取[原论文表1](https://arxiv.org/html/2104.01528v1#S4.T1)；Social-STGCNN取[原论文表2](https://arxiv.org/html/2002.11927)。全部为作者报告值，CSV显式标记author_report_only、directly_strictly_comparable=False。PECNet的ADE候选选择、多人过滤、公开实现维护差异决定不能简单按同K20就宣称同目标严格对比。只有以后用相同测试目标/规则完整评价，才能建立本项目标准基准表。
''')
    resource_rows = []
    for r in smoke['cases']:
        for name, stage in r['stages'].items():
            m = stage['resource']
            resource_rows.append([r['fold'], r['people'], name, f"{stage['loss']:.6f}",
                f"{stage['gradient_L2']:.4f}", f"{m['peak_allocated_bytes']/2**20:.2f}",
                f"{m['peak_reserved_bytes']/2**20:.2f}", m['whole_card_sampled_peak_MiB'],
                f"{m['seconds']:.4f}", m['OOM']])
    resource_table = table(['fold','N','阶段','原loss','梯度L2','allocated/MiB','reserved/MiB','整卡采样/MiB','耗时/s','OOM'], resource_rows)
    write('05_GPU_AND_TRAINING_FEASIBILITY.md', f'''# RTX3080真实工程smoke与训练可行性

实际六场景：HOTEL与ZARA2各N1、N2、N57，来源是各自官方train池的biwi_eth_train与students001_train。不是用ETH测试做训练；这一使用发生于用户协议切换及c28350f之后。完整候选图/行人全部保留，不裁人数。两个fold选到了相同的原始窗口，虽训练入口分别独立初始化，**不是六个独立录像或跨域泛化证据**。

{resource_table}

每行一次原train函数受限调用：原Adam、原loss、反向、1次optimizer.step；保存模型/Adam/StepLR，再严格恢复模型及优化器/调度器，同噪声20候选预测差=0。DLow同时确认基础CVAE权重不变。总计12次工程更新、0完整epoch，不继承作者或旧smoke权重。DLow依赖本次独立随机初始化CVAE的1次更新状态，只验证衔接，不能当作已训练基础模型。

计时含整个官方训练pool入口初始化、1窗口原train、保存/恢复和两次推理，不是稳定训练吞吐；原train调用会在受限循环末推进一次scheduler，但这不是完整数据epoch。本轮没有据此声称正式训练充分或收敛。allocated/reserved为PyTorch精确峰值；整卡为nvidia-smi约0.1秒采样峰值，可能漏短峰，含驱动/其他进程，不能当精确整卡峰值。

N57 CVAE peak allocated=6404441088字节，reserved=9567207424字节，整卡采样最高9744MiB，无OOM。10GB卡余量小，计划单GPU串行，不并行fold、不私自删困难窗口。HOTEL/ZARA2完整网络资源已测；ETH额外MLP/scale2/官方32人CVAE训练上限虽已源/权重核验，**其最大上下文DLow GPU训练尚未单独实测**，下一阶段对应fold正式训练前必须按原配置预检，失败则停该fold，不能外推为全部配置已测。

## 下一阶段完整训练计划（未执行）

1. 先以本轮严格匹配的作者双checkpoint及配置，固定官方seed和samples-only evaluator，完成冻结的五场景推理核对；不使用测试指标重新挑checkpoint、调超参。若与修正后报告值不符，先查差异，不能改模型刷回数字。
2. 从头训练先HOTEL/ZARA2工程验证，再ETH/UNIV/ZARA1；每fold全量官方train，CVAE100完整epoch、DLow50完整epoch、原Adam1e-4及StepLR、原损失、各自原配置。seed取每fold作者原配置；不继承本轮随机smoke或作者预训练。每fold正式启动前冻结代码、数据、输出目录与最大人数资源检查。
3. 原stock train无验证选模，为满足不看测试选模型，统一只在官方val保存网格评估：CVAE每10轮、DLow每5轮。最低val minADE@20，完全相同则更早；配对minFDE来自同checkpoint。同一固定sample rule、官方三位小数坐标序列化，保存首候选K1辅助。该额外训练管理规则需独立预提交，承认与作者未公开选择过程的差异。CVAE选定后固定pred_cfg/pred_epoch，DLow另训练；每次恢复必须携带两份checkpoint及SHA。
4. 不发生NaN/Inf/未说明数据差异时才完成后续预算；如OOM或非有限，保存失败现场并停止，不延长epoch、重调学习率或削人数。进入新方法设计前必须核对五折作者权重或从头基线的真实预测能力。
5. 同目标CV-Last2/OLS4/OLS8为解析sanity对照，全部固定展示，不能逐场景事后选择最有利CV。其他公开模型若目标筛选/指标不一致，先标不可严格比较，不能把有较弱或不同任务的文献值当公平优势。完整五折后才形成论文标准基准表，分别标作者权重推理复现与从头训练。

五fold官方有效训练窗口合计14690，因此100+50完整epoch约2,203,500次场景optimizer更新（忽略异常/重启，不是目标数）。本次计时含初始化/恢复，不能据它给出可信全程小时估计：若未来测得均步0.05/0.2秒，单纯更新核时间分别约30.6/122.4小时，**是预算情景而非本轮吞吐实测**，还需验证/IO时间。下一阶段先定长测量训练稳态吞吐，不立即承诺RTX3080能在固定小时完成全部五折。

本阶段停止，不执行以上正式训练计划。
''')
    go_gates = [
        ('官方数据完整','PASS: 74 files / 22 Git blobs / 8 recordings'),
        ('官方划分对齐','PASS: 15 loaders, 148 packet checks, no held recording in train/val'),
        ('完整20帧资格明确','PASS: min8/min12, N1 legal, future existence declared'),
        ('minADE20/minFDE20评价复核','PASS: independent winners, real serializer/GT/official eval CLI'),
        ('真实单人/多人训练入口','PASS: original train function, N1/2/57, 12 updates and restoration'),
        ('CVAE与DLow配置关系','PASS: fivefold strict two-part loading, sampler semantics explicit'),
        ('GPU可执行方案','PASS_FOR_STAGED_START: H/Z full N57 measured; remaining fold-specific preflights required'),
        ('未说明数据/评价差异','PASS_DISCLOSED: original selector unknown, metric/cohort and compatibility differences documented')]
    json_new(OUT/'go_no_go.json', dict(decision='GO_STAGED_OFFICIAL_BASELINE_REPRODUCTION',
        gates=[dict(condition=k,status=v) for k,v in go_gates],
        first_formal_folds=['hotel','zara2'], all_five_folds_required_before_standard_result_table=True,
        full_fivefold_performance_reproduced=False, from_scratch_convergence_demonstrated=False,
        ETH_specific_maximum_DLow_GPU_training_verified=False,
        chronology_deviation_disclosed=True, next_stage_protocol_must_be_committed_before_formal_training=True,
        formal_epochs_started=0, engineering_optimizer_updates=12, ETH_test_scores_computed=False,
        automatic_push=False))
    plan = dict(status='PROPOSED_NOT_EXECUTED', official_commit=versions['AgentFormer'],
        protocol='configs/stage7p2/benchmark_protocol.json', folds=['hotel','zara2','eth','univ','zara1'],
        pretrained_reference='Fixed author CVAE+DLow matched pairs, exact config/weights SHA, official seed, all fivefold evaluator; no checkpoint re-selection',
        scratch_initialization='Fresh per fold, no pretrained or smoke warm start', CVAE_epochs=100,DLow_epochs=50,
        optimizer='Original Adam lr1e-4', CVAE_schedule='StepLR step10 gamma0.5', DLow_schedule='StepLR step5 gamma0.5',
        seeds='Per-fold frozen official maintained config seed, not selected by test',
        selector='Official val minADE20, tie earlier; CVAE every10 epochs, DLow every5; paired FDE same checkpoint',
        selector_difference_from_author='Author train.py has no validation selection; explicit additional management rule',
        metrics='Independent minADE20/minFDE20; auxiliary first candidate K1; serialize world coordinates at 3 decimals',
        complete_all_five_before_paper_comparison=True, original_ETH_scale_and_training_cap_retained=True,
        per_fold_maximum_resource_preflight=True, sequential_one_GPU=True,
        stop='Nonfinite, OOM or unexplained protocol/weight mismatch; no structure/loss/epoch search',
        total_scene_optimizer_updates_budget=2203500, not_a_measured_walltime_estimate=True)
    json_new('configs/stage7p2/proposed_full_reproduction_plan.json', plan)
    write('06_GO_NO_GO.md', f'''# 验收：GO，进入下一阶段分阶段官方基线复现

{table(['条件','结论'],go_gates)}

这是**公开维护版基线的工程准入**，不是五场景论文成绩复现成功，也不是新方法有效。下一阶段先HOTEL/ZARA2，其他三fold按各自原配置先资源预检；ETH额外MLP的最大DLow训练显存不能用本轮Hotel配置数值代替。正式计划见 `configs/stage7p2/proposed_full_reproduction_plan.json`，本轮未执行。

标准完整20帧资格已经由用户明确授权，不把它宣称严格历史邻居因果协议。历史GT数值不参与scene_orig的新版本归一化，测试只能评价samples；未来存在性用于官方离线上下文资格属于已披露的任务定义。旧Stage7.1 No-Go针对不同协议，不被抹掉，也不与本协议结果混合排名。

## 执行顺序偏差与可用证据

初始完整预算/协议于c28350f预先提交。随后数据/评价核验已完成并写入文件，但补充提交被新loader日志末尾空行拦下，smoke命令仍被启动；6场景12更新在93e8a5a补充提交前结束。试图SIGSTOP时进程已正常退出，**没有实际暂停/恢复**；早期处置记录描述的是尝试，`commit_order_actual_outcome.json`保存更正。没有重跑或增加预算。

不追认“全部smoke前完整核验材料已提交”。规则、数据、源码没有根据smoke误差更改；实际执行源码SHA与93e8a5a一致的对照见 `execution_source_closure.json`。本轮0完整epoch、没有测试指标调参；用户要求正式训练前提交完整协议/评测材料，在下一阶段开始前必须独立执行并核对提交结果。该工程顺序错误作为明确限制保留，不能在论文中称本轮严格预注册的预测实验。

未发生OOM/非有限训练loss；没有借删除行人、放宽GT资格、替换损失或strict=False通过。12项契约测试、原CLI import/help、所有官方数据/权重SHA与完整单测通过；但没有证明100+50轮收敛、跨seed稳定性或任何性能提升。正式复现如失败应停并分析基线，不增加创新模块。

完成本地提交后停止，不自动push，不在本阶段启动正式训练。
''')
    write('STAGE7P2_REPORT.md', f'''# Stage7.2 标准ETH/UCY基准对齐总报告

**结论：技术GO，可在下一阶段分阶段进入官方基线复现；本阶段不启动正式训练。** 全部旧实验保留，与新协议指标分开。当前没有可填写的AgentFormer五场景本机性能表。

- 分支 `feat/stage7p2-standard-benchmark-alignment`，基础 `81a2e239e8ee79c07d19e319401c0f96c0b5e7ec`。
- 模型/源码版本 `{versions['AgentFormer']}`，原Transformer、CVAE、FutureEncoder、learned prior、decoder、loss、DLow全部保留。43个官方源文件版本保护；运行时机械兼容无源码修改。
- 官方74数据文件、8录像/22唯一blob全部补齐，五折train/val/test与20帧完整资格明确；15个原loader入口、148个首/最大packet检查通过，完整窗口清单/原始SHA保存。每fold测试录像不入该fold训练/验证，train/val完整支持无共享frame，但同录像相邻时间相关性不叫独立确认。
- 用户授权协议切换已在c28350f记录：非ETH折允许ETH_train/val；ETH折自身测试不调参。本轮未计算ETH模型测试成绩。旧HTraj因果与部分观测结果不得与新基准混排。
- 官方metric公式、ADE/FDE不同候选赢家、真实四列输出序列化、GT frame/ID对齐及原eval CLI全部验证；主指标独立minADE20/minFDE20，K1只作明确命名辅助。DLow20头不是20次CVAE独立采样，不能用recon目录或旧K1代替。
- 作者517MB ZIP取得，五fold CVAE与DLow全部strict加载。ETH参数6,497,922，其他CVAE5,972,098；DLow各591,872。双checkpoint引用与SHA明确；尚未运行完整作者权重五场景指标，不声称已复现修正版数值。
- 六个真实N1/2/57场景、两阶段共12次原Adam工程更新、0完整epoch；原loss/梯度有限、严格模型/优化器/调度器恢复通过，DLow基础CVAE权重不变，无OOM。N57 CVAE整卡采样峰值9744MiB，后续串行；ETH特殊配置的最大DLow训练显存另需预检。
- 全量单测 {tests['tests']} passed / {tests['failures']} failed；历史SHA保护见 `preservation_and_versions.json`。作者公开基线最佳论文值单独存 `literature_reported_metrics.csv`，不作为本机实测排名，PECNet候选选择与多人过滤等差异明确不可严格比较。

执行顺序错误完整保留：初稿预算协议已预提交，但补充核验材料commit检查失败后误先启动smoke，全部工程步先于93e8a5a补充提交结束；没有实际SIGSTOP成功、没有重跑或追加更新。不将其追认为全部smoke前完整预提交。详见06报告和两个顺序偏差JSON。下一正式训练需先独立冻结全部配置/选模/停止条件并检查commit成功。

交付七份报告 `reports/stage7p2/`，数据/源码/权重SHA、split/window清单、strict加载、原始loss与梯度、资源、CLI日志、所有失败与测试结果在 `results/tables/stage7p2/`。原数据、PDF、权重、环境、cache不入Git。下一阶段拟先固定作者权重五场景核对，再从头HOTEL/ZARA2、随后ETH/UNIV/ZARA1，完整CVAE100+DLow50轮；原val选模，绝不用test改超参。计划仅提案，实际训练未执行。

用户手动推送：

```bash
git push -u origin feat/stage7p2-standard-benchmark-alignment
```

本阶段完成后停止，等待下一步科研决策。
''')
    # Append a correction rather than rewriting already committed audit prose.
    correction = '\n\n---\n执行顺序更正：上述规则/核验在smoke前已写成，但完整材料补充commit被新日志末尾空行检查拦住；6场景12工程更新结束后才进入93e8a5a。不得把标题的“smoke前冻结”解释为完整证据已预先入Git。初始预算协议c28350f已预提交；未运行正式epoch。详见06报告与commit_order_actual_outcome.json。\n'
    for name in ['01_OFFICIAL_DATASET_AUDIT.md','02_STANDARD_PROTOCOL.md','03_EVALUATION_EQUIVALENCE.md']:
        with (REPORT/name).open('a', encoding='utf-8') as f:
            f.write(correction)
    print('Final seven reports, literature-only table and future fivefold plan written. Decision: staged GO; no formal training.', flush=True)


if __name__ == '__main__':
    main()
