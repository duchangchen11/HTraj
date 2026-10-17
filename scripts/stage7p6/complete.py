"""Complete source-grounded Stage7.6 tables and reports; no model execution."""
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new

OUT = ROOT/'results/tables/stage7p6'
REPORT = ROOT/'reports/stage7p6'
OFFICIAL = 'e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5'
AF = 'https://github.com/Khrylx/AgentFormer/blob/'+OFFICIAL+'/'
LIT = ROOT/'experiments/stage7p6/cache/literature'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT/path).read_text())


def report(name, content):
    with (REPORT/name).open('x') as stream:
        stream.write(content.strip()+'\n')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    with gzip.open(ROOT/'results/tables/stage7p1/context_windows.csv.gz','rt') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        if int(row['targets'])>0:
            groups[row['fold'],row['split']].append(row)
    target_summary=[]
    for (fold,split), group in sorted(groups.items()):
        target_summary.append(dict(fold=fold,split=split,supervised_windows=len(group),
            context_node_windows=sum(int(r['people']) for r in group),
            supervised_target_windows=sum(int(r['targets']) for r in group),
            incomplete_future_node_windows=sum(int(r['incomplete_future_nodes']) for r in group),
            mixed_label_windows=sum(int(r['incomplete_future_nodes'])>0 for r in group),
            no_future_node_windows=sum(int(r['no_future_nodes']) for r in group),
            partial_history_excluded=sum(int(r['partial_history_excluded']) for r in group),
            max_context=max(int(r['people']) for r in group)))
    csv_new(OUT/'supervised_cohort_summary.csv',target_summary)
    # Data hashes: copy identity metadata only; do not inspect test trajectories/errors.
    official_data=load('results/tables/stage7p2/official_data_audit.json')
    csv_new(OUT/'official_data_identity.csv',[
        dict(path=r['path'],sha256=r['sha256'],git_blob=r['git_blob'])
        for r in official_data['official_files']])
    sources=[]
    def source(name,year,venue,url,code,commit,review,missing_future,history_only,posterior,dynamic,eth,verdict):
        sources.append(dict(name=name,year=year,venue=venue,paper_url=url,code_url=code,code_commit=commit,
            review_scope=review,incomplete_future_context=missing_future,history_only_context=history_only,
            posterior_treatment=posterior,dynamic_population=dynamic,standard_ETH_UCY_targets=eth,
            relation_to_proposal=verdict))
    source('AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting',2021,'ICCV',
        'https://arxiv.org/abs/2103.14023','https://github.com/Khrylx/AgentFormer',OFFICIAL,
        'Stage7.1/7.2 source and this static reread; original paper reused',
        'No: frozen complete20 loader excludes them','Prior input uses history after selection; selection uses future existence',
        '12C future tokens, allC KL; loss mask is not posterior mask',
        'Variable N across scenes; not a birth/death forecast','Official original fivefold yes',
        'Specific implementation incompatibility demonstrated; not a general CVAE impossibility')
    source('Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data',2020,'ECCV',
        'https://arxiv.org/abs/2001.03093','https://github.com/StanfordASL/Trajectron-plus-plus',
        '1031c7bd1a444273af378c1ec1dcca907ba59830',
        'Fixed preprocessing.py/mgcvae.py/scene.py/scene_graph.py; foundational precedent outside2022-2026',
        'Yes in target-centric representation: neighbors are not full-future supervision targets',
        'Neighbor feature slice is historical; precomputed graph t_fut can use future adjacency, so not end-to-end causal certified',
        'Target own future encoder; neighbors contribute to historical x, not required as future tokens',
        'Dynamic scene graph, variable neighbors; no assertion of full unseen-agent generation',
        'ETH/UCY supported; exact evaluator/objects not revalidated this stage',
        'C/S separation is prior art; not an identical joint AgentFormer posterior')
    source('SocialVAE: Human Trajectory Prediction using Timewise Latents',2022,'ECCV',
        'https://arxiv.org/abs/2203.08207','https://github.com/xupei0610/SocialVAE',
        'eb23cbae562bac2c2dcf0d81529e6ab2ac280962',
        'Full open paper and data.py/social_vae.py at frozen commit',
        'Yes: target complete20, other actors may lack observations/future; sentinel and distance validity',
        'Inference can use historical features; raw official union-of20 eligibility not certified causal here',
        'Training backward RNN uses AVAILABLE neighbor future features too; missing future contributions masked by validity/radius',
        'Per-time neighbor mask allows appearing/disappearing context; no novel-agent birth distribution claimed',
        'ETH/UCY8/12 supported; FPC/evaluator differ from AgentFormer',
        'Direct precedent for incomplete-neighbor context; not proof original AgentFormer rectangular posterior already solved')
    source('Posterior Matching for Arbitrary Conditioning',2022,'NeurIPS',
        'https://arxiv.org/abs/2201.12414','https://github.com/lupalab/posterior-matching',
        '08a355ddd22d0cfb14287e02bf30b439c21a0058',
        'Full paper Sec3/4 and AppendixE; models/vae.py',
        'Not a pedestrian-future task; arbitrary conditioning supported',
        'Observed-feature masks, not pedestrian context rules',
        'Partial posterior learned by matching full posterior samples; base training still uses complete x',
        'Not studied','No',
        'General masked/partial posterior is established; does not prove naturally absent future labels handled under MNAR')
    source('Uncovering the Missing Pattern: Unified Framework Towards Trajectory Imputation and Prediction',2023,'CVPR',
        'https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Uncovering_the_Missing_Pattern_Unified_Framework_Towards_Trajectory_Imputation_and_CVPR_2023_paper.html',
        'https://github.com/colorfulfuture/GC-VRNN','b0bcad969616a575567dee90a88daba59ddcbcd7',
        'Full paper Sec3/4/5; repository contains README and supplement, not released implementation',
        'Not established for naturally absent future labels: Eq1 future y and Eq19 all-agent prediction likelihood',
        'Handles incomplete past; does not certify causal C/S rule',
        'Conditional VRNN prior/posterior, imputation+prediction sequential ELBO, temporal decay',
        'Simulates camera visibility/entry/exit over fixed agents; not automatically variable future identity set',
        'Basketball-TIP/Football-TIP/Vehicle-TIP; not standard ETH main evaluation',
        'Related missing observation work; not identical missing future supervision')
    source('MS-TIP: Imputation Aware Pedestrian Trajectory Prediction',2024,'ICML/PMLR235',
        'https://proceedings.mlr.press/v235/chib24a.html','https://github.com/Pranav-chib/MS-TIP',
        '888a95f828ef516a803e4a6e00ad45b219c9b938',
        'Full paper Sec3.1/3.2/3.6/4.1; frozen README/train.py',
        'Not demonstrated: Eq22/23 use future control-point GT; not inferred from history-imputation title',
        'Artificial and native missing-history masks; history context qualification not fully code-audited',
        'Not an AgentFormer CVAE; imputation plus GMM/control-point/refinement objectives',
        'Missing past coordinates, not explicit unseen future agent generation',
        'ETH/UCY8/12 andSDD,20 samples; exact cohort equivalence not demonstrated',
        'Partial-history problem is well studied; future-unlabeled joint posterior remains a different question')
    source('TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports',2024,'ACCV',
        'https://arxiv.org/abs/2410.17785','Unverified author code location','not_frozen',
        'Full v2 paper Sec4.1-4.3; supplementSec4 andSec6',
        'Yes at masking/missing-training level: unavailable goalkeeper observations ignored; NOT a CVAE proof',
        'Arbitrary conditioning mask supports forecast; other tasks intentionally allow future conditioning',
        'Deterministic Set Attention; observation/prediction mask plus distinct NaN-mask, no CVAE KL',
        'Variable length/N padding supported; sports roster and full birth/death are different',
        'SupplementSec6 ETH/UCY8/12/2.5Hz deterministic comparison; not AgentFormerK20 matched cohort',
        'Direct precedent for separate observation/label-validity masks; general mask engineering cannot be claimed new')
    source('Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling',2025,'CVPR',
        'https://openaccess.thecvf.com/content/CVPR2025/html/Capellera_Unified_Uncertainty-Aware_Diffusion_for_Multi-Agent_Trajectory_Modeling_CVPR_2025_paper.html',
        'Unverified author implementation location','not_frozen',
        'Full paper Sec4/5, not code executed',
        'Arbitrary completion conditioning; naturally absent supervised labels not established by conditioning mask alone',
        'History-only is a possible mask; task also permits other temporal conditioning',
        'Diffusion variational objective, not AgentFormer learned CVAE posterior',
        'Variable agent/time attention axes; no assertion of online birth/death handling',
        'Four sports datasets, no standard ETH/UCY table in inspected main protocol',
        'Related arbitrary-mask generative modeling, not an authorized new diffusion route')
    source('SocialTrans: Transformer based social intentions interaction for pedestrian trajectory prediction',2025,'PhysicaA663:130435',
        'https://doi.org/10.1016/j.physa.2025.130435','https://github.com/XiaodZhao/SocialTrans','not_obtained',
        'Publisher Introduction/Proposed approach/Conclusion indexed text; full equations unavailable',
        'Unknown','Unknown: claims variable number of neighbors, not proof of missing future labels',
        'Unknown masks, objective, posterior details',
        'State tensors adjusted to varying pedestrian counts; within-window future validity not established',
        'ETH/UCY+SDD claimed; exact K/cohort not verified',
        'SIE/NPIE parallel state-intention processing, trajectory optimizer/clustering. Missing-label novelty unresolved')
    source('Inter-GraphFormer: A spatial-temporal graph transformer-based method for pedestrian trajectory prediction',2026,'PhysicaA694:131566',
        'https://doi.org/10.1016/j.physa.2026.131566','Author source not located','not_obtained',
        'Publisher detailed Introduction/contributions and CrossRef metadata; full equations unavailable',
        'Unknown','Unknown','CVAE reported; posterior token validity, KL and reconstruction denominators unknown',
        'Unknown; directed sparse interaction graph is not a dynamic population guarantee',
        'ETH/UCY+SDD claimed; exact cohort/K not verified',
        'Directed weighted+sparse graph, parallel spatial/temporal Transformer, dualGLU, graphTransformer decoder; no verified C/S mechanism')
    csv_new(OUT/'prior_work_matrix.csv',sources)
    json_new(OUT/'literature_sources.json',dict(search_date='2026-10-17',
        skill='nature-academic-search multi-source workflow; OA primary PDFs only',
        scope='Focused reproducible search, not exhaustive systematic review',
        queries=['partial observation trajectory prediction','missing trajectory supervision','dynamic agent set forecasting',
                 'causal pedestrian trajectory forecasting','masked conditional variational autoencoder','variable pedestrian population forecasting'],
        queries_coverage='Author/arXiv/CVF/PMLR/publisher/GitHub; older foundational precedents retained',
        rows=sources,pdf_manifest=json.loads((LIT/'pdf_fetch.json').read_text()),
        repository_manifest=json.loads((LIT/'repositories.json').read_text()),
        additional_retrievals=json.loads((LIT/'additional_retrievals.json').read_text()),
        successful_extra_repositories=[dict(repo='lupalab/posterior-matching',commit='08a355ddd22d0cfb14287e02bf30b439c21a0058'),
                                     dict(repo='colorfulfuture/GC-VRNN',commit='b0bcad969616a575567dee90a88daba59ddcbcd7')],
        misleading_repository_redirect=dict(repo='colorfulfuture/UniTraj-pytorch',commit='bfebe75c863fddff535ac4e381246d997f554db1',
            actual_readme_title='Sports-Traj, ICLR2025; not a verified GMC-VRNN implementation'),
        limitations=['OpenAlex fallback HTTP429; no academic MCP available',
            'SocialTrans and Inter-GraphFormer publisher fetch403; prior OpenReview browser challenge not bypassed',
            'No full-method novelty certification for these two papers',
            'Guessed TranSPORTmer/U2Diff repository URLs unavailable; not evidence no public source exists',
            'GMC-VRNN TPAMI2025 early online /2026issue metadata found, full code/method unresolved; not used to claim exact solution'],
        no_training_or_downloaded_source_execution=True))
    anchors=[]
    def anchor(path,start,end,claim):
        p=ROOT/path
        lines=p.read_text().splitlines()
        anchors.append(dict(path=path,file_SHA256=digest(p),start_line=start,end_line=end,
            selected_lines_SHA256=hashlib.sha256('\n'.join(lines[start-1:end]).encode()).hexdigest(),claim=claim))
    prefix='experiments/stage7/cache/AgentFormer/'+OFFICIAL+'/'
    for p,a,b,c in [
        ('data/preprocessor.py',80,88,'get_valid_id requires past AND future existence'),
        ('model/agentformer.py',26,35,'generate_mask assumes same agent_num'),
        ('model/agentformer.py',103,168,'ContextEncoder historical scene_norm/velocity'),
        ('model/agentformer.py',171,257,'FutureEncoder 12C tokens; fut_mask not used'),
        ('model/agentformer.py',395,427,'Learned prior history only, posterior latent only in training'),
        ('model/agentformer.py',522,526,'scene origin depends selected historical population'),
        ('model/agentformer_loss.py',1,46,'Reconstruction mask is not KL or posterior mask; normalize by C'),
        ('model/dlow.py',11,50,'DLow KL/diversity/reconstruction have different GT dependencies'),
        ('model/dlow.py',97,130,'DLow mean=True deterministic latent heads from historical context')]:
        anchor(prefix+p,a,b,c)
    social='experiments/stage5/cache/socialvae/eb23cbae562bac2c2dcf0d81529e6ab2ac280962/'
    anchor(social+'data.py',195,227,'Complete target + potentially incomplete neighbor positions')
    anchor(social+'social_vae.py',170,225,'Posterior uses available neighbor future; inference does not take y')
    tr='experiments/stage7p6/cache/literature/Trajectron-plus-plus/'
    anchor(tr+'trajectron/model/dataset/preprocessing.py',88,150,'Target future separately; neighbor feature history slice')
    anchor(tr+'trajectron/model/dataset/preprocessing.py',212,240,'Target min_future not a direct full-future-neighbor filter')
    anchor(tr+'trajectron/model/mgcvae.py',658,680,'Target future encoder for posterior')
    anchor(tr+'trajectron/environment/scene.py',51,86,'Precomputed graph passes positive t_fut addition-filter length')
    anchor(tr+'trajectron/environment/scene_graph.py',204,223,'Graph slices can aggregate adjacency after current t; no blanket causal certification')
    anchor('experiments/stage7p6/cache/literature/posterior-matching/posterior_matching/models/vae.py',116,150,
           'Full x VAE training and separately masked partial posterior; not naturally missing-label proof')
    json_new(OUT/'static_source_anchors.json',dict(code_executed=False,anchors=anchors))
    gates=[
        dict(condition='真实问题得到现有数据支持',status='PASS',evidence='Stage7.1 mixed-label cohorts and posterior mask-only counterexamples; prevalence scope limited'),
        dict(condition='与近年公开方法存在明确区别',status='NOT_MET',evidence='C/S separation and masked supervision established; no distinct new inference mechanism established; two requested full texts unresolved'),
        dict(condition='数学训练目标自洽',status='PASS_CONDITIONAL',evidence='Observed-label conditional ELBO possible under stated likelihood/missingness assumptions; original loss is augmented, not exactELBO'),
        dict(condition='不依赖虚构未来GT',status='PASS_DESIGN_ONLY',evidence='Omit U future tokens; U latents from history prior, no invented labels; not implemented'),
        dict(condition='公平基线和消融',status='DESIGN_POSSIBLE_NOT_RUN',evidence='Separate complete20 main and C8 auxiliary tracks; retrain all protocol-matched baselines. No comparable results yet'),
        dict(condition='资源预算具备可行性',status='NOT_ESTABLISHED',evidence='N73 inference only; original N57 training already9677MiB whole-card; no mixed-label training memory measurement'),
        dict(condition='解释与官方基准差异',status='PASS',evidence='Unchanged target/K/units insufficient if context eligibility differs; separate rankings mandatory')]
    json_new(OUT/'decision_gates.json',dict(decision='NO-GO',gates=gates,terminate_specific_AgentFormer_innovation_route=True,
        new_model_started=False,new_training_started=False,next_stage_automatic=False,
        alternative='User decision: evidence-led protocol/robustness thesis study, or redefine a new task/dataset first; neither is claimed a ready novel algorithm'))
    table='|fold/split|有监督窗口|C节点窗口|S目标窗口|缺未来节点窗口|混合标签窗口|\n|---|---:|---:|---:|---:|---:|\n'
    for r in target_summary:
        table+=f"|{r['fold']}/{r['split']}|{r['supervised_windows']}|{r['context_node_windows']}|{r['supervised_target_windows']}|{r['incomplete_future_node_windows']}|{r['mixed_label_windows']}|\n"
    report('01_PROBLEM_DEFINITION.md',f'''
# 问题定义与历史证据复核

结论：问题真实存在于已检查的动态轨迹数据，但“给所有历史邻居补零未来再mask损失”不是合法修复。本阶段不更改Stage7.1 NO-GO，也不否定Stage7.3官方完整20帧协议的权重复现。

固定源码`{OFFICIAL}`；本轮研究脚本不运行网络、不训练、不重算任何测试指标；完整回归测试包含既有人工前向/后向检查。Stage7.2的标准协议取代旧ETH封存规则；本轮仍不访问ETH评价误差。Stage7.3测试集已经被观察，不能称为未来未接触确认集。

## 三种不同问题

|问题|定义|现有证据|不能推出的结论|
|---|---|---|---|
|未来GT数值泄漏|固定历史/资格，推理结果取决于未来坐标|Stage7.1严格历史适配器10项原始数据反事实，K1与固定噪声K20差值全部0|不是证明官方get_valid_id仅用历史，也不是证明所有预处理都严格因果|
|未来存在性资格依赖|未来记录是否存在改变入模节点集合|官方get_valid_id同时检查8帧历史与12帧未来；改变集合还改变scene_orig及attention上下文|完整20帧离线benchmark已声明该条件，不能将其偷偷等同在线历史场景|
|训练后验缺标签|历史可见节点无合法完整future，原后验仍要求12帧token|FutureEncoder未使用fut_mask；mask-only人工补值影响有标签目标的q|不代表训练后验合法使用监督目标GT本身是泄漏；不代表所有CVAE都无法解决|

源码：[资格]({AF}data/preprocessor.py#L80)、[后验]({AF}model/agentformer.py#L171)、[损失]({AF}model/agentformer_loss.py#L1)。本地逐段SHA见`static_source_anchors.json`。

## C、S和未解决范围

C为当前窗口8帧均可见的行人ID交集，仅由历史决定。S⊆C为有完整12帧标签且属于冻结监督目标集合的节点；U=C\\S。记录未来是否存在只用于训练标签与事后评分资格，不参与C构建或推理前向。这里故意不填补历史不足节点；不解决未来新行人生成、birth/death、身份切换。离开视野不等于停走或物理消失。

下表仅复用Stage7.1的Stage5.1支持区域，并筛选|S|>0窗口。不能外推为官方74文件发生率：

{table}

全部支持窗口另见`saved_cohort_summary.csv`：HOTEL train2221/val149，ZARA2 train1959/val172；有监督分别2121/94、1684/172。差额是|S|=0窗口，不能参与监督平均。逐行核验发现这些支持区域中完整future节点恰好都属于S，未将该偶然一致提升为普遍定义。HOTEL验证94个有监督窗口全部含缺未来邻居；ZARA2为123/172。计数单位为窗口及节点窗口，不是独立行人数，不能按数万个独立样本做显著性推断。

最大历史完整上下文C=73、S=53、缺完整future20，位于students001训练支持start30。该73人不是HOTEL验证人数，也不是官方74文件重新扫描最大值。短历史节点仍被排除，数量另表保存。

## 不改变原失败结论

Stage7.1 HOTEL(start14950,target344,neighbor343)将未知邻居补值从末位置改为末位置+100m，监督后验均值最大差0.5597405434，q对未知future梯度L1为0.1061537489/0.0149514638；ZARA2最大差0.4992415309（完整原始精度以JSON为准）。这只是非法补值负探针，绝非预测性能。两种HOTEL补值KL恰被clip为2，sample项相同，不能编造已观察的KL变化。NaN补值导致非有限loss，NaN×0不安全。

原始10个反事实、raw变体SHA及检查字段已复用至`counterfactual_reuse.json`；HOTEL shorten案例原本仅1帧future，与delete后效果相同，不算额外独立证据。7个完整标签真实batch loss/backward通过；3个混合标签case拒绝训练。适配器的prior推理通过，与原训练后验NO-GO同时成立。[原报告](../stage7p1/STAGE7P1_REPORT.md)

本轮没有统计“新增这些邻居会使预测更准”，真实问题的存在性与新增信息的预测价值是两个尚未打通的证据环节。
''')
    lit_rows='|论文|未来缺标签邻居|仅历史上下文|后验处理|动态人数|标准ETH/UCY目标|判定|\n|---|---|---|---|---|---|---|\n'
    for r in sources:
        lit_rows+='|['+r['name']+']('+r['paper_url']+') '+str(r['year'])+'|'+r['incomplete_future_context']+'|'+r['history_only_context']+'|'+r['posterior_treatment']+'|'+r['dynamic_population']+'|'+r['standard_ETH_UCY_targets']+'|'+r['relation_to_proposal']+'|\n'
    report('02_PRIOR_WORK_AUDIT.md','''
# 近年论文、源码与重合审查

结论：上下文/监督集合分离、观测与标签有效性分离、带缺失掩码的生成推断均已有实质先例；不能把这些常规做法宣布为新算法。尚未核实到“与本草案完全相同的AgentFormer联合潜变量处理”不构成新颖性证据。

检索日期2026-10-17。按nature-academic-search多源流程，对2022—2026六组用户指定关键词检索作者/arXiv/CVF/PMLR/出版社/GitHub，并保留2020/2021基础工作。OpenAlex备用脚本HTTP429；本轮用一手源补充，非穷尽性系统综述。6篇开放全文PDF取得并阅读方法/实验；原始PDF只放忽略缓存，发布SHA、来源与原创分析。并发PDF抽取曾报告xref修复提示，6份文本均成功提取；日志未逐文件分离警告，故不推定具体文件。关键公式按论文上下文核对，不把抽取错误当方法结论。

以下逐篇回答用户要求的五个资格问题。“未知”不能改成“没有”，论文支持8/12或ETH/UCY不代表测试目标、预处理和K20完全相同。

'''+lit_rows+'''

## 最直接的源码和方法证据

Trajectron++固定`1031c7b…`：`preprocessing.py:88–150`为目标取未来标签，但从scene_graph取得的其他邻居只读取[t-max_ht,t]；`mgcvae.py:658–680`用目标未来构造q。因此“监督目标完整而邻居不必完整”本身并不新。**未认证整个官方入口严格因果**：`scene.py:82–85`对预计算图传入正t_fut，`scene_graph.py:204–219`会对包含未来区间的adj_cube取max。该静态路径可能使用未来关系；是否触发取决于配置及图预计算，未运行反事实，不升级为全部实验已证实泄漏。可用该实现证明目标/上下文结构先例，不能用它宣称历史资格已通过本项目准入。[源码](https://github.com/StanfordASL/Trajectron-plus-plus/blob/1031c7bd1a444273af378c1ec1dcca907ba59830/trajectron/environment/scene_graph.py#L204)

SocialVAE冻结`eb23cba…`：`data.py:195–227`保持完整目标，但对其他行人的缺测使用sentinel；`social_vae.py:170–225`训练时backward RNN还聚合**可用的邻居future**，并非后验完全不读邻居未来。推理y=None与训练后验必须分开。既有可用性mask不证明它与AgentFormer联合CVAE语义相同，也不能重新启动已经结束的SocialVAE结构路线。[源码](https://github.com/xupei0610/SocialVAE/blob/eb23cbae562bac2c2dcf0d81529e6ab2ac280962/social_vae.py#L170)

TranSPORTmer §4.1–4.3有任务条件mask和独立NaN-mask；补充§4.1明确训练中忽略不可用门将观测，§4.2直接报告在原始不完整数据上训练。补充§6还有ETH/UCY8/12、2.5Hz的确定性实验。因此“允许缺失、分别mask输入/监督”已有直接方法先例。它是Set Attention确定性补全，不是本项目CVAE联合后验，不能据此宣称全部细节相同。未来条件在其补全任务中可以合法存在；换到forecast模式仍需单独限定只能条件于历史。[全文](https://arxiv.org/html/2410.17785v2)

Posterior Matching §3与官方`models/vae.py:116–150`：完整x编码的样本用于训练另一个任意观测条件后验；AppendixE专门讨论直接零填充送原VAE的失败。它通常假定基模型完整训练数据/人工mask，不能拿来证明天然缺未来标签和非随机缺失已经解决。将任意mask后验、prior completion直接当新思想风险很高。[论文](https://arxiv.org/abs/2201.12414)；[冻结代码](https://github.com/lupalab/posterior-matching/tree/08a355ddd22d0cfb14287e02bf30b439c21a0058)

GC-VRNN §4 Eq1把历史mask与未来y分开，但Eq19明确对未来预测计算likelihood，Eq20说明all-agent聚合；不能从“缺失轨迹”标题推导无GT未来邻居已被正确边缘化。作者仓库`b0bcad9…`目前只有README/补充PDF和legal-review说明，并非可从头训练代码。README转向的UniTraj-pytorch `bfebe75…`实际标题是**Sports-Traj(ICLR2025)**，不是已核实的GC-VRNN或GMC-VRNN权重复现。不得误认模型。

MS-TIP §3.2补的是历史坐标；§3.6 Eq22/23仍使用GT控制点和轨迹监督。源码README/train已冻结，不声称通过全数据因果测试。U2Diff §4用可变条件mask完成多主体补全，§5以四个体育数据集为主；这是不同生成机制/输入任务，未证明天然缺标签时的训练likelihood可直接套用。上述方法足以否定宽泛的新颖性宣称，却不足以宣告所有动态缺标签问题均已彻底解决。

## SocialTrans与Inter-GraphFormer补充到什么程度

SocialTrans：出版社Introduction/Proposed approach/Conclusion可检索到的正文说明：目标和邻居组分别构造运动状态张量，以SIE与NPIE在各观测时刻并行提取意图，结果经交互decoder输出；优化器结合位置距离/相邻向量角度，并对多结果聚类。论文明确提到邻居数变化。**仍未取得后验token、缺失标签mask、损失分母的公式和可冻结作者源码**；不能将“任意人数”解释成“自然缺future标签已解决”。

Inter-GraphFormer：出版社可获取Introduction包含有向加权且稀疏的空间图、并行时空Transformer、dual-GLU融合、CVAE潜空间及图Transformer未来decoder；2026年7月PhysicaA694:131566身份核实。**未取得完整后验/损失公式和作者代码**，不能判断C/S与本草案是否完全相同。

两篇本轮出版社下载均HTTP403；此前OpenReview需浏览器验证，不绕过。只补充可合法获取的正文细节，不能写成已全文精读。来源与失败保存在`literature_sources.json`和忽略缓存。2026检索也定位GMC-VRNN(IEEE TPAMI，2025早期在线、2026卷48期3，[身份来源](https://pubmed.ncbi.nlm.nih.gov/41171661/))但全文/源码未完成核查，未用其摘要宣告相同问题已解决。此项文献盲区直接阻止“新颖性已通过”。

## 重合判断

已有方法实质覆盖：C与S分离；独立观测/GT有效性mask；缺失观测生成补全；部分条件后验。尚未证明覆盖：保持AgentFormer特定联合潜变量及DLow分布时，未知未来邻居对监督目标后验的精确或有效边缘化。后者只是待证研究空白，没有独特机制、预测收益和资源证据，不足以启动训练。
''')
    report('03_MATHEMATICAL_FEASIBILITY.md',r'''
# 数学可行性与原模型语义变化

结论：可以定义不虚构未来GT的训练目标，但这需要更改原AgentFormer后验、身份映射及损失聚合，不能称loader修复，也不是已经证明有效的新算法。以下仅为静态数学审计；没有实现或训练替代网络。

## 条件与张量

令c=|C|、s=|S|>0、U=C\S，H_C=E(X_C)，X_C∈R^{c×8×2}，Y_S∈R^{s×12×2}。历史token为8c×d；真实future token仅12s×d；d=256，z_i∈R^32（所审HOTEL/ZARA2配置）。用当前和先前历史定义C、scene_orig和历史关系；训练标签资格S单独保存。推理对所有C预测，再由评测程序提取固定目标，不能先凭未来存在性删节点。

历史缺帧但不满足完整8帧者不在本问题C中。S=∅时没有本定义的监督目标，该窗口跳过监督训练，不能除以零，也不把U当静止监督。

## 为什么只mask原始loss不够

原FutureEncoder把12c个future token自注意力融合后，cross-attend 8c个history token，逐agent时间池化给q_i。其forward没有读取fut_mask。原generate_mask与AgentAware身份分解假设query/memory有同一个agent_num；简单将12c改成12s而不改身份映射会把“同一agent”判错。原KL对所有c聚合并clip，重建/sample即使乘零mask仍按c归一化；加入U稀释监督梯度。NaN×0无效，未知位置作为零/末值参与future attention同样是伪标签。

## 一种自洽但常规的受限变分构造

固定历史后，维护版先验是各节点对角高斯乘积（共享历史上下文不等于全协方差）：

\[
p_\theta(Z_C\mid X_C)=\prod_{i\in C}p_{\theta,i}(z_i\mid H_C).
\]

为说明可行性，可以只为合法Y_S建后验token，自注意力只允许S标签；cross-attention的memory为C历史，query身份S与key身份C用真实ID对齐。未来token的速度只由该监督目标最后历史点和合法未来点得到。U没有“假future token”；但联合decoder仍给U潜变量和预测输出：

\[
q(Z_C\mid X_C,Y_S)=q_\phi(Z_S\mid X_C,Y_S)\prod_{u\in U}p_{\theta,u}(z_u\mid H_C).
\]

其中q_S可保持逐节点对角高斯分布参数输出，允许参数共享已标注节点的合法Y_S信息；这是训练后验，不是推理偷看标签。U由历史prior采样，不能detach prior重参数梯度后仍宣称优化同一个ELBO。没有标签不等于没有latent，也不意味着其latent应被强制置零。

采用原论文固定方差高斯重建解释，联合decoder是从X_C,Z_C到所有未来**均值**的确定性映射f_θ。其时序自回归输入是自身生成均值，不是未标注节点真实future。条件于Z_C，输出噪声独立的假设下，积分未观测Y_U后，已观测likelihood只保留S：

\[
\log p_\theta(Y_S\mid X_C)\ge
\mathbb E_q\!\left[\sum_{i\in S}\log p_\theta(Y_i\mid X_C,Z_C)\right]
-\sum_{i\in S}\mathrm{KL}(q_i\Vert p_i).
\]

U的KL因q_U=p_U逐项为0而消去，不是“给没标签的节点随意mask惩罚”。Y_U完全未被伪造；Z_U通过期望被边缘化。若换成以真实Y_U为输入的随机自回归likelihood，该推导不再自动成立，必须重新积分，不能只照搬这条式子。

该受限q有效但可能很松：Y_S也携带Z_U的信息，而q_U固定prior将忽略它。解析双高斯例`analytic_checks.json`与数值积分测试证明ELBO≤log evidence，同时gap可明显大于0。这是概率公式验证，**不是轨迹性能或新网络**。没有理由仅凭式子就预期ADE提高。

## 从数学ELBO到原工程目标

设m=s；完整12步目标的重建为L_rec=(1/m)∑_{i∈S}∑_{t=1}^{12}||Y_it−f_it||²。它沿用原始每目标时间求和尺度，不偷偷改成每坐标平均；KL_S=(1/m)∑_S KL，原cfg clip_min=2的操作位置应保持在平均之后；sample辅助项为(1/m)∑_S min_k∑_t||Y_it−Ŷ_ikt||²。

当S=C，分母、求和和原有sample/clip可以退回原目标形状。S⊂C时，上述后验、KL范围和分母都不同于冻结原实现。保留clip与best-of-K辅助项的工程目标**不是精确边缘NLL，也不再是纯ELBO**。数学上可定义不等于本阶段获准修改官方loss。

有部分future标签但不足12帧的U，此构造不使用其零散标签；若以后按逐时刻标签集合S_t训练，需要单独定义token/latent与监督权重，不是同一消融。本阶段不把其已知点与未知点一起填成常数。

## DLow不能自动沿用

DLow的历史qMLP生成20个(A,b)，mean=True使用b作为各head潜变量；不是20次独立CVAE prior抽样。它训练时reconstruction需要Y_S，而KL与diversity可对C定义，因为不需要GT。一个明确但尚未实施的选择是recon按S平均，KL/diversity按C平均，保持两项合法的无标签正则；若三项都改S，正则对象也改变，需作为不同目标报告。

新CVAE后验/训练上下文改变后，需要重新训练CVAE和与其配对的DLow；旧作者权重只证明原协议性能，不能当新协议训练结果。重新定义DLow聚合不是原论文完全复现。

## 缺失机制与选择偏差

上述观测likelihood是给定标签索引模式的可定义目标。跨样本直接优化它以恢复全体未来分布，还需要缺失机制可忽略等统计假设。行人离开视野可能与未来位移相关，属于潜在非随机缺失；S可携带未来存在性信息。这里只可声明在固定有标签队列上的估计与评价，不能声称已识别所有行人的真实运动分布。没有建模缺失机制，本构造不解决MNAR偏差，也不支持将“未观测未来”当作真实终止事件。

## 必须申报的语义变动

|位置|变化|性质|
|---|---|---|
|历史loader与origin|C仅按8帧历史；S与GT分离|协议改变|
|FutureEncoder|12S queries，对8C memory；agent-ID矩形映射|核心后验与attention身份语义改变|
|U latent|posterior无标签token，使用历史prior|变分族改变|
|KL/重建/sample|标签范围和分母显式定义|训练目标改变|
|decoder输出|继续预测C；只监督/评分S|节点输出与评价接口需声明|
|DLow|新CVAE配对，标签loss/无标签正则区分|重新训练和目标审计|

这是可行性推导，不是可直接启动的最小创新方案。一般目标/上下文分离、mask及缺失变量边缘化已有研究；目前没有数学机制区别足以通过新颖性门槛。
''')
    report('04_PROTOCOL_COMPARABILITY.md',r'''
# 主基准与历史上下文协议的可比性

结论：能够保留官方ETH/UCY主评价，同时单列历史上下文研究轨道；无法把改变上下文资格后的指标直接宣称为对原AgentFormer的严格公平提升。本轮没有新指标或新排名。

## 两个轨道

|项目|P20官方主协议|P8历史上下文辅助协议（仅设计）|
|---|---|---|
|五折/时间/单位|ETH,HOTEL,UNIV,ZARA1,ZARA2；8/12；0.4s；米|同左|
|输入集合|官方get_valid_id要求完整20帧资格|C8仅依赖完整8帧历史|
|评分目标|官方P20监督/评测ID集合S20|为隔离上下文影响，固定同一S20，只在预测后评分|
|未来GT|后验/监督合法使用；未来存在性参与离线资格|只在训练后验的合法标签/损失/评分使用，推理资格和origin不使用|
|候选|完整CVAE+DLow K20，mean=True原语义|同预算但必须配对重新训练的CVAE/DLow|
|主指标|minADE20、minFDE20分别取最小|同定义，独立表格|
|作者权重|已在Stage7.3原协议验证|不能称为该新协议正式复现或公平训练对照|

保留官方74文件和train/val/test序列划分；官方ETH traj_scale=2，坐标还原后以米评分。UNIV两测试序列按目标数汇总、五折宏平均等权；窗口目标与行人等权是两种不同口径，不混用。K1另列。原始K20不是GT重建、FPC或GT重排。

在P8中，推理先预测全部C8，再由评测器按冻结S20取出结果；不能先告知网络哪些人有12帧future、向attention透露S20或者仅对S20推理。S20未来资格仅决定最终评测对象；因此这仍是完整future目标上的benchmark，不代表对所有未来消失者已建立真实ADE/FDE。

## 最低公平对照设计（未执行、未批准训练）

1. P20表只保留原始结构/同样资格的复现；若报告一个新变体的P20成绩，必须用该轨道自己的数据与训练结果。
2. P8内部所有学习模型从头在相同C8、同S20、同训练/验证区域、同GT SHA与候选预算训练。必须包括一个合法缺标签后验的最小适配参考；原“补值+仅mask重建”是已失败负探针，不是可靠性能基线。可加入一个目标中心的公开方法作跨架构参考，但不能当单因素结构控制。
3. 固定P8后，比较同一合法后验在C=S20和C=C8的结果，分离“多历史上下文”效果；再固定C8比较不同合法后验机制。新增机制须与这类最小适配相比，而不是只打败不兼容原实现。
4. 同目标CV-Last2/OLS4/OLS8均只使用目标历史；所有学习对照的容量、训练访问次数、选择规则、K20噪声/heads预算要一致。GT、targetID、序列化和官方/独立评估双核验。
5. 增加history-only上下文全量支持统计、future完整与不完整邻居分组、负控制。若做未来存在性反事实，资格变动与GT数值变动分开；训练后验依赖合法Y_S不应被错误要求不变。

没有可训练且公平的最小适配基线时，就不能声称提升可独立归因。以上设计在原则上可解释，但尚未执行；本轮NO-GO不授权以此启动下一阶段。

## 数据曝光与停止界限

Stage7.1统计来自旧Stage5.1五录像支持，不可直接代替官方74文件发生率或窗口数。Stage7.4/7.5两个官方validation fold来自部分共同来源，Stage7.5记录重复目标窗口3944，不是两个完全独立验证集。旧test已观察，不允许以其结果挑选新机制、K、checkpoint或窗口；将来若进行新实验必须明确这个曝光限制。协议变化是研究设置而非创新点。

官方身份SHA清单复用并单独保存`official_data_identity.csv`，本轮只核对身份与历史保护SHA，不读取测试轨迹误差，不重算Stage7.3成绩。
''')
    budget=load('results/tables/stage7p6/resource_audit.json')
    disk=budget['current_storage']['free_bytes']/(1<<30)
    report('05_RESOURCE_FEASIBILITY.md',f'''
# 资源、训练成本与消融可承受性

结论：RTX3080 10GB上原完整20帧协议的完整epoch已运行；**73人混合标签训练可行性尚未建立**。不能用推理成功代替训练准入，也不能删除拥挤窗口制造可行性。SSD障碍已解除，本轮观察可用约{disk:.2f}GiB；不沿用Stage7.4当时不足1GB的过期存储NO-GO理由。

## 实测（复用Stage7.4 execution_v2）

|HOTEL原协议|CVAE一完整epoch|DLow一完整epoch|
|---|---:|---:|
|有效窗口/optimizer updates|3118/3118|3118/3118|
|目标窗口|29676|29676|
|最大上下文|57|57|
|总时间(s)|509.882801|307.354930|
|peak allocated(bytes)|6471955968|3982871040|
|peak reserved(bytes)|9590276096|9642704896|
|采样整卡峰值(MiB)|9677|8982|
|OOM|否|否|

这些是完整训练epoch而非小batch外推；但各只有一轮，不能推断100/50轮收敛。DLow使用1epoch CVAE只是工程计时；其CVAE未被优化器改动。optimizer/scheduler/RNG严格恢复验证已通过，checkpoint索引与SHA保留在原JSON。

原协议HOTEL CVAE100+DLow50按实测点估约18.4322h；原五折估计点86.0389h，范围60.0447–129.5158h，包含原Stage7.4的跨fold/验证/checkpoint情景假设，不是实测五折。每个匹配基线/变体都要独立承担训练成本；两个受控方法×至少两开发fold的原预算点量级约73.7h，再增加多种子或五折会成倍增长。本轮不批准任何这些run。

## 73人的证据边界

Stage7.1最大C=73/S=53仅运行learned-prior K20 inference，无DLow，无后验backward：allocated2684130304B，reserved9512681472B，整卡采样峰9315MiB（仅3次采样，可能漏瞬时峰），0.39977s，未OOM。不能和训练值混为一次同任务测量。

attention矩阵规模主要随N²增长：73/57=1.28070，平方1.64020。单层FP32、20候选、8heads、12N token的一个attention矩阵从299427840B增至491120640B；这不是整网总显存，autograd保存多层且包含交叉attention、激活、参数、优化器等。

仅作敏感性情景，把原CVAE allocated按线性人数缩放为8.289GB，按平方为10.615GB（十进制），后者约9.886GiB尚未包括其他整卡开销；二者都不是实际峰值、下界或可信上界。原训练整卡9677MiB已接近10GB卡容量，allocator reserved、allocated与整卡读数不能直接相加。缺标签后验12S自注意力和12S×8C cross-memory可小于12C，但历史与decoder仍对C交互、DLow展开20候选，不能据此保证省显存。

## 时间与消融估算

若窗口数完全不变，HOTEL基础18.432h按1/人数比/平方比分别是18.432/23.606/30.232h，只是敏感性情景，不是新协议上下界。C8可能引入原P20没有的窗口且官方74文件的C8规模未重扫，不能据旧Stage5.1最大73精确预测所有fold。ETH原配置max_train_agent=32，其他fold100，原作者截断配置本身也须披露，不能为了本研究OOM临时新增截断或宣称所有原折均无上限。

需要重新训练CVAE：是，训练后验、C资格和监督分母改变。需要重新训练DLow：是，CVAE/上下文分布及有标签reconstruction对象变了。最低公平实验还要同协议最小适配参考、同目标CV与机制消融，不能只跑一个模型得出独立归因。

本阶段禁止新训练，故没有73人混合标签完整backward证据，没有验证非截断显存执行方案，没有新协议完整epoch计时。资源GO不能填写“通过”；结论是**未建立**，不是已证明必OOM。保持协议公平性前提下未来可讨论独立执行方案，但本轮不自动测试混合精度/重计算/分块或改变K20。
''')
    gate_table='|门槛|状态|依据|\n|---|---|---|\n'+''.join('|'+g['condition']+'|'+g['status']+'|'+g['evidence']+'|\n' for g in gates)
    report('06_FINAL_RESEARCH_DECISION.md','''
# 最终科研判定：NO-GO

停止AgentFormer“历史合法上下文+未来缺标签联合后验”这条具体创新开发路线；不自动开展Stage7.7，不增加GNN/attention/关系GRU/门控，不启动训练。保留原始AgentFormer作为已经复现的公开基准，保留所有历史失败与有效工程证据。

'''+gate_table+'''

## 回答五个核心问题

1. 问题真实吗？是，Stage7.1实际窗口中存在历史完整但未来不完整邻居，原FutureEncoder mask-only负探针失败。没有证据说明新增邻居一定带来预测收益。
2. 已有论文解决了吗？宽泛的C/S分离、有效性mask与部分条件推断已有明确先例；精确到AgentFormer联合latent+缺失未来边缘化并未核实有完全相同实现。两篇指定近文完整方法仍不可得，不能以未发现当成新颖性通过，也不能宣布整个领域已无开放问题。
3. 有区别的数学机制吗？本轮给出了自洽但受限的变分构造，U从prior、S仅用真实标签。它体现常规缺失变量处理，可能产生较松ELBO；尚没有不同于已有思想且能解释预测收益的新机制。后验/身份映射/聚合需改，不能称纯适配。
4. RTX3080能做吗？原协议训练已实测，73人仅推理通过。新混合标签训练显存、完整epoch和可承担消融预算未建立，不宣称已不可运行，也不能填写资源GO。
5. 能保留标准主协议吗？可保留官方完整20帧五折主表，把C8实验另列；C8内必须重新训练匹配基线。仅8/12、K20相同不足以严格公平，不能把不同上下文条件下的结果直接排名。

## 为什么不启动一个“先试试看”训练

GO是所有条件同时成立，不能由“问题真实”替代“新颖性、独立归因、资源可行”。此时训练一个修改后模型即使提升，也难区分上下文更多、后验变更、目标权重与不同训练分布。没有独特机制支撑，训练预算投入无法补足研究论证。

本轮没有证明原AgentFormer预测失败，没有推翻Stage7.3权重复现，没有将未完整future节点补成静止。原Stage7.1在“保持核心语义不变”限制下NO-GO仍然正确；本轮允许数学讨论也不等于那次准入突然成功。

## 论文方向调整建议（仅建议，等待用户决策）

优先将已积累材料整理为硕士论文中的**协议边界与可复现性/动态观测鲁棒性评价**：明确离线上下文资格、历史可见上下文、有效标签和部署输入之间的边界，保留真实反例、同目标CV和公开模型复现。这是可以证据化的研究内容，但单靠协议重命名和mask工程未必达到独立算法小论文要求。

若第四章必须有新算法，先由用户决定是否调整数据任务或重新立项：需要有原生可见性/标签缺失标注、能排除评测与选择偏差的数据，并把自然缺失与人工mask区分。不能在目前已充分曝光的ETH/UCY小开发集上连续换模块直到出现最低误差。该建议不是新模型方案，不宣布数据已取得、算法已实现或性能已提升。

当前不具备本具体AgentFormer创新路线的正式训练启动条件。不再自行更换模型或调整运动特征/候选数量。没有新增ADE/FDE，未运行项明确保留未验证。
''')
    report('STAGE7P6_REPORT.md','''
# Stage7.6 研究总报告

**最终判定：NO-GO。停止当前AgentFormer因果上下文创新路线，等待用户科研决策。**

在`feat/stage7p6-causal-context-research-decision`上完成历史证据复核、2022—2026论文/源码核查、C/S训练目标推导、协议公平性及资源审计。没有模型修改、新预测推理、优化器更新、正式训练或测试指标重算。

真实问题已确认：历史可见邻居可能缺完整future；原AgentFormer联合FutureEncoder不能靠重建mask消除未知future影响。未来GT数值、未来存在性资格、训练后验缺标签是不同问题，Stage7.1原NO-GO保持不变。复用的10个原始数据反事实为严格历史适配器prior测试，不能充作官方loader严格历史准入证据。

已有Trajectron++、SocialVAE和TranSPORTmer分别提供目标/上下文分离或可用性mask的实质先例；一般masked/partial后验也非新思想。SocialTrans与Inter-GraphFormer补充了可获取的正文结构，但完整mask/KL/监督机制及作者源码仍未核实。没有可以宣告创新性通过的新机制。

可以写出仅用Y_S、对U潜变量prior补全的合法变分目标，不必补造未来；但变分族可能很松，原sample/clip工程目标不是精确NLL；后验、agent-ID映射、KL和分母须显式变化。还存在非随机缺失与未来资格选择偏差，不能声称已解决全体行人的真实未来预测。

标准ETH/UCY五折、8/12、0.4s、米、DLowK20主协议可保留；C8是独立辅助上下文协议，需匹配重训参考和公平消融。不同上下文资格不能混榜。旧test已观察，不能称为未来未接触确认集。

Stage7.4原训练预算HOTEL100+50约18.43h点估，五折原估60.0–129.5h；Stage7.1最大73人只运行prior推理，原57人训练整卡已到9677MiB。新混合标签训练资源**未建立**，不编造OOM，不删拥挤窗口。当前存储足够，不重复已解除的SSD阻碍。

## 交付索引

- [问题与反事实边界](01_PROBLEM_DEFINITION.md)
- [近年文献及源码矩阵](02_PRIOR_WORK_AUDIT.md)
- [数学自洽性、token与latent语义](03_MATHEMATICAL_FEASIBILITY.md)
- [主基准与历史上下文公平对照](04_PROTOCOL_COMPARABILITY.md)
- [实测预算、73人风险与消融成本](05_RESOURCE_FEASIBILITY.md)
- [门槛逐项判定与方向调整](06_FINAL_RESEARCH_DECISION.md)
- `results/tables/stage7p6/`：数据/代码SHA、全部/有监督窗口双口径、future长度直方图、反事实/失败探针复核、文献矩阵、解析检查、资源情景、测试及历史保护记录。

配置预先冻结提交`a27cf62`；分析基于Stage7.5 `a00c9a38793135b77635ee4f99862f1f017eee23`。实际脚本以execution_record与交付SHA索引为准；同一提交中无法自包含最终Git SHA，最终提交由交付时git log给出。所有缓存PDF/第三方源码/权重保持忽略，不进入Git。

研究执行没有优化器更新或新增数据推理；完整回归测试中的既有人工前向/后向和优化器检查不属于正式研究训练。本轮完整测试442通过（含9项新增解析/审计测试），详见unit_tests.xml/txt；历史SHA核验和最终交付检查见historical_preservation.json与delivery_checks.json。没有把解析高斯例或单元测试当作真实轨迹性能。

只完成本地提交，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p6-causal-context-research-decision
```

完成后停止，不自动开启Stage7.7。
''')
    print('Wrote',len(sources),'literature rows, 7 reports, target-conditioned evidence and decision gates')


if __name__=='__main__':
    main()
