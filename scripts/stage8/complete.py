"""Publish Stage8 evidence and a research decision, without further model runs."""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from contracts import validate_research_gates
OUT=ROOT/'results/tables/stage8'
REPORT=ROOT/'reports/stage8'
RE='7358b3d4fd3fcf160c18be9425445a67ee8211ee'
MF='f1b89b0e80ce95646214923e271e3c280d305cfa'
REURL='https://github.com/cocoon2wong/Re/blob/'+RE+'/'
MFURL='https://github.com/DSL-Lab/MoFlow/blob/'+MF+'/'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads((ROOT/p).read_text())


def write(name,text):
    with (REPORT/name).open('x') as f:f.write(text.strip()+'\n')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    repositories=[];file_rows=[]
    for name,rel,url,expected in [
        ('Resonance','experiments/stage8/cache/Re','https://github.com/cocoon2wong/Re',RE),
        ('Qpid','experiments/stage8/cache/Re/qpid','https://github.com/cocoon2wong/Project-Qpid','c7233b3f7e7305294b1d5644e695b44727ac0bea'),
        ('Luna','experiments/stage8/cache/Re/dataset_original','https://github.com/cocoon2wong/Project-Luna','b25d44354a75f7652c25c37aa9c16b353578ad79'),
        ('Resonance author weights','experiments/stage8/cache/Re_weights','https://github.com/cocoon2wong/Project-Monandaeg/tree/Re','42d88ba95e45f441f72318d673c5effd5ef356ec'),
        ('MoFlow','experiments/stage8/cache/MoFlow','https://github.com/DSL-Lab/MoFlow',MF)]:
        p=ROOT/rel
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
        assert commit==expected
        assert not subprocess.check_output(['git','diff','--name-only'],cwd=p,text=True).strip()
        repositories.append(dict(name=name,url=url,path=rel,commit=commit,tracked_source_unchanged=True))
        for path in subprocess.check_output(['git','ls-files','-z'],cwd=p).decode().split('\0'):
            if path and (p/path).is_file():
                f=p/path
                file_rows.append(dict(repository=name,path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size))
    json_new(OUT/'source_versions.json',dict(repositories=repositories,playground_submodule='Not initialized; GUI not needed',
        third_candidate_engineering=False,source_licences=dict(Resonance='GPL-3.0',MoFlow='MIT'),
        third_party_source_and_weights_committed=False))
    csv_new(OUT/'source_and_weight_SHA256.csv.gz',file_rows)
    probe=load('results/tables/stage8/resonance_probe.json')
    author=load('results/tables/stage8/resonance_author_checkpoint.json')
    mf=load('results/tables/stage8/moflow_cpu_probe.json')
    manifest=load('results/tables/stage8/resonance_validation_manifest.json')
    with gzip.open(OUT/'resonance_history_audit_targets.csv.gz','rt') as f:targets=list(csv.DictReader(f))
    sources=defaultdict(list)
    for r in targets:sources[r['sequence']].append(r)
    counts=[]
    for source,rows in sorted(sources.items()):
        counts.append(dict(sequence=source,target_windows=len(rows),person_ids=len({r['target_id'] for r in rows}),
            potentially_ambiguous_target_windows=sum(r['same_sector_position_motion_ambiguity']=='True' for r in rows),
            qualifying_pairs=sum(int(r['qualifying_pairs']) for r in rows),
            interpretation='Descriptive history exposure, no causal/accuracy claim; overlapping windows not independent'))
    csv_new(OUT/'history_exposure_by_source.csv',counts)
    versions={}
    for package in ('torch','numpy','scipy','einops','opencv-python','tensorboard','PyYAML'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not_available'
    json_new(OUT/'environment.json',dict(python=sys.version,packages=versions,
        execution_venv='experiments/stage8/cache/venv',
        frozen_dependency_path='experiments/stage4/venv/lib/python3.10/site-packages',
        isolation='Separate venv and execution workspace; existing dependency tree imported read-only, no old environment packages installed/updated',
        author_requirements_exact=False,Resonance_declared='Python3.10,torch2.0.0,numpy1.26',
        MoFlow_declared='Python3.11,torch2.0.1,numpy2.2.4 plus accelerate/EMA/tensorboardX; exact full trainer environment not installed',
        MoFlow_probe_dependencies='easydict1.13 reused from existing frozen compatibility path; no trainer imported',
        GPU='RTX3080',driver='570.133.07',nvidia_smi_memory_MiB=10240,
        torch_visible_memory_bytes=probe['GPU_total_bytes']))
    # No author best-score files or event logs are parsed; only args and file hashes.
    releases=[]
    for p in sorted((ROOT/'experiments/stage8/cache/Re_weights').glob('re*/args.json')):
        cfg=json.loads(p.read_text())
        releases.append(dict(directory=p.parent.name,args_SHA256=sha(p),split=cfg['split'],
            checkpoint_paths=[str(x.relative_to(ROOT)) for x in p.parent.glob('*.pt')],
            model_name=cfg.get('model_name'),epochs=cfg.get('epochs'),lr=cfg.get('lr'),
            output_pred_steps=cfg.get('output_pred_steps'),learn_self_bias=cfg.get('learn_self_bias',cfg.get('compute_non_social_bias',1)),
            learn_re_bias=cfg.get('learn_re_bias',1),disable_linear_base=cfg.get('disable_linear_base',0)))
    json_new(OUT/'resonance_weight_release_audit.json',dict(args_only=releases,score_files_read=False,
        strict_loaded_checkpoint=author['checkpoint'],strict_loaded_checkpoint_SHA256=author['checkpoint_SHA256'],
        caveat='rehotel release is no-linear-base/no-self-bias ablation, not Full; no replacement chosen by metric'))
    csv_new(OUT/'model_inventory.csv',[
        dict(model='Resonance Full',parameters=probe['parameters'],source='Instantiated and strict author ZARA2 load',
             input='B,8,2 target; B,N,8,2 relative neighbors',output='B,20,12,2',training_stages=1,executed='GPU original forward/backward, no updates'),
        dict(model='MoFlow teacher A1',parameters=mf['models'][0]['teacher_parameters'],source='Original CPU backbone instance',
             input='B,1,8,6 context + B,20,1,24 noisy future + flow time',output='B,20,1,24 + B,20,1 scores',training_stages=1,executed='CPU backbone forward only; no flow sampler'),
        dict(model='MoFlow IMLE student A1',parameters=mf['models'][0]['student_parameters'],source='Original CPU backbone instance',
             input='context and latent ensemble',output='K trajectories; not executed',training_stages=2,executed='Parameter count only; teacher generation + student training needed')])
    protocol_rows=[
        dict(item='source',AgentFormer='Frozen74 official17-column files,8recordings',Resonance='Luna true_pos_.csv -> custom config/annotations; no74-file identity equivalence',MoFlow='SocialGAN/EigenTrajectory original pickle vs LED npy; two versions'),
        dict(item='window',AgentFormer='current frame grid, contiguous8+12, includesN1',Resonance='per-target track start/stop, step/interval; different indices; native partial-neighbor padding',MoFlow='original generator sorts unique frames20 and min_ped1 strict>1, so single-person windows excluded'),
        dict(item='context',AgentFormer='Complete20 valid_id, all legal selected nodes',Resonance='Visible at current obs; nearest cap; missing prefixes/suffixes padded; not fullfuture-neighbor qualification',MoFlow='original all_data[:,None] makesA1; LED format needs separate qualification audit'),
        dict(item='split',AgentFormer='Separate official train/val/test sequence lists',Resonance='Luna get_splits val=test; UNIV13 groupsuniv+univ3',MoFlow='train/test-only loading by main scripts; trainer val_loader defaults test_loader'),
        dict(item='ETH sampling',AgentFormer='Frozen official ETH file grid/scale2; outputmetres',Resonance='Author eth raw6frames, otherclips10; eth10 alternative; native declared0.4 not cohort equality',MoFlow='Chooseoriginal/LED explicitly; normalization fromtrain; no alignment by name alone'),
        dict(item='UNIV',AgentFormer='students001+students003 target-count weighted',Resonance='univ13 combinesuniv/univ3; exact ID/frame map unverified',MoFlow='raw/original/LED files need explicit source mapping and count check'),
        dict(item='K20',AgentFormer='DLow20 mean=True latent heads',Resonance='K1 repeats x Kc20 learned styles with random self/re-bias noise',MoFlow='20 correlated flow queries, teacher multistep or separately trained one-step student'),
        dict(item='metrics',AgentFormer='minADE20/minFDE20 independently, targetweighted then5foldmacro',Resonance='ADE_2D min acrossK, FDE ownlaststep; cohort differs',MoFlow='separate minADE/minFDE over20; end4.8s; cohort differs'),
        dict(item='current strict comparability',AgentFormer='Retained reproduced reference, oldtestnotusedselection',Resonance='NO direct ranking against AgentFormer',MoFlow='NO direct ranking against AgentFormer/Re')]
    csv_new(OUT/'protocol_comparison.csv',protocol_rows)
    literature=[
        dict(model='AgentFormer',year=2021,venue='ICCV',paper='https://arxiv.org/abs/2103.14023',code='https://github.com/Khrylx/AgentFormer',
             reviewed='Reused full methods and frozen source',input='joint history',output='CVAE+DLowK20',mechanism='Agent-aware temporal/social attention, learned prior, diversity learning',overlap='Ordinary social attention, latent diversification alreadycovered'),
        dict(model='TUTR',year=2023,venue='ICCV',paper='https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf',code='https://github.com/lssiair/TUTR',
             reviewed='Prior fullPDF/source; method/experimental sections reread',input='normalized target/neighbors+train modes',output='K predictions and probabilities',mechanism='Mode/social Transformer, regression+classification',overlap='Scoring/anchors/mode selection not new; no new model admission'),
        dict(model='SocialCircle',year=2024,venue='CVPR',paper='https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf',code='https://github.com/cocoon2wong/SocialCircle',
             reviewed='Full method Eq1-11 and existing frozen source identity',input='history neighbors, angular summaries',output='backbone K future',mechanism='Per-angular-sector speed/displacement,distance,angle means',overlap='Ordinary velocity/distance/angle additions and partition features alreadycovered'),
        dict(model='MoFlow',year=2025,venue='CVPR',paper='https://openaccess.thecvf.com/content/CVPR2025/html/Fu_MoFlow_One-Step_Flow_Matching_for_Human_Trajectory_Forecasting_via_Implicit_CVPR_2025_paper.html',code='https://github.com/DSL-Lab/MoFlow',
             reviewed='Full method/experiments, frozen teacher/student/train/eval/data source',input='history + K noisy futures/latent',output='K correlated candidates and scores',mechanism='K-shot flow matching regression+CE, IMLE/Chamfer one-step distillation',overlap='Fewer steps or generic distillation/coverage improvement not new'),
        dict(model='Resonance',year=2025,venue='ICCV',paper='https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_Resonance_Learning_to_Predict_Social-Aware_Pedestrian_Trajectories_as_Co-Vibrations_ICCV_2025_paper.pdf',code='https://github.com/cocoon2wong/Re',
             reviewed='Final CVF fullpaper+supplement; official frozen network+Qpid+Luna',input='target history+relative neighbor histories',output='linear+self-bias+resonance-bias K20',mechanism='Haar/differential motion, spectral pair product, angle aggregation',overlap='Motion/social decomposition, frequency features and social averaging alreadycovered'),
        dict(model='Inter-GraphFormer',year=2026,venue='PhysicaA694:131566',paper='https://doi.org/10.1016/j.physa.2026.131566',code='Not verified',
             reviewed='Publisher detailedIntroduction+metadata only; fullmethod unavailable',input='historical motion/social graph; exact eligibilityunknown',output='CVAE multi-future',mechanism='Sparse directedweighted spatial graph, parallel temporal/spatial attention, dualGLU',overlap='Dynamic/asymmetric interaction ideas overlap; precise formula comparison unknown, cannot claim novelty cleared'),
        dict(model='Reverberation',year=2026,venue='IEEE TPAMI earlyonline Oct7; author repo stillsays underreview',paper='https://arxiv.org/abs/2511.11164',code='https://github.com/cocoon2wong/Rev (literature only; not downloaded/admitted)',
             reviewed='Full available arXiv methods/supplement, formal identity DOI10.1109/TPAMI.2026.3741469',input='target/neighbor histories',output='latency-conditioned multi-future',mechanism='Retains temporal-spectral x angular social tensor; two reverberation kernels',overlap='Adding temporal angle tensors or generic delayed interaction to Re overlaps directly; within-sector identity pooling remains in inspected formulas'),
        dict(model='Multi-modal Knowledge Distillation-based Human Trajectory Forecasting',year=2025,venue='CVPR',paper='https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Multi-modal_Knowledge_Distillation-based_Human_Trajectory_Forecasting_CVPR_2025_paper.pdf',code='https://github.com/Jaewoo97/KDTF (literature only)',
             reviewed='Full methods3.2/3.3 and protocol4.1',input='Teacher trajectory+pose+text; studentlimitedmodalities',output='multi-future regression',mechanism='Intra/inter-agent embedding distillation withlocal/globalencoders',overlap='Different fromMoFlow distribution distillation, but generic KDnewness already insufficient; extra modalities not available for standard coordinateonly baseline')]
    csv_new(OUT/'literature_comparison.csv',literature)
    json_new(OUT/'literature_evidence.json',dict(date='2026-10-18',workflow='nature-academic-search multi-source + legitimate OA downloader',
        original_downloads=load('experiments/stage8/cache/literature/download_manifest.json'),
        extra_download=load('experiments/stage8/cache/literature/extra_downloads.json'),
        reused_full_texts=['experiments/stage7p5/cache/literature/agentformer_pdf.pdf','experiments/stage7p5/cache/literature/tutr_pdf.pdf',
                           'experiments/stage7p5/cache/literature/socialcircle_pdf.pdf','experiments/stage7p5/cache/literature/moflow_pdf.pdf'],
        keywords=['2024-2026 dynamic pedestrian interactions relative velocity','trajectory multimodal efficiency flow distillation',
                  'Resonance motion-geometry association and pooling','Inter-GraphFormer full method','Reverberation temporal social latency'],
        failure='AcademicMCP absent; OpenAlexfallbackHTTP429; Inter-GraphFormer publisher fulltext unavailable again; no bypass attempted',
        unavailable_details='Inter-GraphFormer missing posterior/loss/data formulas and author source stay unverified',
        nonprimary_sources_not_used=['automatedpaperreviews','aggregatorclaims','reddit'],
        no_third_model_engineering=True))
    gates=dict(problem=True,mechanism_difference=False,legal_data=False,fair_controls=False,hardware_cost=False,stop_rule=True)
    assert not validate_research_gates(gates)
    json_new(OUT/'final_decision.json',dict(decision='NO-GO',recommended_backbone=None,gates=gates,
        gate_meaning='False data/control/resource gates mean not fully established in a frozen end-to-end training setup, not proof impossible',
        exact_research_problem='Resonance within-sector independently averaged position and motion features erase association',
        issue_support='Architecture identity and synthetic frozen-author-weight witness, plus history-only validation exposure; predictive harm unproven',
        candidate_mechanism='Unimplemented joint geometry-spectral crossmoment in original aggregation; standard bilinear/DeepSets/GNN connections make novelty insufficient',
        no_go_reason='No currently justified distinct algorithm+fair frozen training protocol+measured full training budget; do not equate engineering success to a paper backbone',
        AgentFormer='Retain reproduced benchmark, no more innovation modifications',
        no_third_candidate=True,no_next_stage=True,no_training=True,no_automatic_push=True))
    csv_new(OUT/'candidate_scores.csv',[
        dict(model='Resonance',A_novelty=1,B_protocol_comparability=2,C_RTX_feasibility=3,D_fair_ablation=3,E_thesis_fit=4,
             rubric='Qualitative1-5, not performance scores; higherengineering score cannot override failed gates',decision='Not selected as current innovation backbone'),
        dict(model='MoFlow',A_novelty=1,B_protocol_comparability=2,C_RTX_feasibility=2,D_fair_ablation=2,E_thesis_fit=3,
             rubric='Qualitative1-5; CPU/singleagent/multistage restrictions; not a prediction ranking',decision='Not selected')])
    gpu_wall=probe['total_seconds']+author['total_seconds']
    json_new(OUT/'compute_audit.json',dict(Resonance_original_probe_seconds=probe['total_seconds'],
        Resonance_author_probe_seconds=author['total_seconds'],combined_recorded_probe_wall_seconds=gpu_wall,
        wall_scope='IncludesCPU scan/setup, excludes process import/launch and earlier failedimport; not CUDAkernel time or throughput benchmark',
        GPU_budget_seconds=7200,MoFlow_GPU_seconds=0,formal_epochs=0,
        max_tested_context=42,max_tested_target_batch=42,
        peak_allocated_bytes=max(x['allocated_peak'] for x in probe['cases']),
        original_paper_Resonance=dict(device='RTX3090',epochs=200,batch_size=1000,lr=.0003),
        available_ZARA2_weight_args=dict(epochs=200,batch_size=1000,lr=.0001),
        MoFlow_default=dict(teacher_epochs=150,student_epochs=150,teacher_sample_generation='Additional multistep train-pool offline generation;100step README example'),
        planning_only=dict(reference_train_targets=29676,reference='ExistingofficialHOTELtraincohort; not nativeLuna pooledatasetcount',
            updates_200ep_batch128=200*math.ceil(29676/128),
            hours_if_step_seconds={str(t):200*math.ceil(29676/128)*t/3600 for t in (.1,.5,1.)},
            caveat='Assumed step times, not measurements; validation/checkpoint/IObudget extra; full1000batch and epochnot run'),
        full_training_feasibility='Not measured for either new candidate; no false claim originalbatch1000 full200epochs verified'))
    csv_new(OUT/'execution_source_SHA256.csv',[
        dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts/stage8').glob('*.py'))])
    # Explicit failures are retained, including the corrected data dictionary error.
    json_new(OUT/'failures_and_limitations.json',dict(
        import_probe='Initial Qpid import failed because cwd lacked dataset_configs; resolved by isolated annotation metadata only, not source edits',
        MoFlow_initial_probe=load('results/tables/stage8/moflow_cpu_initial_failure.json'),
        MoFlow_resolution='Provide the declared past_traj_original_scale field; original model unmodified; retry passed',
        pretrained_HOTEL='Release directory contains declared ablation; not treated as Full',
        source_read_attempts='Several early reads preceded clone completion / guessed filenames absent; later exact paths inspected; not model failures',
        PDF_warning='pdftotext font mismatch warning; extraction succeeded; paper formulas checked with source',
        not_run=['Officialtestevaluation','Formalepochs','Teacherflowtraining/sampling','IMLEstudentforward/training','Fullbatch1000/fullepochtrainingbudget'],
        novelty='Inter-GraphFormer complete method unavailable; no declaration that it lacks a specific mechanism'))
    table='|用例|上下文/目标batch|输出|allocated峰值(MiB)|秒（多前向+反向）|\n|---|---:|---|---:|---:|\n'
    for r in probe['cases']:
        table+=f"|{r['case']}|{r['context_people']}/{r['target_batch']}|{r['output_shape']}|{r['allocated_peak']/(1<<20):.2f}|{r['seconds']:.4f}|\n"
    lit_table='|论文|机制与阅读范围|对本轮问题的重合/限制|\n|---|---|---|\n'
    for r in literature:lit_table+=f"|[{r['model']}]({r['paper']}) {r['year']} {r['venue']}|{r['mechanism']}；{r['reviewed']}|{r['overlap']}|\n"
    write('01_RECENT_RESEARCH_REVIEW.md','''
# 近期研究与查重

检索截至2026-10-18。使用nature-academic-search多源流程；无学术MCP，OpenAlex备用检索HTTP429，因此通过作者/arXiv/CVF与固定源码补充。不是穷尽性系统综述。仅Resonance、MoFlow进入工程审查；TUTR/SocialCircle/Rev/KDTF仅作方法先例，没有创建第三模型复现任务。

'''+lit_table+'''

## 已有工作解决到什么程度

研究问题A：AgentFormer已联合编码时空交互，TUTR已有mode/social两级编码；SocialCircle已编码角分区内运动、距离和方向。Resonance以中心化运动谱乘积和角聚合建模社会修正，普通速度/距离/角度或社会残差分解本身均非空白。Rev全文§III/IV Eq12–14进一步保留Th×Nθ角向时序谱，学习双核处理交互延迟；“把Re的时间维留下”“增加记忆/时延权重”与它高度重合。新检索定位2026年连续动态图/patch-wise图、2025多时刻交互等论文，但未完整阅读的只作排重线索，不能以摘要判定确切机制不同。

本轮具体问题是**位置—运动配对信息被独立聚合抹除**，并不是笼统说模型没有相对速度或时间模块。原代码和作者权重给出可复核的同表示反例；这仍不证明真实预测受损，也不证明简单的联合特征就是新算法。原论文补充已比较SocialPooling、GCN、SocialCircle与ResonanceGathering，不能重新添加普通图模块就宣称创新。

研究问题B：MoFlow §4.2/4.3已联合考虑K候选覆盖/一致性、回归与分类，并用IMLE/Chamfer进行一步蒸馏；少步生成、teacherstudent、集合匹配都已直接研究。KDTF §3.2/3.3将完整轨迹/姿态/文本教师的个体和交互知识蒸馏给较少模态学生，属于额外输入下的知识转移，不同于MoFlow分布蒸馏，也不能直接拿其指标作为纯坐标公平对照。当前没有取得本地teacher/student配对预测误差与同设备延迟—覆盖证据，因此不提出“再减层/再减步”的创新。

## 版本与全文边界

Resonance采用最终ICCV2025 CVF版（三位作者）及补充，不将早期arXiv四作者版本混作同一最终方法细节。Rev作者仓库README仍写underreview，而[公开身份](https://pubmed.ncbi.nlm.nih.gov/42848575/)已列TPAMI2026-10-07 earlyonline、DOI10.1109/TPAMI.2026.3741469；本文以此区分公开/正式状态，不把旧README当最新发表状态。

Inter-GraphFormer只取得出版社详细Introduction与元数据：有向加权稀疏图、并行时空Transformer、dualGLU、CVAE、图Transformerdecoder。完整后验/损失/资格公式和作者代码仍未取得，出版社请求失败，不绕过验证。因此指定文献中这篇**未达到完整方法精读**；其缺失细节构成新颖性未通过的限制，不能写成“它没有这个机制”。来源、PDF与版本SHA见literature_evidence.json及source_and_weight_SHA256.csv.gz。
''')
    write('02_RESONANCE_TECHNICAL_AUDIT.md',f'''
# Resonance技术审查

官方源码`{RE}`；Qpid=`c7233b3f7e7305294b1d5644e695b44727ac0bea`；Luna=`b25d44354a75f7652c25c37aa9c16b353578ad79`。GPL-3.0源码独立置于忽略缓存，未复制进HTraj模型目录，三个仓库tracked内容均未修改。playground可视化子模块未初始化，训练/推理不需要GUI。

## 原结构与计算路径

输入为目标X_i∈R^(B×8×2)、相对邻居X_/i∈R^(B×N×8×2)，可选类别仅异构数据集使用；输出B×20×12×2。完整模型实测3,149,192参数，与论文补充Full参数相同。

Ŷ_i=Y_linear,i+ΔY_self,i(z_s)+ΔY_re,i(z_r)。LinearDiffEncoding用历史最小二乘线性拟合、平移使最后历史点连续，编码真实运动与线性拟合的Haar谱差。SelfBias用4层encoder-decoder Transformer预测waypoints，再作速度连续插值；社会分支用2层Transformer、邻居中心化运动谱成对乘积及角向汇总。作者配置feature_dim128、8heads、Haar/Haar、8角分区，Kc20风格通道、K=1重复构成20候选。

这是单阶段端到端模型，不是CVAE/DLow；训练目标为原Qpid l2/best-of-K距离，无新增loss。本轮直接调用原ADE_2D损失核心，对完整有效GT和单位目标权重计算；未声称完整LossManager/原train入口已经运行。源码[model.py]({REURL}resonance/model.py)、[社会汇总]({REURL}resonance/_resonance.py)、[差分运动]({REURL}resonance/linearDiffEncoding.py)。GraphConv位于多风格候选解码中，不应将其误认为直接跨行人图传播；显式社会信息来自rc与ReBias。

## 真实工程执行

新stage8环境目录复用既有torch2.5.1+cu124/numpy1.26.4依赖只读路径，没有改动旧环境；作者声明torch2.0.0，故这是兼容性执行，非完全相同环境。Qpid导入要求cwd存在dataset_configs，第一次空目录导入失败；使用独立annotation元数据后正常。未修改网络或源码以掩盖该约束，也未调用原train_or_test入口。

只选官方HOTEL fold的*_val文件，采用其完整20帧目标/上下文，直接转换为原模型输入；这是validation适配器，不是Luna官方loader复现。单人、两人、五人、最大42人窗口均保留所有上下文。每个case前向K20、原距离目标反向、梯度有限；无optimizer更新。邻居存储排列差异最大约1.91e-6，平移还原误差最大约7.63e-6米（同随机噪声）。这些是结构数值测试，不是预测优劣证明。

{table}

随机初始化roundtrip权重strict恢复成功。第一次工程probe使用3_7_11 waypoints，不是发布ZARA2的2_5_8_11；已在记录中区分，未以其宣称作者精确复现。

## 作者权重核查

权重仓库Re分支冻结`42d88ba95e45f441f72318d673c5effd5ef356ec`。预先固定ZARA2文件P8_haar_fixed_epoch129.pt，按其2_5_8_11与其他shape参数严格加载，missing/unexpected keys均空；真实42人validation前向有限，未计算ADE/FDE，原文件SHA未变。该权重的源训练录像可能包含当前validation片段，故此项仅是工程功能，不是泛化成绩。

发布的**rehotel目录是no-linear-base/no-self-bias消融**，args为300epochs、output_pred_steps=all，不可充作Full-HOTEL。ZARA2发布args为200epochs、lr1e-4，论文主文写lr3e-4；不能默认所有fold同超参、所有目录均Full。未读取best_ade_epoch或events数值选权重。

论文说RTX3090、batch1000、200epochs；本轮只测≤42目标的小batch，不能保证原batch1000的完整训练成本。模型结构便于关闭self/re分支，但公平归因必须**分别从头训练**并审计容量，不能仅将推理no_interaction开关当新模型消融。结果目前支持工程可运行，不构成小论文选题GO。
''')
    write('03_MOFLOW_TECHNICAL_AUDIT.md',f'''
# MoFlow技术审查

官方源码`{MF}`，与Stage7.5已审版本相同；MIT许可。这里补齐完整train/eval/data/teacher/student源码核查，没有重复正式训练，没有执行test评价。

## 输入、输出与阶段

ETH原加载器提供B×A×8×6：绝对历史、相对最后观测点历史、相邻位移（不是已除0.4s的m/s），加K=20噪声未来与flow时间。teacher backbone输出B×20×A×24及分数；流采样多步积分，student以一次网络调用生成候选集合。ETHEncoder社交Transformer后还有agent-index位置编码和agent_order_embedding，关系可用性必须结合实际数据A判断。

训练链为：teacher flow matching → 在来源训练池离线生成teacher样本 → IMLE student训练。Y_t=(1−t)Y_0+tY_1；原目标不是普通MSE微调，而是K-shot回归/分类等。IMLE代码将student和teacher候选做双向最近集合距离，按agent聚合，可另加GT监督；这些都已是原论文机制。默认配置teacher150epochs、student150epochs，README teacher样本生成示例100采样步。不能把只跑一个teacher网络forward称为完整一步MoFlow复现。

## 数据和选模实质问题

[ETHDataset]({MFURL}data/dataloader_eth_ucy.py) original分支将traj转换为[:,None,:,:]，因此A=1，虽然保留SocialTransformer类，也没有真实其他行人消息。LED分支为另一套数据，不能推定资格/样本与original相同。

原始转换器使用排序unique frame切20步；要求每目标覆盖完整窗口，且num_peds_considered>min_ped(default1)，排除仅一个合法行人的窗口，然后再将每条轨迹独立存储。原AgentFormer包含单人窗口，二者测试对象不等同。

主训练入口只构建train/test loaders；Trainer令val_loader默认等于test_loader，并按该验证分支的ADE选best。**不能把这个默认入口直接当合法validation选模**。本轮没有调用它。后续若重训必须由外部冻结入口显式传入独立validation；不能混入旧validation区域，不应修改目标对象后仍拿论文表格直接排名。

## 运行、权重与依赖

CPU实例化/teacher backbone人工前向通过。A1 teacher4,700,977参数，student4,086,320参数；A2 teacher4,701,233、student4,086,576。student只实例化计数，未做其forward/IMLE loss。A2同步置换历史、噪声和输出后最大差0.1739883，验证当前固定agent-index编码有顺序敏感性；人工随机权重不说明论文实际指标受损，A1场景更不受这个N2测试直接约束。

第一次人工输入漏past_traj_original_scale字段，报KeyError；补齐作者声明字段后原网络通过，失败现场保留。未修改模型。作者声明Python3.11/torch2.0.1/numpy2.2.4；本轮未安装其完整accelerate/EMA等trainer环境，只运行已记录的兼容CPU组件，不称完整训练入口已通过。

官方HuggingFace元数据可获取，含五折ETH/UCY **teacher** checkpoint及配置；本次返回清单未见对应ETH/UCY IMLE student，不能把NBA student文件当ETH权重。文件OID/LFS SHA保存于moflow_checkpoint_availability.json；未下载和strict加载MoFlow权重，不算性能复现。整个MoFlow GPU测试耗时为0。它在本轮“社会关系”为主的选题中面临A1数据、排列/协议与两阶段成本问题；纯准确性/效率选题也没有新的本地缺陷证据。
''')
    proto_table='|项目|AgentFormer既有官方协议|Resonance原发布|MoFlow原发布|\n|---|---|---|---|\n'
    for r in protocol_rows:proto_table+=f"|{r['item']}|{r['AgentFormer']}|{r['Resonance']}|{r['MoFlow']}|\n"
    write('04_ETH_UCY_PROTOCOL_COMPARISON.md','''
# ETH/UCY协议可比性

标准主目标固定五场景留一、8/12、0.4秒、米、K20，minADE与minFDE各自选最优候选。论文名称相同或时间长度相同不足以构成严格公平比较。

'''+proto_table+'''

## ETH时标与坐标

Luna配置eth=[6,25]、其他录像[10,25]，作者解释ETH视频被加速，6帧间隔用于其物理时长对齐，并提供eth10替代脚本。本轮不擅自裁决哪个时间解释“正确”，也不据此重算测试。关键事实是原始索引间隔/窗口及目标不同，必须核验帧时标和GT身份。AgentFormer维护版ETH还需traj_scale2还原；Luna用自有order/scale1；MoFlow按训练统计min-max再还原/历史朝向旋转。单靠米制不能保证坐标与测试对象一致。

Luna univ13把univ与univ3一起测试，并排除二者训练；其alias到students001/003的逐ID/frame映射未完整核验。AgentFormer明确两条学生序列按目标数汇总。不能只因两个文件就假定完全一致。

## 合法适配方案（未启动训练）

后续若重新立项，可保持AgentFormer已冻结74文件与独立train/val/test清单，先导出稳定(sequence,frame,personID)与GT SHA。Re每个目标用同一C20上下文，输入邻居相对目标最后位置；保留所有合法目标和拥挤窗口，不因单人/高密度删样本。MoFlow若A1仍不引入社会关系，可在同一目标集合训练个体生成参考；如要改为多人，数据资格、固定人数/排列编码和完整训练均需重新审查，不能作为原结构同协议现成社会基线。

所有学习基线必须在该冻结协议从头训练，显式独立validation选模；模型loss/输入语义与作者原发布的差异逐项报告。保留Resonance原协议结果为来源对照，不能把原权重与新协议结果直接排名。CV-Last2/OLS4/OLS8用同目标，明确它们只有一条确定性轨迹，不与K20构成相同候选预算证明。

本轮仅构建了HOTEL-fold validation的小量Re输入适配，688窗口、5203目标窗口，不是训练池/五折loader已全部验证。对应数据可与作者预训练模型来源训练重叠，工程前向不能作泛化评价。旧AgentFormer test已经被观察，此后也不能称其为完全未接触独立确认集；本轮不读取旧测试成绩作选择、不重新计算测试误差。

因此当前无法发布严格可比的三方法成绩排名。存在可解释的适配方案，不等于该方案已经冻结、训练和双评测验证完成。
''')
    write('05_INNOVATION_PROBLEM_SELECTION.md',r'''
# 创新问题筛选：只保留一个待验证草案，当前不立项

## 研究问题A的具体证据

在原Resonance中，邻居中心化后得到运动谱特征h_j；同角分区的位置特征g_j=(distance,angle)与h_j分别平均。论文Eq11对concat求均值；代码位置编码在均值后，二者细节不同，但都没有在池化前绑定位置—运动配对：

\[
R_p=[\frac1{n_p}\sum_{j\in p}h_j,\ \psi(\frac1{n_p}\sum_{j\in p}g_j)].
\]

固定各人的最终位置，并在同一分区交换完整中心化运动历史，h_j的多重集不变，g_j统计不变，因此R_p相同。它不是邻居存储索引置换：后者把同一个人的位置和运动一起换序，应保持预测一致；这里换掉的是两者**配对关系**，物理历史发生变化。

随机权重的汇总差7.45e-9；严格加载作者ZARA2权重后差3.58e-7，均属浮点级别。人工输入、全部历史数组与原代码版本保存。它证明一类不同历史上下文在此表示中不可辨识，不能证明两个真实未来一定不同，更不能将原网络的这种归纳偏置直接称为性能bug。

只按历史固定阈值（8角分区，同区两邻居最终距离差≥0.5m、8帧平均速度向量差≥0.3m/s）统计官方HOTEL-fold validation，5203目标窗口中3869存在这种潜在配对歧义条件。逐来源表与ID全部保存。此比例不是独立行人比例、错误率或因果贡献；未计算这批样本的ADE/FDE差异，没有GT反事实标签。不得因比例高就宣布修复会改善预测。

## 草案A：在原聚合位置保留联合交叉统计（未实现，不获GO）

1. 技术问题：独立均值会丢失“哪个距离/方位上的邻居具有哪种运动谱”。
2. 证据：上述静态恒等关系、原作者权重人工见证、真实历史暴露统计。局限是尚未证明预测损害。
3. 最接近先例：SocialCircle的分量统计、Resonance的谱配对、Reverberation的时谱×角张量、Inter-GraphFormer的相对/有向关系。常规GNN、双线性池化和DeepSets联合特征也能保留绑定。
4. 机制区别草案：不只保留E[h]、E[g]，而保留M_p=E[(g−E[g])(h−E[h])^T]，或其低秩交叉量。该量在某些配对交换时变化，可解释为关联信息，不能保证对任意集合可逆或信息完备。**协方差/双线性表示是已有数学工具，尚未证明这是新算法机制。**
5. 修改位置：仅ResonanceLayer的汇总及其投影；保留个体线性、自偏置和原ReBias解码，不加GNN/新attention。本轮没有写这个模块。
6. 原loss：拟保留原best-of-K距离目标；不引入GT选候选、额外碰撞loss或部署oracle。
7. 成本：全交叉O(Nd_gd_h)，低秩O(Nr(d_g+d_h))，投影仍输出原d；参数和训练时间增幅须实测，不能预填“可忽略”。至少原模型/独立均值容量对照/联合绑定三组同预算训练。原模型约3.15M参数。
8. 对照：同协议Re原版、等容量个体/独立分量MLP、普通联合concat-MLP或边特征聚合、同目标CV；AgentFormer只在完全匹配训练协议后作公开参考。
9. 消融：同容量有/无交叉量，保留/破坏真实配对，分区人数/有邻居分组，跨来源敏感性；分别从头训练，不只推理关闭社会输入。容量与归一化必须一致。
10. 否定条件：结构见证不能复现；所述暴露不对应真实误差问题；普通联合MLP同样解释收益；仅一个来源改善/明显FDE退化；收益达不到事前固定门槛；创新查重发现等价方法。任何一种都停止，不加复杂模块挽救。

门槛草案可预设为validation来源内target-window与person-ID等权ADE均至少改善max(0.005m,1%)、FDE不超过同阈值退化，且至少两个不重叠来源方向一致；这不是本轮实际结果或已经批准的训练协议。窗口重叠和长期开发曝光仍须报告。该草案当前在新颖性、预测损害证据和完整公平协议上未通过，不启动实现。

## 研究问题B为什么不凑出第二方案

本轮没有得到MoFlow完整teacher/student的本地配对覆盖—时延数据。原论文已提出K相关候选、回归/分类、IMLE双向集合蒸馏；KDTF也已有个体/交互知识蒸馏。仅减层、减步、加一个候选评分头或把Chamfer换名字不构成机制区别。A1 ETH输入也不能支持“动态社会互动蒸馏”的已有证据。故不提交第二个未经证据支持的网络草案。

SCI三区/四区审稿的主要质疑会是：仅修复特定池化的常规联合特征、没有预测损害证据、增参而非绑定信息导致收益、与动态关系/Rev方法重复、协议/测试曝光差异。当前材料尚不能充分回答这些问题。
''')
    write('06_COMPUTE_FEASIBILITY.md',f'''
# 计算、资源与论文实验成本

RTX3080，nvidia-smi总显存10240MiB，PyTorch当前可见{probe['GPU_total_bytes']}字节。新环境目录与运行工作区独立；公共依赖只读复用，不修改Stage4环境。执行版本与作者pinned版本不同，完整环境分别记录。

原Re随机初始化4个真实validation batch与一次作者权重forward的记录wall总计{gpu_wall:.3f}s（含CPU扫描/设置，不含进程导入；不是纯GPU kernel计时），远低于7200秒预算。MoFlow GPU使用0秒，只做CPU组件检查。不是GPU还有时间就继续训练。

Re最重case为42上下文/42目标，peak allocated约{max(x['allocated_peak'] for x in probe['cases'])/(1<<20):.2f}MiB；reserved及整卡快照详见JSON。整卡读数是时点快照，不是高频峰值监测。单人/多人无OOM、无NaN/Inf、原损失反向有限，checkpoint严格恢复。没有优化器更新、没有完整epoch、没有测作者batch1000，因此不能宣称200epochs完整预算已验证。

## 训练成本的可报告与不可报告部分

Re原论文为一阶段200epochs、batch1000、RTX3090；发布ZARA2 args实际lr1e-4，论文lr3e-4，HOTEL发布的是300epoch消融。未来如重训需要先冻结一套可核验配置，不能根据val成绩在几套作者配置之间切换。

以已有官方HOTEL训练目标29676为**适配后规划参考**，batch128、200epochs约46400次更新。若未来实测每step为0.1/0.5/1s，对应1.29/6.44/12.89h每run；这些step时间是假设而非本轮测量，实际还需validation、IO和checkpoint开销。原batch1000为约6000次更新，但其显存和step时间均未测，不能用42人的小batch直接线性保证。原Luna训练目标数不同，不替换为这一参考数量冒充原论文预算。

一个最低可解释算法比较至少3组×2来源/fold，再做容量/机制消融与多seed，成本成倍放大。单个小batch便宜，不代表一篇论文全部实验成本已经得到准确估计。当前不建议在新颖性尚不明确时支付这笔训练预算。

MoFlow默认teacher150epochs+student150epochs，另需来源训练池teacher离线样本（README100步示例）、存储、teacher权重及EMA。其teacher4.70M/student4.09M的CPU参数量不是训练显存；从未运行其完整loss/backward、采样/IMLE训练，所以不能借用AgentFormer Stage7.4完整epoch时间来估算MoFlow。

结论：Re有限工程运行在本机可行；两者完整论文训练预算未实测。MoFlow完整依赖和原协议匹配尚不足。没有删高密度窗口、缩K20、改网络来制造资源可行性。研究GO未成立，后续大batch或完整epoch工程预检也不自动启动。
''')
    write('07_FINAL_GO_NO_GO.md','''
# 最终判定：NO-GO；当前不选定创新主干

本阶段只考察Resonance与MoFlow，没有第三候选，没有正式训练或test重评。AgentFormer保留为已成功复现的标准公开参考，其连续创新改造路线正式停止。

|条件|判定|理由|
|---|---|---|
|具体可验证问题|部分通过|Re存在可证明的位置—运动配对表示歧义；真实历史有暴露，但预测损害未验证|
|区别于近年工作的机制|未通过|联合交叉统计仍属常规双线性/联合关系表示；Rev已覆盖时谱动态角向建模；Inter-GraphFormer完整方法未核实|
|合法训练/validation|方案可设计，未完成冻结执行|两原入口都有val/test混用路径，需独立协议适配|
|公平基线/消融|未通过准入|数据、ETH采样、上下文/单人资格不一致；尚未形成完整同目标重训与双评测链|
|硬件/成本|小batch通过，完整成本未建立|Re≤42目标GPU已跑；MoFlow仅CPU；没有200epoch/150+150预算实测|
|明确停止条件|已给出|草案A有结构、容量、配对、跨来源和新颖性否定条件|

综合A–E评分保存在candidate_scores.csv，是定性审查量表，不是ADE/FDE或SOTA排名，不能靠总分覆盖任何关键门槛。

## 对两个模型的选择与放弃

Resonance工程上更接近社会交互研究：源码完整、单阶段、参数适中、可严格加载一个作者Full checkpoint、具备可复核的表示现象。但当前机制补丁新颖性不足、作者协议不等价且发布HOTEL为消融，故**不推荐它作为已获准的小论文创新主干**。这不是宣称Resonance预测能力失败。

MoFlow有真实公开teacher/student实现，适合研究生成效率；当前ETH original分支却只有单节点，另一版本/节点顺序/val选模和两阶段成本尚需独立处理。没有本地完整teacherstudent误差—效率证据支持新的具体问题，因此也不推荐。不是因为它论文分数高低，也不是因为CPU能跑就选择。

最终唯一选择是：**当前两个都不适合作为已确定的小论文创新主干；不启动创新模型训练。** 不给出第三模型或自动下一阶段。

本轮留下的实质成果是一个具体而非泛泛的可辨识性反例，以及两套公开实现的协议边界。它可以纳入硕士论文的技术分析和可靠基线章节；尚不足以直接命名新算法或承诺SCI投稿。若用户以后决定推进，须先解决新颖性和同协议基线，明确它是新立项授权，不从本轮NO-GO自动转入下一阶段。
''')
    write('STAGE8_REPORT.md',f'''
# Stage8 选题与主干总报告

**最终NO-GO。当前未选定Resonance或MoFlow作为小论文创新主干。** AgentFormer保持已复现公开基线；不再连续改造它，不引入第三模型，不启动正式训练。

Resonance版本`{RE}`，Qpid/Luna子模块分别冻结，MoFlow版本`{MF}`。所有源码/权重/原数据与日志独立保留，旧Stage1–7.6结果不修改。Stage8协议先于工程执行提交`19c19d9`。

实际执行：Re原网络4个真实官方validation batch（N1/2/5/42）K20前向及损失反向、随机权重strict恢复；ZARA2作者Full权重strict加载并做42人validation前向；MoFlow A1/A2原teacher backbone CPU人工前向、teacher/student计数。没有正式epoch、参数更新、MoFlow完整采样/蒸馏或test评价；没有新的ADE/FDE基准成绩。

找到一个明确结构现象：Re在同角分区分别平均中心化运动谱与位置，可能抹除“谁在什么位置以什么运动前进”的绑定。作者权重人工交换见证汇总差约3.58e-7；真实历史5203目标窗口中3869满足预设的潜在歧义条件。该统计不是错误率/独立行人比例，更不是修复收益。仅用联合矩/联合MLP修复还不足以通过新颖性，近年动态交互/Rev等方法已覆盖相关思路。

协议差异不能回避：Re ETH6帧间隔、Luna val=test、univ13和邻居资格不同；MoFlow original变A1且原转换排除单人窗口，main trainer默认val=test；其LED是另一数据版本。原公开成绩不能与AgentFormer直接严格排名。原权重ZARA2与本轮validation也可能训练曝光，功能测试不能称独立泛化验证。

原Re约3.15M参数，小batch资源可行；MoFlowteacher约4.70M/student约4.09M、还有离线样本与两阶段训练。两个新模型的完整训练成本均未实测。由于创新性与公平实验链未通过，没有为填满GPU预算继续跑。

## 报告与证据

1. [近期研究查重](01_RECENT_RESEARCH_REVIEW.md)
2. [Resonance技术审查](02_RESONANCE_TECHNICAL_AUDIT.md)
3. [MoFlow技术审查](03_MOFLOW_TECHNICAL_AUDIT.md)
4. [ETH/UCY协议对比](04_ETH_UCY_PROTOCOL_COMPARISON.md)
5. [创新问题与否定条件](05_INNOVATION_PROBLEM_SELECTION.md)
6. [计算与训练成本](06_COMPUTE_FEASIBILITY.md)
7. [最终NO-GO](07_FINAL_GO_NO_GO.md)

结果均位于`results/tables/stage8/`：文献/协议矩阵、源码及权重SHA、作者args核查、真实工程前向/梯度/资源、validation目标ID与来源SHA、配对歧义见证、失败记录、完整测试与历史保护。

Inter-GraphFormer完整方法仍不可得，明确记录未核实；不从摘要推导它缺少某机制。完整单元测试与最终历史校验数量以unit_tests.xml、historical_preservation.json、delivery_checks.json为准。

完成本地Git提交，不自动push。用户命令：

```bash
git push -u origin feat/stage8-topic-and-backbone-selection
```

完成后停止，等待用户科研决策。
''')
    print('Wrote8 reports, protocol/literature/source evidence, resource and no-go decision')


if __name__=='__main__':main()
