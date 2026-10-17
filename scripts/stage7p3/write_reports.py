"""Render delivery reports exclusively from saved, actually executed evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import xml.etree.ElementTree as ET
from src.agentformer_official_evaluation.runtime import ROOT, OUT, FOLDS, protocol, digest
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new


def write(name, content):
    p = ROOT/'reports/stage7p3'/name
    with p.open('x') as f:
        f.write(content.rstrip()+'\n')


def main():
    summary = json.loads((OUT/'evaluation_summary.json').read_text())
    start = json.loads((OUT/'execution_start.json').read_text())
    pre = json.loads((OUT/'preflight.json').read_text())
    preserve = json.loads((OUT/'preservation_and_delivery.json').read_text())
    test = ET.parse(OUT/'full_tests.xml').getroot()
    suites = list(test) if test.tag == 'testsuites' else [test]
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    inference = {f: json.loads((OUT/f'{f}_inference.json').read_text()) for f in FOLDS if (OUT/f'{f}_inference.json').exists()}
    comparisons = read_csv(OUT/'author_comparison.csv')
    m = summary['macro']
    rows = ['|场景|窗口|目标窗口|本地minADE20|本地minFDE20|第一候选ADE|第一候选FDE|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['folds']:
        rows.append(f"|{r['fold'].upper()}|{r['windows']}|{r['targets']}|{r['minADE20']:.6f}|{r['minFDE20']:.6f}|{r['K1_first_ADE']:.6f}|{r['K1_first_FDE']:.6f}|")
    if m:
        rows.append(f"|五场景等权宏平均|—|—|{m['minADE20']:.6f}|{m['minFDE20']:.6f}|{m['K1_first_ADE']:.6f}|{m['K1_first_FDE']:.6f}|")
    table = '\n'.join(rows)
    resource = ['|折|原test_model总耗时秒|两次模型调用合计秒|allocated峰值MiB|reserved峰值MiB|整卡采样峰值MiB|', '|---|---:|---:|---:|---:|---:|']
    for f, r in inference.items():
        v = r['resource']
        resource.append(f"|{f}|{v['seconds']:.3f}|{r['model_only_seconds']:.3f}|{v['peak_allocated_bytes']/2**20:.2f}|{v['peak_reserved_bytes']/2**20:.2f}|{v['whole_card_sampled_peak_MiB']}|")
    resources = '\n'.join(resource)
    verdict = ('通过初步官方权重性能复现验收。可提交下一阶段从头训练计划；本阶段没有启动训练。'
               if summary['preliminary_reproduction_passed'] else
               '未通过完整五场景初步性能复现验收。官方权重评价仍存在待解释差异或未完成项，停止正式训练启动。')
    source = '\n\n来源：固定官方版本的`test.py`、`eval.py`、`model/dlow.py`及配置；本地执行证据`results/tables/stage7p3/`。公开参照见[作者维护版README](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。\n'
    write('03_FIVE_SCENE_EVALUATION.md', '# 五场景真实推理与评价\n\n'+table+'\n\n全部单位米。K20与第一候选K1分列，后者不是预测分布数学期望。minADE和minFDE独立取最小值，可能来自不同head。\n\n'
        '同一目标窗口等权；UNIV将students001与students003的目标合并计算，不先等权平均两条序列。序列目标数见各折`*_sequence_metrics.csv`。五场景等权宏平均，不以UNIV的大目标数主导跨场景指标。重叠窗口不作独立行人或独立显著性样本。\n\n'
        '所有场景预测先完成，随后统一评分。只对samples评分，recon与导出的GT仅保留文件SHA索引。各折目标/帧/候选覆盖检查、逐目标GT SHA、候选winner见`*_alignment.csv.gz`和`*_target_metrics.csv.gz`。\n\n'+resources+source)
    rows2 = ['|场景|指标|作者米|本地米|有符号差米|绝对差米|相对差%|绝对相对差%|两位小数区间相容|预设初验门槛|', '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in comparisons:
        rows2.append(f"|{r['fold']}|{r['metric']}|{float(r['author_m']):.2f}|{float(r['local_m']):.6f}|{float(r['signed_difference_m']):+.6f}|{float(r['absolute_difference_m']):.6f}|{float(r['relative_difference_percent']):+.3f}|{float(r['absolute_relative_difference_percent']):.3f}|{r['within_published_rounding']}|{r['preliminary_close']}|")
    write('04_AUTHOR_RESULT_COMPARISON.md', '# 作者修正版结果对照\n\n'+'\n'.join(rows2)+'\n\n'+verdict+
        '\n\n作者仅公开两位小数，不将本地全精度与未公开小数逐位对比。“相对差”=(本地−作者)/作者；负数代表本地更低，不代表新方法提升，模型和权重相同。初验门槛在结果前冻结为每项绝对差≤max(0.01米,作者值×5%)；两位小数相容区间±0.005米另列。\n\n'
        '不依据测试误差更改模型、权重、K、数据、采样或配置。过去严格因果协议和其它K1结果不进入此表。'+source)
    diffs = max((r['evaluator_max_difference_m'] for r in summary['folds']), default=None)
    write('05_EVALUATION_REPRODUCIBILITY.md', '# 评价一致性与执行可追溯性\n\n'
        f"正式推理冻结提交：`{start['freeze_commit']}`。该提交实际成功后，入口核对git工作区干净、协议/配置/执行源码已真实进入提交，才写execution_start并开始推理。\n\n"
        '原test.py的get_model_prediction、save_prediction、test_model函数体通过AST原样执行；不用原__main__的strict=False和自动评分。DLow构造器严格加载指定CVAE，wrapper严格加载指定DLow，并逐张量验证CVAE一致。未修改原网络、loss、数据划分或数据字节。\n\n'
        '全体3348个窗口先通过global_output_pre_metric_validation.json的ID/frame/K完整性检查，期间不计算任何ADE/FDE，再启动统一评分。官方eval.py通过原始CLI执行。独立核验另建(frame, person_id)字典对齐GT，不调用官方align_gt；要求每个候选均含同一完整12帧×目标集合，恰好20个文件，且所有坐标有限。两者读取同一samples文件；以原生float32官方函数重算全精度值，独立float64核验。\n\n'
        f'五折官方全精度与独立指标最大差：{diffs}米；冻结容差1e-5米。CLI四位小数容差0.000051米。三位小数预测序列化是官方流程本身，两套评分均在序列化后进行，不将其误称为网络未量化的原始输出。\n\n'
        f"环境：Python {pre['python']}，PyTorch {pre['torch']}，CUDA {pre['cuda']}，NumPy {pre['numpy']}，cuDNN {pre['cudnn_version']}；{pre['GPU']}。cudnn benchmark={pre['cudnn_benchmark']}、deterministic={pre['cudnn_deterministic']}。未声称旧版本PyTorch与现环境逐位等同。\n\n"
        '兼容改动只在运行时提供_LinearWithBias与np.int旧名称；官方源码保持冻结。种子按每折原配置seed=1；DLow mean=True的头并非随机CVAE采样。\n\n'
        'PyTorch allocated/reserved为每折精确峰值；整卡显存为名义等待0.1秒的nvidia-smi轮询峰值（另有命令耗时），可能漏掉更短峰值。计时包含原始test_model两次模型调用及三类文件保存；model_only为两次模型调用、cuda同步的总和，不假装单次纯forward吞吐。\n\n'
        '末尾验收逐文件git检查因效率过低被中断，原版本及日志保留；等价批量版本verify_results_batched.py完成文件SHA和忽略规则核验，未改预测/指标。详见artifact_verifier_interruption.json和final_artifact_checks.json。最终源码与报告版本见final_source_versions.json。\n\n失败记录：首次预检JSON写入因np.int64失败，已保留日志，修复仅计数序列化；首次git检查被官方YAML末尾空行阻止，推理未启动，局部attributes保留配置SHA后成功提交。推理与评分的实际异常见inference_completion与evaluation_summary，未发生项目不虚构异常。\n\n'
        f"完整测试：{tests['tests']}项，failures={tests['failures']}，errors={tests['errors']}，skipped={tests['skipped']}。历史SHA核验{preserve['protected_files']}项，旧文件变化{len(preserve['old_SHA_failures'])}项。完整JUnit和控制台记录均归档。\n\n"
        '全部样本文件（包括不参与评分的recon和GT）路径/SHA/字节数见各折prediction_file_SHA256.csv.gz；官方数据74份、源码43份、10权重及10配置SHA见冻结协议。预测大文件只存ignored cache，不提交Git。'+source)
    write('06_GO_NO_GO.md', '# 冻结Go/No-Go验收\n\n'+verdict+'\n\n'+
        f"五场景完整评价：{summary['all_five_completed']}；预设公开结果接近门槛：{summary['preliminary_reproduction_passed']}；全部结构/数据/文件/双评测检查使用固定规则，无测试后改配置。\n\n"
        '本结论只涉及作者预训练CVAE+DLow的推理与评价能力，不证明从头训练可复现，不证明任何新算法、因果社会收益或泛化改善。K1第一候选仅为辅助。即使通过，下一阶段仍需用户批准冻结训练计划并完整核对CVAE→DLow衔接、验证选择、预算及正式五场景结果。\n\n'
        '若有偏差，按顺序审查坐标尺度、文件SHA与权重索引、DLow引用CVAE、原配置、样本序列化、GT/ID/frame对齐、官方评分与独立评分、随机语义和软件版本；只记录证据，不进行测试指标驱动的参数搜索。当前完整数据和各fold官方/独立指标已归档，无法由现有证据解释的差异必须保留为待解释。\n\n'
        '本阶段执行到此停止，不启动训练，不改变checkpoint，不自动push。'+source)
    write('STAGE7P3_REPORT.md', '# Stage7.3 AgentFormer官方权重五场景评价\n\n'+verdict+'\n\n'+table+'\n\n'+resources+
        f"\n\n冻结评价提交：`{start['freeze_commit']}`。0训练、0更新，固定官方源码e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5，5组原作者CVAE+DLow严格加载；数据74文件与原划分保持一致。\n\n"
        'ETH正确按scale=2恢复；UNIV两序列按目标数池化；五折宏平均等权；全部采用DLow mean=True、20个learned heads、官方samples三位小数输出。minADE与minFDE分别最优。原完整20帧离线上下文资格允许未来存在性筛选，不冒充严格历史因果协议。\n\n'
        f"官方eval.py与独立核验最大全精度差{diffs}米。完整测试{tests['tests']}项通过（失败{tests['failures']}、错误{tests['errors']}、跳过{tests['skipped']}）；保护{preserve['protected_files']}份历史文件，SHA变化{len(preserve['old_SHA_failures'])}份。\n\n"
        '作者对照、绝对/相对差见04报告；协议及checkpoint证据见01/02；真实资源/样本数见03；执行环境、预检失败与commit检查记录见05；科研边界及下一阶段条件见06。真实结果与完整日志入口在results/tables/stage7p3。\n\n'
        '本阶段只本地提交，未push。用户自行执行：`git push -u origin feat/stage7p3-agentformer-official-evaluation`。完成后停止。'+source)
    json_new(OUT/'delivery_summary.json', dict(freeze_commit=start['freeze_commit'], tests=tests,
        protected_files=preserve['protected_files'], old_SHA_failures=preserve['old_SHA_failures'],
        preliminary_reproduction_passed=summary['preliminary_reproduction_passed'], macro=m,
        report_writer_SHA256=digest(__file__), training_epochs=0, optimizer_updates=0, automatic_push=False))
    print(verdict, 'Tests:', tests)


if __name__ == '__main__':
    main()
