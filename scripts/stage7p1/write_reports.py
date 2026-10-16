"""Write evidence-grounded admission reports; no model execution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import xml.etree.ElementTree as ET
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

ROOT = Path('results/tables/stage7p1')
REPORT = Path('reports/stage7p1')


def load(path):
    return json.load(open(ROOT/path))


def write(name, text):
    with (REPORT/name).open('x', encoding='utf-8') as f:
        f.write(text.strip()+'\n')


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |']+
        ['| '+' | '.join(map(str, r))+' |' for r in rows])


def main():
    data = load('data_audit.json')
    setup = load('execution_setup.json')
    execution = load('execution_summary.json')
    real = [json.load(open(p)) for p in sorted((ROOT/'real_cases').glob('*.json'))]
    cf = [json.load(open(p)) for p in sorted((ROOT/'counterfactuals').glob('*.json'))]
    negative = [json.load(open(p)) for p in sorted((ROOT/'negative_probes').glob('*.json'))]
    resource = load('maximum_context_resource.json')
    suites = ET.parse(ROOT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(r.attrib.get(k, 0)) for r in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    context_rows = read_csv(ROOT/'context_windows.csv.gz')
    coverage = []
    for s in data['summaries']:
        rows = [r for r in context_rows if r['fold'] == s['fold'] and r['split'] == s['split'] and int(r['targets']) > 0]
        full = sum(int(r['people']) for r in rows)
        partial = sum(int(r['partial_history_excluded']) for r in rows)
        coverage.append(dict(fold=s['fold'], split=s['split'], windows_with_targets=len(rows),
            eligible_full8_node_windows=full, partial_history_node_windows_excluded=partial,
            complete8_coverage_fraction=full/(full+partial), supervised_target_windows=s['targets']))
    json_new(ROOT/'coverage_summary.json', dict(rows=coverage,
        counting_unit='node-window, only windows with at least one frozen supervised target; overlapping windows are not independent people'))
    source = '官方版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；[源码与配置](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。本阶段模型执行代码版本 `'+setup['code_commit']+'`，逐文件SHA见 `results/tables/stage7p1/execution_setup.json`。'
    summary_table = table(['留出录像', '划分', '支持窗口/有监督窗口', '冻结目标窗口', '混合标签窗口', '最大历史完整人数'],
        [[s['fold'], s['split'], f"{s['windows']}/{s['windows_with_targets']}", s['targets'], s['mixed_label_windows'], s['max_people']] for s in data['summaries']])
    write('02_CAUSAL_CONTEXT_PROTOCOL.md', f'''# 严格历史上下文协议与冻结目标

{source}

**A：HTraj-AgentFormer-A_offline_complete20-v1**仅为离线完整20帧参照，完整未来存在性可以决定其场景节点。本阶段不运行A评价，也不以A替代B。

**B：HTraj-AgentFormer-Causal8x12-v1**是本轮主准入：8帧历史、12帧未来监督、0.4秒、米制。原始四列来源由 `Recording.read`重新解析；上下文 `C=intersection(ids(X_t),t=0..7)`，按ID排序，不设人数或距离上限。不完整历史节点记录并排除，不填补未知位置/速度。这个历史规则限制了可用信息，绝不称为全部历史可见行人均已纳入。

上下文构建只调用历史8帧的字典查询；确定上下文后，标签单独读取。监督集合S严格匹配Stage5.1冻结target CSV；S的历史与12帧GT逐目标SHA均一致。上下文C不读取S或邻居未来存在性。未来缺失坐标在标签审计中为NaN，不能当零坐标标签。

{summary_table}

“混合标签窗口”只计有监督目标且存在未来不完整的上下文节点。HOTEL支持区共149窗口，其中94窗口含冻结目标；这94个全部混合标签。其余窗口仍保留上下文统计，未制造额外监督目标。ZARA2的172窗口中123个混合标签。目标窗口不是独立行人数。

## 数据版本与支持区域

HOTEL manifest SHA256：`{data['protocol']['manifests']['biwi_hotel']['sha256']}`。

ZARA2 manifest SHA256：`{data['protocol']['manifests']['crowds_zara02']['sha256']}`。

五个非ETH原始文件SHA、四份target CSV SHA、完整人数分布见 `data_audit.json`；逐窗口上下文ID、完整未来目标ID、逐人未来有效长度及部分历史排除ID见 `context_windows.csv.gz`。训练与留出录像隔离，旧validation未加入训练，20帧支持区域不改变。ETH轨迹及指标均未读取。两开发录像已被历史实验曝光，不是独立确认集。

## 工程覆盖补充

真实测试需要含监督目标的N1/N2及全标签多人场景。冻结验证支持不足时，仅从另一fold已冻结的原train区域取首个对应人数的窗口作为功能测试，0更新；不改变训练/验证划分。规则在执行前以 `real_case_coverage_addendum.json`和提交 `6d07b7f`冻结。该文件“no causal N1”的准确含义是**无含监督目标的验证N1窗口**；HOTEL仍有14个无监督目标的历史N1窗口，已保留在全支持统计中。

## 模型语义边界

没有新增监督mask、KL mask、未来token mask或输出节点类型。`original_training_packet`遇到C中任意未来缺失即拒绝。推理接口的未来NaN/零有效mask只是满足原 `set_data`结构，完全不进入infer计算，不是可用训练标签。原始loss只在C全部有真实12帧GT时执行，未改loss数学含义。
''')
    cf_table = table(['录像', '原始数据变更', 'K1最大预测差/m', 'K20最大预测差/m', '历史/资格/scene_orig/关系mask'],
        [[r['recording'], r['variant'], r['K1_prediction_max_difference'], r['K20_prediction_max_difference'], '全部一致' if r['passed'] else 'FAIL'] for r in cf])
    negative_table = table(['录像/开始帧', 'C/S人数', '监督q均值最大变化', '对未知邻居未来梯度L1（补值1）', 'NaN loss有限'],
        [[f"{r['recording']}/{r['start_frame']}", f"{r['people']}/{r['supervised_targets']}", f"{r['supervised_posterior_mu_max_difference']:.9f}",
          f"{r['completions'][0]['supervised_q_gradient_wrt_missing_neighbor_future_L1']:.9f}", r['completions'][2]['loss_finite']] for r in negative])
    write('03_COUNTERFACTUAL_LEAKAGE_TESTS.md', f'''# 原始数据级未来反事实与训练负例

{source}

从两个真实混合标签窗口取原始四列20帧记录，写入本阶段独立cache文件，再由 `Recording.read`重读后重新构建上下文；没有只改预处理tensor。原始来源文件保持不动。五类变更分别是：全部未来位置变化、邻居未来删除、邻居未来缩短、添加仅未来出现行人、修改监督目标之外不完整未来邻居的坐标。HOTEL首个混合窗口邻居仅有1帧未来，缩短至0与删除相同；两项结果不视为独立证据。ZARA2邻居2帧缩短至1帧。

每项检查原始历史行、历史完整ID、部分历史ID、历史位置SHA、模型pre_motion/pre_vel、scene_orig、agent_mask及历史context_enc一致。K1采用learned prior的零epsilon，K20采用稳定录像/窗口/行人ID产生的同组epsilon，未将潜变量固定成绕过learned prior的常量。K1不称为预测分布数学期望。

{cf_table}

10/10通过，两个K下所有最大预测差均为0。真实修改行数、原始反事实文件路径/SHA和逐项布尔值见 `counterfactuals/*.json`。原始数据副本保存在被忽略的 `experiments/stage7p1/cache/raw_counterfactuals/`，不上传轨迹。

## 不能用监督mask修复原始训练后验

以下是**故意无效的补值负例**，不是真实监督训练。在相同真实历史与完整监督GT下，所有未来不完整邻居的监督mask全置0，只将其未知未来填为末位置或末位置+100m。固定随机状态并关闭dropout比较目标q；还计算目标q对未知邻居补值的梯度。未更新权重，未以该loss汇报预测能力。

{negative_table}

两窗口监督目标q都改变且存在直接梯度；sample loss在两种有限补值间相同，因为它走历史先验，但MSE随训练后验改变。KL在这些随机权重例中均被min_clip=2截住，**本例没有实测KL数值变化**；未mask的KL依赖由原loss源码和自动化公式测试确认，不能据此编造数值变化。NaN哨兵使后验、原loss非有限；“NaN乘零”并不屏蔽未知标签。

这证明当前冻结实现的mask-only接法不正确，不证明所有可能的因果CVAE都不能实现。训练后验可合法使用完整监督目标GT；本轮失败点是未知邻居未来参与后验且KL没有对应目标语义。没有编写替代后验、KL或loss来强行通过。
''')
    batch_table = table(['录像/开始帧', '历史完整N', 'GT完整目标', '原loss', '梯度L2', '原始训练路径', '严格恢复'],
        [[f"{r['recording']}/{r['start_frame']}", r['people'], r['targets'],
          '未定义/未运行' if r.get('true_supervised_loss') is None else f"{r['true_supervised_loss']:.6f}",
          '未运行' if r.get('gradient_L2') is None else f"{r['gradient_L2']:.6f}",
          r['original_loss_status'], r['strict_checkpoint_restore']] for r in real])
    write('04_REAL_DATA_FORWARD_BACKWARD.md', f'''# 真实数据前向、原loss、反向与恢复

{source}

共10个预设工程场景，覆盖真实N1/2/3/6/9/13/15、部分历史邻居排除、历史完整未来不完整/消失邻居及两个验证区域的最大上下文。窗口列表和来自train/validation的来源、选择理由保存在 `execution_setup.json`；均处于Stage5.1冻结支持，不做训练或模型选择。训练与验证含义没有互换。

{batch_table}

7个全标签因果场景执行官方 `set_data → forward → compute_loss → backward`；保留原训练模式、随机场景旋转、后验、MSE+KL+sample(K20)，loss与梯度有限。每例5,972,098个参数具有loss梯度张量，梯度非零总范数见表；这不是每个标量每次梯度都非零的声明。无optimizer构建或更新、无完整epoch、未载预训练。

3个混合标签真实场景：历史先验推理正常，**原真实监督loss/反向不执行**，原因在 `UnlabeledContextError`中保存；不是以零loss代替。HOTEL验证的15人高密度窗口8人GT完整、7人未来不足，未删邻居或困难目标。

10例均保存随机权重checkpoint并 `strict=True`加载到同结构新实例；恢复K1与原预测逐值相同。checkpoint路径/SHA保存在 `real_cases/*.json`，权重本体仅留本机被忽略cache。不涉及训练后optimizer状态恢复，因为本轮0更新。

## 训练入口兼容性

真实执行的是官方模型单batch训练调用链，不是官方 `train.py`完整epoch入口。原入口仍使用依赖完整未来筛选的DataGenerator/loader，**未通过严格因果训练准入**。没有运行官方CLI开始epoch，也未通过改官方preprocessor/np.int等源码掩盖问题。Stage7的运行时 `_LinearWithBias=Linear`导入别名沿用；官方文件不改。原始模型及loss接口可执行不等于混合标签训练可执行。

## 测试与失败记录

全量测试 {tests['tests']} passed / {tests['failures']} failed / {tests['errors']} errors / {tests['skipped']} skipped；见 `full_tests.xml/txt`。10项新增契约测试先通过，验证只查询历史、真实标签、NaN缺失、fail-closed、ID固定噪声、原始记录重读、原loss分母/KL数学行为。

首次pytest启动被本机ROS自动插件的缺失lark阻断，保留 `initial_test_launcher_failure.txt`；后续仅设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`隔离外部插件，不安装或更改模型环境。2个故意NaN补值负例、3个原训练拒绝完整保留；不能把“单测全部通过”解释为严格因果训练GO。
''')
    res_table = table(['录像/帧(N)', '执行范围', 'allocated峰值/MiB', 'reserved峰值/MiB', '整卡采样峰值/MiB', '时间/s', 'OOM'],
        [[f"{r['recording']}/{r['start_frame']}({r['people']})", 'K1+严格恢复'+('+原K20 loss/反向' if r.get('true_supervised_loss') is not None else ''),
          f"{r['resource']['peak_allocated_bytes']/2**20:.2f}", f"{r['resource']['peak_reserved_bytes']/2**20:.2f}",
          r['resource']['whole_card_sampled_peak_MiB'], f"{r['resource']['seconds']:.4f}", r['resource']['OOM']] for r in real])
    rr = resource['resource']
    cov_table = table(['划分', '仅有监督窗口的full8节点窗口', '部分历史排除节点窗口', 'full8覆盖率'],
        [[r['fold']+'/'+r['split'], r['eligible_full8_node_windows'], r['partial_history_node_windows_excluded'], f"{r['complete8_coverage_fraction']*100:.2f}%"] for r in coverage])
    write('05_RESOURCE_AND_CAPACITY_AUDIT.md', f'''# 真实上下文资源与参数审计

{source}

GPU：{setup['GPU']}，{setup['torch_total_memory_bytes']/2**30:.2f} GiB。执行环境 `experiments/stage4/venv/bin/python` 加 `experiments/stage7/cache/deps`；torch {setup['torch']}、numpy {setup['numpy']}。完整pip依赖快照在 `execution_setup.json`。没有新增可学习层，原参数量 {setup['model_state']['parameter_count']:,}；保留官方Hotel pre-CVAE配置，两录像共用，不含DLow阶段。

{res_table}

计时包括对应场景推理、可定义时的原loss/反向、随机checkpoint保存/读取及恢复推理；**不是纯训练吞吐**。allocated/reserved是PyTorch峰值；整卡显存为nvidia-smi每约0.1秒采样最大值，可能漏过短峰，包含驱动/其他进程；不冒充精确整卡峰值。

## 最大上下文

冻结来源支持最大N={resource['people']}，`{resource['recording']}:{resource['start_frame']}`，53个完整12帧GT目标，20个未来不完整节点；比Stage7人工N57多16人。未设人数上限、未裁剪。K20 learned-prior推理实际执行 {rr['seconds']:.4f}s，无OOM；allocated峰值 {rr['peak_allocated_bytes']/2**20:.2f} MiB，reserved峰值 {rr['peak_reserved_bytes']/2**20:.2f} MiB，整卡采样峰值 {rr['whole_card_sampled_peak_MiB']} MiB。

该73人窗口来自已冻结训练来源，只做资源检查，不读取预测误差。因混合标签原训练语义不成立，**73人真实loss反向/optimizer训练显存没有验证**；不以K20推理无OOM外推训练可行。显存保留量已接近10GB卡容量，后续训练可行性仍有风险，但本阶段不据此删邻居或改结构。

## 历史完整上下文的信息损失

{cov_table}

节点窗口分母为历史内至少一帧可见节点（含目标），仅计有冻结监督目标的窗口；不是独立行人覆盖率。完整支持区域的人数直方图与无监督窗口另保留在 `data_audit.json`。部分历史行人按预先历史规则排除，不伪造历史，不按未来缺失排除上下文；本轮不承诺对这类邻居充分利用。
''')
    gates = [
        dict(id=1, condition='原始来源/manifest/监督目标及GT SHA一致', status='PASS'),
        dict(id=2, condition='原始数据未来反事实10项全部通过', status='PASS'),
        dict(id=3, condition='真实单人/多人完整标签原loss反向及推理', status='PASS_ON_FULLY_LABELED_CASES_ONLY'),
        dict(id=4, condition='历史完整、未来不完整邻居的原语义训练处理', status='FAIL'),
        dict(id=5, condition='原始训练loss/梯度对全部因果场景有限且有定义', status='FAIL_FOR_MIXED_LABEL_CONTEXT'),
        dict(id=6, condition='原始网络/loss/参数无未经声明修改', status='PASS_UNMODIFIED_BUT_NO_MIXED_LABEL_TRAINING_SOLUTION'),
        dict(id=7, condition='真实batch资源可运行', status='PARTIAL_K20_INFERENCE_AND_SMALL_FULL_LABEL_BACKWARD_ONLY'),
        dict(id=8, condition='checkpoint严格恢复', status='PASS_10_RANDOM_STATE_CHECKPOINTS')]
    json_new(ROOT/'go_no_go.json', dict(decision='NO-GO', strict_causal_inference_checks=True,
        strict_causal_original_training_admitted=False, gates=gates,
        formal_epochs=0, optimizer_updates=0, ETH_evaluated=False,
        original_network_or_loss_changed=False, future_training_automatically_authorized=False,
        conclusion='Stop: missing-neighbor futures require explicit posterior/KL/target-semantics adaptation beyond the frozen data-only admission scope.'))
    gate_table = table(['条件', '证据/状态'], [[r['condition'], r['status']] for r in gates])
    write('06_GO_NO_GO.md', f'''# 最终准入：NO-GO

{gate_table}

严格历史数据适配与先验推理隔离成立；**冻结AgentFormer原始联合训练在本协议B下不成立**。主要阻断是未来不完整上下文的后验token、KL及目标归一化语义，非发现先验偷读未来坐标。HOTEL全部94个有目标验证窗口均出现该条件，不能用“少数边界样本”略过。

真实负例在监督mask为0时仍观测到监督目标q对未知邻居未来补值的依赖，两录像均成立。NaN失败、拒绝执行与完整标签成功分开记录。没有将缺失GT当零标签，没有以过滤邻居、丢弃困难目标、退回完整20帧或改loss强行GO。

如果未来考虑仅监督目标后验查询、缺失future token屏蔽、KL/归一化重定义或context-only节点潜变量等处理，均需新的模型语义审计和用户阶段授权；本轮**未实施或验证**这些方案，不推荐直接正式训练。也没有重选其他公开模型。

保留Stage7全部结构、learned-prior、排列一致性、GT数值隔离和N57证据。本轮只补足原始数据/真实batch检查，不把此前人工通过重写成真实因果训练成功。无正式训练、无optimizer更新、无ETH评价、无新创新模块；所有随机权重输出仅验证工程性质，未报告ADE/FDE或论文复现成绩。

完成报告、历史SHA核验、全量测试和本地提交后停止。后续科研决策交给用户。
''')
    write('STAGE7P1_REPORT.md', f'''# Stage7.1 AgentFormer严格因果准入总报告

**结论：NO-GO，不进入正式训练。** 严格8帧历史上下文与原始数据级先验隔离检查通过，但未能在保留冻结原始后验/KL/loss语义时处理只有历史、未来不完整的邻居。

## 真实执行结果

- 官方源码 `{setup['official_commit']}`，5,972,098参数；审计先提交 `e0ae133`，执行配置及代码先提交 `{setup['code_commit']}`。官方文件未修改。
- 两fold全部Stage5.1训练/验证目标与历史/GT SHA一致；HOTEL验证318目标窗口、ZARA2验证1092目标窗口。上下文只取完整历史8帧，不按未来资格筛选。
- 原始四列数据反事实10/10通过；未来坐标、存在性、长度、未来新增行人和非监督未来变化均不改变历史context、scene_orig、关系mask、K1及固定噪声K20预测，最大预测差0。
- 10个真实工程场景均通过历史先验推理与随机状态strict checkpoint恢复；7个全标签场景通过原始真实loss及反向；3个混合标签场景明确拒绝原训练入口，未伪造监督。
- 两个混合标签补值负例确认mask-only方案不正确，监督q最大变化分别 {negative[0]['supervised_posterior_mu_max_difference']:.6f}、{negative[1]['supervised_posterior_mu_max_difference']:.6f}（各录像明细见03报告），NaN补值loss非有限。
- 最大历史完整上下文73人，超过旧N57；K20推理无OOM，整卡采样峰值 {rr['whole_card_sampled_peak_MiB']} MiB。73人真实原训练显存未确立，不能据推理成功宣称训练可行。
- 完整单元测试 {tests['tests']} passed，0 failed；首次ROS插件启动失败、预期拒绝和NaN负例均保留。

## 核心审计答案

历史context encoder、scene_orig、agent_mask、learned prior及自回归推理不需要邻居未来。FutureEncoder却将所有节点未来token联合attention，未使用fut_mask；KL对所有节点平均且min_clip，重建/sample按全部上下文N归一化。原实现没有仅历史context节点的训练语义。完整GT目标可合法进入训练后验，但未知邻居未来不能作为正常标签。

改loader可以构建严格因果推理输入；要正确训练混合标签场景则必须进一步声明和实现后验、监督集合、KL/输出等语义适配。用户本轮不允许通过核心改动强行准入，因此停止。没有把本阶段工程适配当成原论文复现或创新。

## 交付与边界

7份报告位于 `reports/stage7p1/`；原始JSON、逐窗口人数/ID、来源SHA、loss/梯度、checkpoint SHA、资源、测试和失败日志位于 `results/tables/stage7p1/`。原始反事实副本与随机权重仅在被忽略的 `experiments/stage7p1/cache/`；不上传数据/权重。历史保护清单 `configs/stage7p1/protected_artifacts.json`，最终核验见 `history_and_source_verification.json` 与 `delivery_verification.json`。

本阶段0完整epoch、0优化器更新、0ETH测试；不含正式预测性能，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p1-agentformer-causal-admission
```

执行完成后停止，等待下一步科研决策。
''')
    print('Wrote six new reports plus existing initial audit; decision NO-GO, tests', tests, flush=True)


if __name__ == '__main__':
    main()
