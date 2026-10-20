"""Report a failed evaluator admission honestly; never start a training loop."""
from .common import *
from .admission import training_permitted
import math, platform, xml.etree.ElementTree as ET
import numpy as np

def empty_csv(path, fields):
    with path.open('x', newline='') as f:
        csv.writer(f,lineterminator='\n').writerow(fields)

def report(name, text):
    with (ROOT/'reports/stage11b1'/name).open('x') as f:f.write(text.strip()+'\n')

def main():
    pre=json.loads((OUT/'precheck_execution.json').read_text())
    data=json.loads((OUT/'data_audit.json').read_text())
    history=json.loads((OUT/'history_after.json').read_text()); assert history['passed']
    source_freeze='58c5a7211f2fe7a12785fe1ee26ab844af43ca6c'
    cfg=json.loads((CONF/'train.json').read_text())
    for p in ['train.json','evaluation.json']:
        frozen=subprocess.check_output(['git','show',source_freeze+':configs/stage11b1/'+p],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==sha(CONF/p),p
    suites=[]
    for name in ['full_unit_tests.xml','remaining_legacy_unit_tests.xml']:
        root=ET.parse(OUT/name).getroot()
        for s in root.iter('testsuite'):
            suites.append(dict(file=name,tests=int(s.get('tests')),failures=int(s.get('failures')),
                               errors=int(s.get('errors')),seconds=float(s.get('time'))))
    assert not any(s['errors'] or s['failures'] for s in suites)
    test_count=sum(s['tests'] for s in suites)
    timings=list(csv.DictReader((OUT/'precheck_target_timing.csv').open()))
    per_target=np.array([float(r['total_seconds']) for r in timings])
    old=json.loads((ROOT/'results/tables/stage11b0/budget_estimate.json').read_text())
    # Timing is independent of validation GT/errors. Later learned maps can
    # change KMeans iteration counts; this is an estimate, not a guarantee.
    train30=30*old['observed_full_pass_seconds']
    val998=998*float(per_target.mean())+5*max(float(r['cache_load_segment_seconds']) for r in timings)
    point=train30+7*val998+600
    planning_high=train30*1.3+7*(998*float(per_target.max())*1.5)+600
    # First2 attempts have inner process timers; successful timing and repeat
    # audit also have /usr/bin/time whole-process wall. One failed diagnostic
    # startup was not individually timed. Do not invent an exact total.
    first=json.loads((OUT/'initial_precheck_execution.json').read_text())['wall_seconds']
    second=json.loads((OUT/'compat_precheck_execution.json').read_text())['wall_seconds']
    measured_lower_bound=first+second+18.82+7.04
    t0=int(subprocess.check_output(['git','show','-s','--format=%ct',source_freeze],cwd=ROOT,text=True))
    conservative_upper=time.time()-t0+60
    budget=dict(status='TIMING_FEASIBLE_BUT_EVALUATOR_NO_GO',GPU_hard_budget_seconds=28800,safety_stop_seconds=27000,
        measured_GPU_process_wall_lower_bound_seconds=measured_lower_bound,
        exact_GPU_process_sum_available=False,
        unmeasured_overhead='Python startup in first2 attempts and one failed diagnostic startup; final tests CPU-only',
        conservative_GPU_charge_upper_bound_seconds=conservative_upper,
        upper_bound_rule='Entire wall since preflight freeze commit plus60s; includes idle/code/CPU time, intentionally overcharges GPU',
        prior_stage_training_seconds_per_complete_epoch=old['observed_full_pass_seconds'],
        training_epoch_measurement_source='Stage11B-0 real25-recording7496-target one-pass; NOT new training',
        new_K20_single_target_seconds=pre['first_single_target_seconds'],
        new_K20_twenty_targets_including_scene_cache_seconds=pre['twenty_target_total_seconds'],
        new_K20_mean_target_seconds=float(per_target.mean()),new_K20_p90_target_seconds=float(np.quantile(per_target,.9)),
        new_K20_mean_TTST_KMeans_seconds=float(np.mean([float(r['ttst_kmeans_seconds']) for r in timings])),
        new_K20_mean_decode20_seconds=float(np.mean([float(r['decode20_seconds']) for r in timings])),
        validation998_point_estimate_seconds=val998,training30_point_estimate_seconds=train30,
        seven_validations_point_estimate_seconds=7*val998,checkpoint_and_tests_reserve_seconds=600,
        total_point_estimate_seconds=point+conservative_upper,
        planning_high_seconds=planning_high+conservative_upper,
        planning_high_assumptions='Training1.3x previous pass; each validation target1.5x maximum measured target; not a confidence bound',
        thirty_epochs_plus_seven_validations_fit=point+conservative_upper<28800,
        peak_K20_allocated_bytes=pre['peak_allocated_bytes'],peak_K20_reserved_bytes=pre['peak_reserved_bytes'],
        prior_stage_training_peak_allocated_bytes=old['allocated_peak_bytes'],
        no_formal_training_started=True)
    allowed=training_permitted(data['passed'],True,pre,budget);assert not allowed
    jnew(OUT/'budget_audit.json',budget)
    jnew(OUT/'unit_test_summary.json',dict(passed=True,total_tests=test_count,suites=suites,
         separate_real_data_evaluator_gate='FAILED',previous_unrelated_plugin_failure_retained=True))
    jnew(OUT/'protocol_frozen.json',dict(name=cfg['protocol'],status='PROTOCOL_DEVELOPMENT_ONLY',
         base_commit=BASE,configuration_freeze_commit=source_freeze,
         sampler_compatibility_commit='9266d0a1b867f3400e52281aa47c81699bac813e',
         official_commit=OFFICIAL,train_config_SHA=sha(CONF/'train.json'),evaluation_config_SHA=sha(CONF/'evaluation.json'),
         config_unchanged_since_freeze=True,data_audit_SHA=sha(OUT/'data_audit.json'),
         source_manifest_SHA=data['source_manifest_SHA'],artifacts=data['artifacts'],
         test_set_loaded=False,protocol_limitations=data['limitations'],training_admitted=False))
    jnew(OUT/'training_execution.json',dict(status='NOT_STARTED_EVALUATOR_REPRODUCIBILITY_GATE_FAILED',
         completed_epochs=0,optimizer_steps=0,training_target_visits=0,
         validation_YNet_targets_predicted=0,best_epoch=None,last_epoch=None,checkpoints=[],
         YNet_ADE1=None,YNet_FDE1=None,YNet_minADE20=None,YNet_minFDE20=None,
         no_author_predictor_checkpoint=True,no_previous_training_weights_loaded=True,
         train_configuration_modified_from_validation=False,formal_training_admitted=False,
         reason='Same history/probability/seed yielded different original CUDA proposals/goals; exact repeat/order gates failed'))
    empty_csv(OUT/'epochs.csv',['epoch','global_step','train_loss','goal_loss','path_loss','learning_rate','epoch_seconds','gpu_memory','checkpoint_SHA'])
    cnew(OUT/'validation_metrics.csv',[dict(epoch=e,status='NOT_RUN_EVALUATOR_GATE_FAILED',targets_expected=998,targets_evaluated=0,
          ADE1='',FDE1='',minADE20='',minFDE20='',GoalFDE1='',minGoalFDE20='',checkpoint_SHA='') for e in cfg['validation_epochs']])
    valrows=list(csv.DictReader((OUT/'validation_target_manifest.csv').open()))
    cnew(OUT/'scene_metrics.csv',[dict(scene=s,status='YNET_NOT_RUN',targets_expected=sum(r['scene']==s for r in valrows),
          targets_evaluated=0,ADE1='',FDE1='',minADE20='',minFDE20='') for s in sorted({r['scene'] for r in valrows})])
    (OUT/'qualitative_cases').mkdir(exist_ok=True)
    (OUT/'qualitative_cases/NOT_RUN.md').write_text('Y-Net正式训练与验证未启动，因此没有真实Y-Net验证案例图。不得将随机训练目标前向或CV轨迹伪装为训练后验证结果。选择分位数规则仍保留于evaluation.json。\n')
    rows=list(csv.DictReader((OUT/'baseline_comparison.csv').open()))
    table='|模型|窗口ADE1|窗口FDE1|录像宏平均ADE1|录像宏平均FDE1|\n|---|---:|---:|---:|---:|\n'
    for model in ['CV-Last2','OLS4','OLS8']:
        a=next(r for r in rows if r['model']==model and r['weighting']=='target_window')
        b=next(r for r in rows if r['model']==model and r['weighting']=='recording_macro')
        table+=f"|{model}|{float(a['ADE1']):.6f}|{float(a['FDE1']):.6f}|{float(b['ADE1']):.6f}|{float(b['FDE1']):.6f}|\n"
    report('02_K20_EVALUATOR_AND_FAILURE_ANALYSIS.md',f'''# K20评价器准入失败分析

正式模型训练 **NO-GO**。人工公式/形状/像素还原/输入身份测试通过，但真实K20重复性硬门槛没有通过。{test_count}项单元测试全部通过，不等于真实GPU准入通过。

第一次预检在官方CUDA multinomial内部cumsum处遇到Torch2.5严格确定性拒绝。保留`initial_precheck_execution.json`和压缩原stdout；只在原始采样调用周围临时关闭deterministic-algorithms，调用后恢复，不改官方源码、参数、K20、proposal或分布配置。兼容提交9266d0a没有正式训练。

第二次在未来GT反事实输出相同断言失败，保留`compat_precheck_execution.json`与原stdout。不能将此简单解释为GT泄漏：随后固定**完全同一**历史，5次同sample_id/同种子调用，概率图SHA始终相同，但proposal SHA和goal SHA出现两种值，目标点最大差0.009033203125原像素；随机未训练模型的路径输出恰好相同。真实源数据history投影与预测API不含未来参数。证据见`repeat_probe.json`和`sampling_audit.py`。差异定位到官方CUDA随机采样路径；具体低层kernel根因未另做内核跟踪。

最终20训练目标只完成耗时审计，没有评分。一次缓存/重算配对和一次未来GT替换配对恰好逐位相同；倒序调用仍失败，5次同输入审计也失败。不能用一次偶然相同覆盖既有反例，不能认定未来反事实或缓存等价性已得到稳定验证。

未换CPU采样、未换随机算法、未减少proposal、未事后放宽容差。CPU采样可能在数学分布层面相同，但固定种子的具体候选会变化；尚未通过用户要求的等价性核验，因此本轮不引入。下一轮如授权，应先单独冻结可复现采样实现及数值等价/随机协议判据，再重做准入，不能直接训练。

未发现预测API访问未来GT的路径；但整体GPU反事实输出验证受随机不稳定影响，不能宣称所有隔离与复现门槛通过。随机模型没有验证ADE/FDE，不能判断预测能力或失效类型。
''')
    report('03_BUDGET_AND_EXECUTION.md',f'''# 实测预算与执行状态

真实训练目标完整TTST10000→K20：单目标{pre['first_single_target_seconds']:.6f}秒；20目标含20次图像读取/分割缓存{pre['twenty_target_total_seconds']:.6f}秒；目标平均{per_target.mean():.6f}秒、P90 {np.quantile(per_target,.9):.6f}秒。20目标来自20个训练录像，按预定词法顺序选择，与GT误差无关。没有使用验证误差选择设置。

TTST/KMeans平均{budget['new_K20_mean_TTST_KMeans_seconds']:.6f}秒，20路径解码平均{budget['new_K20_mean_decode20_seconds']:.6f}秒。完整测时进程内部{pre['wall_seconds']:.6f}秒，整进程18.82秒，CPU占用约{pre['cpu_occupancy_process_cores']:.3f}个核。峰值allocated {pre['peak_allocated_bytes']/2**20:.2f}MiB，reserved {pre['peak_reserved_bytes']/2**20:.2f}MiB；整卡读取为{pre['GPU_card_state']}。不是训练显存。

用Stage11B-0实测完整epoch {old['observed_full_pass_seconds']:.6f}秒，30轮纯训练估计{train30/3600:.3f}小时；998验证目标点估{val998/60:.3f}分钟，7次约{7*val998/3600:.3f}小时。计600秒检查/保存开销，新增主体点估{point/3600:.3f}小时。训练1.3倍、验证1.5倍最大实测单目标的规划高情形（未加本轮保守收费）约{planning_high/3600:.3f}小时，不是统计置信区间；后续训练热图会改变KMeans迭代，不能保证常数耗时。

预算点估可行，**准入仍NO-GO**：重复性失败不能被预算通过覆盖。正式训练0epoch、0更新；没有initial/best/last新训练权重，没有checkpoint恢复实验，不得拿旧Stage11B-0训练结果冒充本轮。

本轮GPU相关进程已有计时的合计下界{measured_lower_bound:.6f}秒；两个早期启动开销及一次失败诊断启动没有分别完整计时，不能报告伪精确总数。以预检冻结提交至本次汇总整段墙钟+60秒作为保守GPU收费上界{conservative_upper:.3f}秒，包含CPU、代码整理及空闲时间，明显过计但小于8小时。详见budget_audit.json。最终交付前可再用同一规则更新上界，不追加GPU工作。
''')
    report('04_BASELINE_RESULTS_AND_UNEXECUTED_ANALYSIS.md',f'''# 同目标解析基线与未执行项目

998个固定开发目标，三个合法K1基线均先生成全部路径，再读取GT评分。所有history/GT SHA与冻结清单一致。CV-Last2用末两帧速度；OLS4/8用真实0.4秒时间坐标拟合历史速度，均锚定最后历史位置。不是使用拟合截距替代最后观测。

单位：原始参考图像像素。

{table}

窗口加权不同于录像等权；bookstore_2/ deathCircle_3占423/443目标，其余录像24/45/63，不能混用两种均值。每录像完整结果在baseline_scene_metrics.csv；2994条模型-目标配对误差在baseline_paired_errors.csv，35928条预测点在baseline_predictions.csv.gz。测试集未进入任何计算。

Y-Net ADE1/FDE1、minADE20/minFDE20与GoalFDE **未运行**，CSV留空并标NOT_RUN，未填0。不能判断它是否超过CV/OLS，也不能计算多候选oracle收益。

正式训练曲线、验证曲线、best/last差异、按录像Y-Net误差、低/高/目标误差案例图全部未产生，没有补点或伪造图。qualitative_cases/NOT_RUN.md解释缺失；预先规定的案例分位数规则仍冻结，但没有可用的训练后验证预测供选择。

无法判断欠收敛、验证退化或跨录像稳定性。此前Stage11B-0损失下降只表明原工程流程可训练，不能代替本轮验证性能或说明模型已经收敛。当前反例是采样复现性工程问题，不是Y-Net预测机制无效的证据，更不是论文创新。
''')
    report('05_RESEARCH_DECISION.md','''# 科研判断

**本轮正式训练NO-GO；预算初估可行；创新研究尚无准入证据。**

A. 本轮Y-Net正式训练未启动，不能判断训练稳定或泛化；旧Stage11B-0工程训练证据保留。
B. 30epoch未完成，实际0epoch。
C. minADE20/minFDE20未运行。
D. ADE1/FDE1未运行。
E. 无法判断是否超过CV/OLS；三种解析K1对照已对全部998目标真实执行。
F. 没有Y-Net分录像性能，不能称稳定。
G. 无法判断欠收敛，不能仅凭旧训练loss代替验证。
H. 语义类别编号、分割器确切训练成员、原短轨迹manifest及新开发验证曝光限制仍存在；不能称独立未知场景泛化。
I. 当前不建议追加训练预算；先解决并单独冻结K20随机采样的可复现工程协议。资源不是本次主要阻断项。
J. 没有训练后验证预测，因此没有可支持新算法开发的实际预测失效证据。常规采样兼容修复也不能称为算法创新。

本阶段到此停止。不自动进入Stage11B-2，不训练50/100/300轮，不修改语义模块，不重新读取测试集。
''')
    report('STAGE11B1_REPORT.md',f'''# Stage11B-1 Y-Net SDD有限预算基线研究

**结论：正式训练NO-GO。真实K20评价器的固定种子重复/顺序一致性没有通过，依照用户前置门槛未启动训练。预算点估可行；未生成Y-Net验证成绩。**

分支feat/stage11b1-ynet-baseline-pilot；基础{BASE}；训练/评价配置冻结提交{source_freeze}，至交付未改变。官方源码{OFFICIAL}只读，复用Stage11B-0训练adapter，作者分割权重冻结，预测器随机初始化；未读取作者或旧训练预测权重。协议HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY。

数据核验25训练录像7496目标、5验证录像998目标通过，17测试录像2829目标封存。全部8494来源训练/新验证目标的history/GT/图像身份复核；{history['files']}个历史文件SHA保持。参数53155339，其中冻结分割51513958、预测核心1641381。

## 实际执行

人工指标、像素还原、输入身份和准入门禁单元测试通过；完整回归{test_count}项通过。真实K20训练样本前向覆盖20个训练录像和目标，输出每目标20×12×2路径与20×2目标点，无GT输入，无预测误差评分。最初确定性CUDA调用报错及后续重复性断言失败均保留。相同概率图下proposal/goal反例显示约0.009033像素最大差，不能用随机初始模型路径恰好一致来放行。详见02报告。

实际训练0epoch；优化器0更新；Y-Net验证0目标；best/last/训练checkpoint均未生成。未发生训练恢复，因此恢复项标记未执行，不借用旧结果冒充通过。

## 资源

完整测时目标平均{per_target.mean():.6f}秒，20目标含缓存{pre['twenty_target_total_seconds']:.6f}秒。K20峰值allocated {pre['peak_allocated_bytes']/2**20:.2f}MiB、reserved {pre['peak_reserved_bytes']/2**20:.2f}MiB。GPU流程已有计时下界{measured_lower_bound:.3f}秒，缺少少数启动分项，保守收费上界{conservative_upper:.3f}秒，未接近8小时。30轮训练+7次完整998目标验证及600秒附加项估计{point/3600:.3f}小时（不含已发生预检），训练成本采用上一阶段真实完整epoch，TTST采用本轮训练样本真实测量；不是本轮已执行训练耗时。

## 同目标K1基线

{table}

Y-Net所有ADE/FDE/GoalFDE为未运行，未与K1基线或ETH/UCY米制结果排名。各录像基线与逐目标数据已保存。

## 未产生的材料

训练/验证曲线、best epoch、best/last差异、Y-Net分录像ADE/FDE和9个真实验证案例未产生，原因是前置门槛失败。epochs.csv只有表头；validation_metrics.csv固定7轮标未执行、数值留空；没有虚构图或补齐epoch。无法判断欠收敛、验证退化、跨种子稳定或预测失败模式。

## 协议限制与下一步

SDD数值语义ID/名称未核实、分割checkpoint精确训练清单未公开、短池生成manifest部分缺失、新验证可能已有语义标签曝光。原论文补充4.1训练场景声明不消除当前新切分曝光疑虑。开发实验允许，独立未知场景泛化结论不允许。

先审查可复现采样实现及固定随机协议的数值等价判据，重新通过K20准入后才考虑有限训练；本轮不自动继续。不能以本轮工程失败判断Y-Net方法无效，也不能以解析基线结果提出创新结论。

主要证据：01冻结协议；02评价器失败审计；03预算；04基线与未执行分析；05科研决策；results/tables/stage11b1下data_audit、precheck_execution、repeat_probe、budget_audit、配对CSV、pytest XML、历史SHA与失败原日志。
''')
    print('Reports complete:',test_count,'unit tests PASS; real evaluator gate FAILED;0 training epochs',flush=True)

if __name__=='__main__':main()
