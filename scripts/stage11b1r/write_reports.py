"""Seal the engineering decision from actual observations only."""
from .common import *
from .sampler import kmeans_seed
from scripts.stage11b1.evaluation import sample_seed
import importlib.metadata as metadata,platform,math
import numpy as np
import xml.etree.ElementTree as ET

def report(name,text):
    with (ROOT/'reports/stage11b1r'/name).open('x') as f:f.write(text.strip()+'\n')

def main():
    config=CONF/'sampling_protocol.json'
    frozen='b3d9cb5'
    expected=subprocess.check_output(['git','show',frozen+':configs/stage11b1r/sampling_protocol.json'],cwd=ROOT)
    assert hashlib.sha256(expected).hexdigest()==sha(config)
    c=json.loads(config.read_text());root=json.loads((OUT/'minimal_original_case.json').read_text())
    real=json.loads((OUT/'real_checks_results.json').read_text())
    fresh=json.loads((OUT/'fresh_process_results.json').read_text())
    dist=json.loads((OUT/'distribution_results.json').read_text())
    history=json.loads((OUT/'history_after.json').read_text());assert history['passed']
    suites=[]
    for name in ['full_unit_tests.xml','empty_cluster_test.xml']:
        for s in ET.parse(OUT/name).getroot().iter('testsuite'):
            suites.append(dict(file=name,tests=int(s.get('tests')),failures=int(s.get('failures')),
                errors=int(s.get('errors')),seconds=float(s.get('time'))))
    assert not any(s['failures'] or s['errors'] for s in suites)
    count=sum(s['tests'] for s in suites);assert count==561
    processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('parent_process_*.json'))]
    assert len(processes)==4 and all(r['returncode']==0 for r in processes)
    gpu_seconds=sum(r['wall_seconds'] for r in processes)
    test_seconds=sum(float((OUT/p).read_text()) for p in ['full_unit_tests_wall_seconds.txt','empty_cluster_test_wall_seconds.txt'])
    conservative_charge=gpu_seconds+test_seconds+10 # Preflight CPU pytest/import conservative reservation.
    assert conservative_charge<3600
    rows=list(csv.DictReader((OUT/'paired_target_timing.csv').open()));assert len(rows)==20
    means={k:float(np.mean([float(r[k]) for r in rows])) for k in rows[0] if k not in {'sample_id','scene'}}
    p90=float(np.quantile([float(r['new_total_seconds']) for r in rows],.9))
    val_estimate=998*means['new_total_seconds']+5*means['cache_seconds']
    old_estimate=998*means['old_total_seconds']+5*means['cache_seconds']
    overhead=(means['new_total_seconds']/means['old_total_seconds']-1)*100
    original_rows=[r for r in root['records'] if r['component']=='original_GPU_multinomial']
    rng_states={json.dumps(r['RNG_before'],sort_keys=True) for r in original_rows}
    assert len(rng_states)==1 and len({r['probability_SHA'] for r in original_rows})==1
    assert root['fixed_proposal_KMeans_bitwise'] and root['fixed_goal_decoder_bitwise']
    go=bool(real['all_bitwise'] and fresh['passed'] and dist['passed'] and conservative_charge<3600)
    checks=list(csv.DictReader((OUT/'real_K20_checks.csv').open()))
    assert len(checks)==92 and all(r['bitwise_vs_first']=='True' for r in checks)
    budget=dict(GPU_hard_seconds=3600,GPU_safety_stop_seconds=3300,GPU_worker_process_seconds=gpu_seconds,
        complete_worker_timing_includes='Python startup,model loading,all synthetic/real sampler tests,checkpoint save/load,hashing',
        processes=processes,full_and_extra_CPU_test_wall_seconds=test_seconds,
        CPU_preflight_extra_conservative_seconds=10,conservative_budget_charge_seconds=conservative_charge,
        budget_pass=conservative_charge<3600,optimizer_updates=0,formal_training_epochs=0,
        mean_new_K20_seconds=means['new_total_seconds'],p90_new_K20_seconds=p90,
        mean_original_GPU_K20_seconds=means['old_total_seconds'],paired_mean_overhead_percent=overhead,
        mean_CPU_sampling_seconds=means['new_sampling_seconds'],mean_KMeans_seconds=means['new_kmeans_seconds'],
        mean_new_TTST_seconds=means['new_ttst_seconds'],mean_decode20_seconds=means['new_decode20_seconds'],
        twenty_targets_new_inference_seconds=sum(float(r['new_total_seconds']) for r in rows),
        twenty_targets_cache_seconds=sum(float(r['cache_seconds']) for r in rows),
        validation998_point_estimate_seconds=val_estimate,validation998_with10percent_IO_reserve_seconds=val_estimate*1.1,
        validation998_original_control_point_estimate_seconds=old_estimate,
        peak_allocated_bytes=real['peak_allocated_bytes'],peak_reserved_bytes=real['peak_reserved_bytes'],
        GPU_peak_scope='Shared worker with92 new forward calls and20 original controls; no training',
        card_memory_snapshot=real['whole_card_state'],card_snapshot_is_continuous_peak=False,
        CPU_occupancy_cores=real['CPU_occupancy_cores'],host_peak_RSS_kib=real['peak_host_RSS_kib'],
        limitations='20 fixed train targets,untrained random core; later heatmaps affect KMeans iterations; estimates are not measured validation runtime')
    jnew(OUT/'budget_audit.json',budget)
    jnew(OUT/'unit_test_summary.json',dict(passed=True,tests=count,suites=suites,
         original_GPU_control_nonrepeatability='Expected observed baseline failure,retained; not a failed repaired-sampler unit test'))
    identifiers=list(csv.DictReader((ROOT/'results/tables/stage11b1/train_target_manifest.csv').open()))
    collisions={}
    for label,fn in [('proposal',sample_seed),('kmeans',kmeans_seed)]:
        grouped={}
        for r in identifiers:grouped.setdefault(fn(r['sample_id']),[]).append(r['sample_id'])
        collisions[label]=[dict(seed=s,ids=ids) for s,ids in grouped.items() if len(ids)>1]
    jnew(OUT/'seed_identity_audit.json',dict(train_targets=len(identifiers),seed_bits=32,
        collision_groups=collisions,collision_free_for_audited_pool=not any(collisions.values()),
        rule=c['proposal_seed'],kmeans_rule=c['kmeans_seed'],
        scope='Finite audited train pool; SHA truncation does not mathematically guarantee no future collisions'))
    manifest=json.loads((ROOT/'results/tables/stage11ap/frozen_source_manifest.json').read_text())
    for r in manifest['files']:
        assert sha(ROOT/'experiments/stage11b0/cache/official'/r['path'])==r['sha256']
    jnew(OUT/'source_and_data_evidence.json',dict(base_commit=BASE,
        protocol_freeze_commit=subprocess.check_output(['git','rev-parse',frozen],text=True,cwd=ROOT).strip(),
        implementation_commit='2085d7bf581903850634025bbc378b686928008b',
        fresh_restore_check_commit='81467e69c1185bb794661fa398c26f6c19bbbcf4',
        official_commit=OFFICIAL,official_repository=manifest['repository'],license='MIT; inherited verified author source',
        official_source_files_SHA_verified=len(manifest['files']),protocol_SHA=sha(config),
        original_sampling_SHA=sha(ROOT/'experiments/stage11b0/cache/official/ynet/utils/image_utils.py'),
        original_KMeans_SHA=sha(ROOT/'experiments/stage11b0/cache/official/ynet/utils/kmeans.py'),
        stage11b1_failure_logs_unchanged=True,selected20_manifest_SHA=c['target_manifest_SHA'],
        train_artifact=json.loads((ROOT/'results/tables/stage11b1/data_audit.json').read_text())['artifacts']['train'],
        history_files=history['files'],history_unchanged=True,old_prediction_weights_loaded=False,
        segmentation_SHA='75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194',
        Python=platform.python_version(),packages={p:metadata.version(p) for p in ['torch','torchvision','numpy','pandas','scipy','segmentation-models-pytorch','pytest']},
        test_loaded=False,validation_loaded=False,new_predictive_ADE_FDE=False))
    decision=dict(decision='SAMPLER_ENGINEERING_GO' if go else 'NO_GO',
        formal_training_authorized_by_this_stage=False,formal_training_started=False,
        repaired_sampler_bitwise_on_tested_stack=real['all_bitwise'],order_invariance=True,
        future_GT_counterfactual=True,cache_recompute=True,fresh_process=fresh['passed'],
        raw_pixel_restoration=fresh['raw_pixel_restoration_independently_verified'],distribution_sanity=dist['passed'],
        CPU_GPU_sequences_identical_required=False,CPU_GPU_bitwise_claim=False,
        no_bitwise_claim_for_other_hardware_or_software=True,unit_tests=count,history_unchanged=True,
        GPU_process_seconds=gpu_seconds,conservative_charge_seconds=conservative_charge,
        no_new_prediction_performance_result=True,next_action='Stop; subsequent training requires separate instruction/protocol freeze')
    jnew(OUT/'final_decision.json',decision)
    raw_max=max(r['max_coordinate_diff'] for r in original_rows)
    old_goal_max=max(r.get('max_goal_diff',0) for r in root['records'])
    table='|概率图|CPU最大频率偏差|GPU最大频率偏差|CPU-GPU最大差|通过|\n|---|---:|---:|---:|---|\n'
    for r in dist['maps']:
        table+=f"|{r['map']}|{r['CPU_max_abs_frequency_error']:.8f}|{r['GPU_max_abs_frequency_error']:.8f}|{r['CPU_GPU_max_abs_frequency_difference']:.8f}|{r['passed']}|\n"
    report('01_ROOT_CAUSE_AND_RANDOM_PROTOCOL.md',f'''# 根因与随机协议

基础提交{BASE}；原Y-Net源码{OFFICIAL}，MIT。旧evaluation.py/precheck.py/sampling_audit.py只读，Stage11B-1失败原件保留。先完成独立最小复现，再实现修复。原始函数未编辑、未monkeypatch。

固定第一训练目标`{root['sample_id']}`，捕获概率图SHA `{root['probability_SHA']}`，5次原GPU抽样前Python/NumPy/TorchCPU/CUDA RNG哈希均相同。概率图逐位一致，proposal仍不同，最大采样坐标差{raw_max:.6f}缩放像素。旧完整K20目标点最大差{old_goal_max:.12f}原像素。固定proposal后KMeans5次逐位一致，固定goals后解码5次逐位一致。最小案例张量与逐项SHA/RNG状态见minimal_original_case.json及其本地cache工件。

因此证据将本次不一致定位至原CUDA categorical/multinomial抽样路径，而不是忘记重置NumPy或解码网络随机变化。Stage11B-1已记录Torch2.5 deterministic flag拒绝CUDA cumsum；本轮没有做低层kernel跟踪，不能把某个CUDA指令根因称为已被独立证明。

KMeans仍有正常随机性：NumPy choice选择19个初始中心，空簇时CPU torch.randint补点。固定输入/状态时稳定；新包装为其建立独立、可恢复的随机作用域，不修改原KMeans算法。额外全重合proposal测试强制18个空簇，5次结果一致且调用者状态恢复。

proposal种子为uint32大端SHA256前4字节，字符串`HTraj-YNet-SDD-Development-v1:17:`+sample_id；KMeans用`HTraj-YNet-K20-v2:kmeans:`+sample_id。禁止内置hash。每目标新建独立CPU Generator；KMeans作用域只设置所需NumPy/CPU状态并用fork_rng保存恢复CUDA状态。Python状态不触碰。92次真实修复调用均验证调用前后所有RNG状态相同。固定池32位种子碰撞统计在seed_identity_audit.json，不将有限审计写成一般无碰撞保证。

限制：作用域锁只保护本适配器之间的串行调用；不能保证不遵守该锁的外部线程不会同时使用全局NumPy RNG。当前检查采用单进程串行。若改为并行，需要进程隔离或重新验证。
''')
    report('02_SAMPLING_SEMANTICS_AND_DISTRIBUTION.md',f'''# 新旧采样语义与分布

保持原float32、sigmoid/temperature1、严格小于max×0.01的相对阈值置零、原设备上的全图sum归一化；归一化后原位精度权重复制CPU，不重复归一化、不转换double再抽样。torch.multinomial仍replacement=True、10000 proposal。索引仍按x=index%W，y=floor(index/W)还原，并回到模型设备。保留19原KMeans中心+1 softargmax目标、20条完整12步轨迹，CWS关闭。模型结构、全部参数、预测核心、损失和作者源文件未改变。

CPU和CUDA使用不同PRNG/实现，**不要求两者相同序列**。新的KMeans独立随机域也与旧全局流不同；这是明确授权的随机协议修复，不是旧GPU数值序列的逐位复制。CPU原函数与新adapter在同CPU种子下的采样序列、阈值边界和索引映射单元测试逐位相同。

统计方法于提交{frozen}预注册：4图×2设备×16空间格，50次独立稳定SHA种子、每次10000抽样，每图/设备500000，总计4000000。以原GPU float32阈值后权重的16格概率为期望；仅分析时转float64并归一，采样输入不变。联合Hoeffding/union-bound家族α0.001，epsilon={dist['epsilon']:.12f}，每设备对期望最大频率偏差≤epsilon，两设备差≤2epsilon；非零支持外样本必须0，权重逐位相同。没有看到结果后调整方法、次数或阈值。

{table}

全部通过。分布比较是统计合理性检查，不是CPU/GPU序列bitwise一致，也不是对任意概率图或所有细粒度类别的经验检验定理。对当前定义保持同一类别概率的主要依据是原始权重逐位保留、相同multinomial定义和索引映射；16格统计只是独立经验核查。
''')
    report('03_REPEATABILITY_AND_RESOURCE_RESULTS.md',f'''# 重复性与资源结果

固定真实概率图CPU抽样5次proposal SHA相同；3个真实训练目标各5次完整K20一致；20个目标倒序执行一致；20个目标只替换未来12帧GT后一致；20个场景重算分割与缓存一致。合计92次完整修复前向，proposal/goal/path逐位一致，最大坐标差0。另一个Python解释器严格恢复同一随机初始化工程checkpoint，对3个目标逆序推理，其概率图、权重、proposal、goal、path SHA全部一致。首目标点和首完整路径另用原始神经函数独立确认÷0.25恢复原像素坐标。所有输出20×1×12×2和目标20×1×2有限。

全量回归560通过，加空簇随机回退补充1项，共{count}项通过。原GPU不重复是保留的对照现象，并非修复后单元测试失败。所有失败/警告保留，包含旧式分割checkpoint SourceChangeWarning；没有宣称与原Torch1.5跨版本bitwise一致。

相同20训练目标、相同随机固定权重：新完整K20平均{means['new_total_seconds']:.6f}秒，P90 {p90:.6f}秒，首目标{float(rows[0]['new_total_seconds']):.6f}秒；旧GPU对照平均{means['old_total_seconds']:.6f}秒。整体均值变化{overhead:+.3f}%。CPU抽样平均{means['new_sampling_seconds']*1000:.3f}ms；原KMeans平均{means['new_kmeans_seconds']:.6f}秒；TTST合计{means['new_ttst_seconds']:.6f}秒；20路径解码{means['new_decode20_seconds']:.6f}秒。20目标推理合计{budget['twenty_targets_new_inference_seconds']:.6f}秒，缓存处理{budget['twenty_targets_cache_seconds']:.6f}秒。

CPU/GPU随机序列及KMeans初始化域不同，会改变聚类迭代耗时。整体差异不能全部归因CPU复制开销，也不是对训练后热图的速度保证。

同进程92新前向+20旧对照峰值allocated {real['peak_allocated_bytes']/2**20:.3f}MiB、reserved {real['peak_reserved_bytes']/2**20:.3f}MiB；整卡结束快照{real['whole_card_state']}，未连续测整卡峰值。CPU平均{real['CPU_occupancy_cores']:.3f}个核，host RSS峰值{real['peak_host_RSS_kib']/1024:.1f}MiB。模型总参数{real['parameters']}，预测核心{real['trainable_parameters']}，分割器冻结，所有权重/缓冲前后SHA相同，0优化器更新。

998目标点估{val_estimate:.3f}秒（{val_estimate/60:.3f}分钟）；加10%IO/日志余量{val_estimate*1.1/60:.3f}分钟。估算来自20训练目标，不是实测验证耗时。GPU worker完整进程总{gpu_seconds:.6f}秒，包含启动、检查、checkpoint保存恢复与hash；加完整CPU测试{test_seconds:.2f}秒及预检10秒保守项，共{conservative_charge:.6f}秒，低于新增1小时预算。没有评价验证/测试ADE/FDE。
''')
    report('STAGE11B1R_REPORT.md',f'''# Stage11B-1R Y-Net K20随机采样复现性修复

**结论：{decision['decision']}。仅采样工程准入，不自动启动30epoch，不代表预测性能、独立泛化或算法创新通过。**

分支feat/stage11b1r-sampling-reproducibility；基础{BASE}。协议与统计阈值训练前/观察前提交{frozen}冻结，官方Y-Net {OFFICIAL}与所有Stage1—11B-1文件保持。{history['files']}项历史SHA通过，旧失败日志保留。

## 根因

独立最小复现固定概率图和四类RNG状态，原GPU proposal仍不同；固定proposal的KMeans及固定goal的decoder各5次逐位相同。旧完整K20 goal最大差{old_goal_max:.12f}原像素。本次问题定位在CUDA multinomial路径，非遗漏随机种子；具体低层kernel未单独跟踪。

## 修复

独立HTraj adapter保持原设备float32阈值和归一化权重，仅CPU本地Generator执行10000次有放回抽样，然后还原索引回传模型。19原KMeans中心+softargmax与20完整轨迹不变。KMeans NumPy choice/CPU空簇补点使用独立SHA随机域并恢复调用者状态。原源码、结构、权重和损失未改。CPU/GPU序列不要求相同；新adapter内部必须逐位重复。

## 真实检查

固定图5次；3目标各5次完整调用；20目标倒序、未来GT替换、场景缓存/重算，合计92次新K20输出；独立解释器3目标严格恢复/预测。所有proposal/goal/path SHA一致、最大坐标差0，形状及有限性通过；原像素还原独立验证。{count}项单元测试通过，含强制18空簇随机回退测试。没有放宽任何bitwise判据。

分布核查预注册4图×50次×10000抽样×2设备，总400万；16格同时频率界epsilon={dist['epsilon']:.12f}，全部通过；原阈值权重逐位保留、无支持外采样。不将统计近似称为CPU/GPU bitwise等价。

## 成本

20目标新完整K20平均{means['new_total_seconds']:.6f}秒，旧对照{means['old_total_seconds']:.6f}秒，均值增加{overhead:.3f}%；CPU抽样约{means['new_sampling_seconds']*1000:.3f}ms。998目标估计{val_estimate/60:.3f}分钟，加10%余量{val_estimate*1.1/60:.3f}分钟。候选与聚类初始化不同会改变KMeans耗时，不能把全部变化归因CPU或外推为所有checkpoint速度。

峰值allocated/reserved {real['peak_allocated_bytes']/2**20:.3f}/{real['peak_reserved_bytes']/2**20:.3f}MiB（新旧控制共同进程）。GPU worker总{gpu_seconds:.6f}秒，含CPU测试的保守计费{conservative_charge:.6f}秒，预算3600秒内。模型总参数{real['parameters']}、预测核心{real['trainable_parameters']}，全部参数和缓冲不变，0训练更新。

## 证据与限制

configs/stage11b1r/sampling_protocol.json；results/tables/stage11b1r/minimal_original_case.json、fixed_probability_five_repeats.csv、real_K20_checks.csv、fresh_process_checks.csv、distribution_trials.csv、distribution_bins.csv、paired_target_timing.csv、GPU进程账单、pytest XML、历史SHA与checkpoint/工件SHA。原始案例和随机工程checkpoint仅在忽略cache保存，不上传权重/数据。

逐位结论限当前RTX3080/PyTorch/NumPy与串行执行，不保证跨硬件、驱动、库版本。16格Monte Carlo不是对全部类别的穷尽证明。32位SHA种子有限池碰撞情况单独记录。未经协调的外部线程仍可能干扰原KMeans全局RNG，应使用串行或进程隔离。

仍为HTraj-YNet-SDD-Development-v1 / PROTOCOL_DEVELOPMENT_ONLY；原语义ID映射、分割权重训练清单、新验证潜在语义曝光及短池来源限制不因采样修复消失。所有工程前向使用训练目标与随机预测核心；未计算任何新的验证/测试ADE/FDE，不能判断Y-Net是否超过CV或是否欠收敛。

本轮完成后停止。后续有限训练需新的明确指令，并将本修复后的随机协议纳入训练/评价冻结配置；不自动恢复旧Stage11B-1训练入口、不增加epoch或任何新网络模块。
''')
    print(decision['decision'],'unit tests',count,'GPU worker seconds',round(gpu_seconds,3),flush=True)

if __name__=='__main__':main()
