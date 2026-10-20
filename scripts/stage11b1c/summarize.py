"""CPU-only aggregation. Never re-runs a model or reads the sealed test pool."""
from .common import *
from .validate import METRICS
import numpy as np

def summarize():
    run=json.loads((OUT/'training_execution.json').read_text())
    validations=[json.loads(p.read_text()) for p in sorted(OUT.glob('validation_epoch*.json'))]
    complete=[r for r in validations if r['status']=='COMPLETE']
    best=min(complete,key=lambda r:(r['window']['minADE20'],r['epoch'])) if complete else None
    rows=[];scenes=[];paired=[]
    for e in [1,5,10,15,20,25,30]:
        found=next((r for r in validations if r['epoch']==e),None)
        if found is None:
            rows.append(dict(epoch=e,status='NOT_RUN',weighting='target_window',targets=0,checkpoint_SHA='',**{k:'' for k in METRICS}))
        elif found['status']!='COMPLETE':
            rows.append(dict(epoch=e,status='INCOMPLETE',weighting='target_window',targets=0,
                checkpoint_SHA=found['checkpoint']['sha256'],**{k:'' for k in METRICS}))
        else:
            for weighting,key in [('target_window','window'),('recording_macro','recording_macro')]:
                rows.append(dict(epoch=e,status='COMPLETE',weighting=weighting,targets=998,
                    checkpoint_SHA=found['checkpoint']['sha256'],**found[key]))
            scenes.extend([dict(model='Y-Net',**r) for r in found['scenes']])
            paired.extend(list(csv.DictReader((OUT/f'paired_epoch{e:03d}.csv').open())))
    cnew(OUT/'validation_metrics.csv',rows)
    if scenes:cnew(OUT/'scene_metrics.csv',scenes)
    if paired:cnew(OUT/'paired_prediction_errors.csv',paired)
    previous=ROOT/'results/tables/stage11b1'
    baseline=[r for r in csv.DictReader((previous/'baseline_comparison.csv').open()) if r['model']!='Y-Net']
    comparison=[]
    fields=['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20']
    for r in baseline:
        comparison.append(dict(model=r['model'],status='COMPLETE_FROZEN',weighting=r['weighting'],targets=998,K=1,
            checkpoint_SHA='',epoch='',**{k:r.get(k,'') for k in fields},unit='original reference pixels'))
    if best:
        for weighting,key in [('target_window','window'),('recording_macro','recording_macro')]:
            comparison.append(dict(model='Y-Net',status='COMPLETE',weighting=weighting,targets=998,K=20,
                checkpoint_SHA=best['checkpoint']['sha256'],epoch=best['epoch'],**{k:best[key][k] for k in fields},unit='original reference pixels'))
    cnew(OUT/'baseline_comparison.csv',comparison)
    scene_comparison=[]
    for r in csv.DictReader((previous/'baseline_scene_metrics.csv').open()):
        scene_comparison.append(dict(model=r['model'],scene=r['scene'],targets=r['targets'],epoch='',checkpoint_SHA='',
            ADE1=r['ADE1'],FDE1=r['FDE1'],minADE20='',minFDE20=''))
    if best:
        scene_comparison.extend([dict(model='Y-Net',scene=r['scene'],targets=r['targets'],epoch=best['epoch'],
            checkpoint_SHA=best['checkpoint']['sha256'],**{k:r[k] for k in ['ADE1','FDE1','minADE20','minFDE20']}) for r in best['scenes']])
    cnew(OUT/'baseline_scene_comparison.csv',scene_comparison)
    gpu=parent_seconds()
    budget=dict(GPU_hard_seconds=28800,safety_seconds=27000,GPU_complete_worker_wall_seconds=gpu,
        processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],
        remaining_hard_seconds=28800-gpu,within_budget=gpu<28800,
        CPU_only_summary_plot_time_charged=False,all_startup_loading_checkpoint_and_GPU_preflight_included=True)
    jnew(OUT/'gpu_budget.json',budget)
    decision=dict(status=run['status'],actual_epochs=run['completed_epochs'],best_epoch=best['epoch'] if best else None,
        best_window=best['window'] if best else None,best_macro=best['recording_macro'] if best else None,
        best_checkpoint=best['checkpoint'] if best else None,
        last_complete_validation=complete[-1] if complete else None,
        protocol='PROTOCOL_DEVELOPMENT_ONLY',GPU_seconds=gpu,
        single_seed=True,test_evaluated=False,innovation_GO=False,automatic_extension=False,
        next_stage='Research decision pending; no automatic training extension or semantic module')
    jnew(OUT/'final_decision.json',decision)
    print('Summary:',run['status'],'epochs',run['completed_epochs'],'best',decision['best_epoch'],'GPU',round(gpu,3),flush=True)
    return run,best,complete

def report(run,best,complete):
    a=json.loads((OUT/'data_audit.json').read_text());decision=json.loads((OUT/'final_decision.json').read_text())
    config=json.loads((CONF/'train.json').read_text());epochs=list(csv.DictReader((OUT/'epochs.csv').open()))
    budget=json.loads((OUT/'gpu_budget.json').read_text());baseline=list(csv.DictReader((OUT/'baseline_comparison.csv').open()))
    table='|模型/预算|窗口ADE1|窗口FDE1|窗口minADE20|窗口minFDE20|\n|---|---:|---:|---:|---:|\n'
    for r in baseline:
        if r['weighting']=='target_window':
            table+='|'+r['model']+f" / K{r['K']}|"+'|'.join(f"{float(r[k]):.6f}" if r[k]!='' else '不适用' for k in ['ADE1','FDE1','minADE20','minFDE20'])+'|\n'
    valtable='|epoch|minADE20|minFDE20|ADE1|FDE1|GoalFDE1|minGoalFDE20|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in complete:valtable+=f"|{v['epoch']}|"+'|'.join(f"{v['window'][k]:.6f}" for k in ['minADE20','minFDE20','ADE1','FDE1','GoalFDE1','minGoalFDE20'])+'|\n'
    scene_table='未产生完整Y-Net验证。'
    convergence='未完成完整验证，无法判断欠收敛。'
    k1='无法判断Y-Net是否超过K1基线。';goal='未运行。'
    if best:
        scene_table='|录像|目标数|ADE1|FDE1|minADE20|minFDE20|\n|---|---:|---:|---:|---:|---:|\n'
        for r in best['scenes']:scene_table+=f"|{r['scene']}|{r['targets']}|"+'|'.join(f"{r[k]:.6f}" for k in ['ADE1','FDE1','minADE20','minFDE20'])+'|\n'
        last=complete[-1];bestm=best['window']
        convergence=(f"最低ADE验证轮为{best['epoch']}，最后完整验证轮{last['epoch']}，minADE20为{last['window']['minADE20']:.6f}；"
            f"相对best变化{last['window']['minADE20']-bestm['minADE20']:+.6f}像素。")
        if best['epoch']==run['completed_epochs']:
            convergence+='最佳点位于训练预算边界，应标记预算截断，不能确认收敛；本阶段不自动延长。'
        else:convergence+='最佳点早于末轮；需同时观察后续验证退化与训练loss，不能仅凭loss下降断言收敛或方法无效。'
        refs=[r for r in baseline if r['weighting']=='target_window' and r['model']!='Y-Net']
        k1='；'.join(f"对{r['model']}，ADE1差{bestm['ADE1']-float(r['ADE1']):+.6f}、FDE1差{bestm['FDE1']-float(r['FDE1']):+.6f}像素" for r in refs)+'（负差为Y-Net改善）。'
        goal=(f"best GoalFDE1={bestm['GoalFDE1']:.6f}，minGoalFDE20={bestm['minGoalFDE20']:.6f}；"
              f"解码路径终点与采样goal差：首候选{bestm['goal_path_gap1']:.6f}、20候选平均{bestm['goal_path_gap_mean20']:.6f}像素。"
              f"候选两两终点距离平均{bestm['endpoint_diversity']:.6f}，轨迹距离平均{bestm['trajectory_diversity']:.6f}；"
              f"终点距离≤1像素的候选对比例{bestm['near_duplicate_endpoint_fraction']:.6f}。这些是几何分散度，不等同真实行为模式数量。"
              f"GT终点10/25/50像素内goal覆盖比例：{best['goal_coverage']}。")
    text=f'''# Stage11B-1C Y-Net冻结开发基线

## 一、配置与协议

基础{BASE}；作者源码{OFFICIAL}，只读复用Stage11B-0训练adapter与Stage11B-1R采样器。HTraj-YNet-SDD-Development-v1 / **PROTOCOL_DEVELOPMENT_ONLY**。8历史/12未来、0.4秒，误差单位原始SDD参考图像像素。25训练录像7496目标，5验证录像998目标；17测试录像2829目标始终封存。

seed17、Adam0.0001、目标batch4/场景batch1、不增强、无AMP/梯度裁剪/调度器。预测核心随机初始化；仅分割器加载作者权重并冻结参数/BN。原始Goal BCE与Path BCE各乘1000，GT waypoint只用于训练teacher forcing。每轮7496目标恰好一次、1882更新，无26scene截断。完整配置、数据及代码冻结SHA见protocol_frozen.json。

## 二、执行

实际{run['completed_epochs']}完整epoch，状态{run['status']}，更新{run['optimizer_updates']}次。所有实际epoch/step/目标身份/损失与耗时记录在epochs.csv、training_steps.csv.gz。初始、每轮边界、best和last指向的不可变checkpoint记录在checkpoint_manifest.json；恢复通过新集成GPU测试验证，不加载旧训练checkpoint。

## 三、曲线

真实曲线位于results/figures/stage11b1c/，只使用真实epoch及固定1/5/10/15/20/25/30验证点；无补齐、平滑或伪造。完整训练与验证原始曲线见CSV。若实际未产生对应数据，不绘制该图。

## 四、固定验证

每次先完成全部998无GT预测与缓存，后单独读取GT评分。CPU独立Generator、原float32阈值权重、10000 proposal、19 KMeans+1 softargmax、20完整轨迹。每轮执行训练后分布的重复/倒序/缓存重算检查；非完整验证不参加best。

{valtable}

## 五、K1与K20

best epoch={decision['best_epoch']}，所有best指标来自同一checkpoint，唯一选择指标为窗口加权minADE20，精确同值取早轮。ADE1/FDE1同第一条完整轨迹，不是分布期望；minADE20和minFDE20独立取最佳完整路径，可为不同head；K20是依赖GT的oracle评价，不能当部署选轨能力。

## 六、同目标CV/OLS

{table}

{k1} K1基线与K20候选预算不同，不能用minADE20低于CV ADE1证明单候选更好。基线逐目标history/GT与结果SHA已核验。

## 七、录像差异

{scene_table}

录像宏平均与窗口加权均保存；目标数分别423/443/24/45/63，不能混用两种均值。同背景/重叠时间窗口不能当独立行人或未知场景确认。

## 八、真实案例与目标/路径差异

9例按冻结minADE20低.05/.10/.15、高.85/.90/.95、minGoalFDE20高.85/.90/.95顺序统计量选取，sample_id稳定打破排序同值；跨组重合会在qualitative_cases.json披露。完整RGB/历史/GT/20路径/20目标/oracle轨迹保留，坐标CSV可复核，不做语义碰撞判断。

{goal}

## 九、收敛

{convergence} 单seed开发结果不能证明跨种子稳定。未计算训练ADE，不能以训练BCE替代泛化误差，也没有独立测试确认。

## 十、资源

本轮GPU进程总墙钟{budget['GPU_complete_worker_wall_seconds']:.3f}秒（{budget['GPU_complete_worker_wall_seconds']/3600:.3f}小时），包括启动、模型加载、GPU前检、训练、验证和checkpoint。CPU整理/绘图另行执行。硬预算28800秒、安全阈值27000秒，未通过减少K/目标/proposal节省预算。每轮显存峰值与训练耗时见epochs.csv；每次验证耗时与预测manifest见独立JSON。

## 十一、限制

{'; '.join(a['limits'])}。不计算语义合规率，不从RGB树冠/草地直接推断碰撞。开发协议不等于Y-Net官方论文复现，不与ETH/UCY米制结果比较。原始短池每scene/track一段20帧，不能臆造滑窗步长或跨录像自然人身份。

## 十二、建议

先由科研决策审查best/last曲线、K1相对CV差异、goal覆盖与路径解码差异及真实失败案例。若最佳点逼近预算边界，可讨论另行批准延长；若已出现验证退化，应先审查泛化而非延长追分。语义ID和曝光限制未解决，当前最多提出待验证的语义研究假设，不能认定值得开发创新模块。

## 十三、GO/NO-GO

基线工程执行状态为{run['status']}；是否值得追加训练/语义研究交由后续审查。**本阶段创新开发NO-GO，自动延长NO-GO**。未授权下一阶段，不自动训练50/100/300轮，不增加新网络或loss。所有缺失结果保留NOT_RUN，不写0。
'''
    with (ROOT/'reports/stage11b1c/STAGE11B1C_REPORT.md').open('x') as f:f.write(text)

if __name__=='__main__':
    a,b,c=summarize();report(a,b,c)
