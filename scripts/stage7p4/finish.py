"""Evidence-only reports, bounded estimates, historical preservation and Go/No-Go."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from src.agentformer_feasibility.common import *
from src.socialvae_causal.io import read_csv,csv_new
from src.socialvae_baseline.artifacts import json_new


def load(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None


def write(name,text):
    with (ROOT/'reports/stage7p4'/name).open('x') as f:f.write(text.rstrip()+'\n')


def main():
    begin=time.perf_counter()
    p=protocol();verify_inputs()
    old=json.loads((ROOT/'configs/stage7p4/protected_artifacts.json').read_text())
    failures=[r['path'] for r in old['files'] if not (ROOT/r['path']).is_file() or digest(ROOT/r['path'])!=r['sha256']]
    pred_count=0
    for index in old['prediction_SHA_indices']:
        assert digest(ROOT/index['path'])==index['sha256']
        for row in read_csv(ROOT/index['path']):
            if not (ROOT/row['path']).is_file() or digest(ROOT/row['path'])!=row['sha256']:failures.append(row['path'])
            pred_count+=1
    assert not failures,failures
    xml=ET.parse(OUT/'full_tests.xml').getroot()
    tests={k:sum(int(s.attrib.get(k,0)) for s in xml.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
    json_new(OUT/'historical_preservation.json',dict(historical_files=len(old['files']),stage7p3_prediction_files=pred_count,
        SHA_failures=failures,seconds=time.perf_counter()-begin,official_inputs_unchanged=True,test_metrics_recomputed=False))
    cvae,dlow=load('CVAE_epoch.json'),load('DLow_epoch.json')
    validation=load('validation_summary.json')
    completion=load('execution_completion.json');execution_failure=load('execution_failure.json')
    success=bool(completion and completion['success'] and cvae and cvae['success'] and dlow and dlow['success'])
    budget=dict(complete_epochs_measured=success,estimates_available=False)
    reasons=[]
    if not success:reasons.append('完整epoch、恢复或validation执行未全部通过，见原始失败记录；不启动后续训练。')
    if tests['failures'] or tests['errors']:reasons.append('完整单元测试未通过。')
    if success:
        from scipy.optimize import nnls
        inventory=read_csv(ROOT/'results/tables/stage7p2/official_window_inventory.csv.gz')
        estimates=[]
        fit_info={}
        for stage,row,epochs in [('CVAE',cvae,100),('DLow',dlow,50)]:
            samples=read_csv(OUT/f'{stage}_steps.csv.gz')
            n=np.array([float(s['people']) for s in samples]);a=np.column_stack([np.ones(len(n)),n,n*n])
            y=np.array([float(s['step_seconds']) for s in samples])
            coef,residual=nnls(a,y)
            fit_info[stage]=dict(nonnegative_coefficients=coef.tolist(),RMSE_seconds=float(np.sqrt(np.mean((a@coef-y)**2))),
                measured_people_range=[float(n.min()),float(n.max())],measured_step_sum=float(y.sum()),
                measured_epoch_seconds=row['resource']['seconds'],
                interpretation='Instrumented HOTEL scene-step model, not a trained prediction model; no test ADE/FDE used')
            for fold in ['eth','hotel','univ','zara1','zara2']:
                windows=[r for r in inventory if r['fold']==fold and r['sequence'].endswith('_train')]
                count=np.array([int(r['people']) for r in windows])
                if fold=='eth' and stage=='CVAE':count=np.minimum(count,32)
                x=np.column_stack([np.ones(len(count)),count,count*count])
                outside_valid_steps=max(0.,float(row['resource']['seconds'])-float(y.sum()))
                predicted=float((x@coef).sum())+outside_valid_steps*len(windows)/len(samples)
                if fold=='hotel':predicted=float(row['resource']['seconds'])
                estimates.append(dict(fold=fold,stage=stage,epochs=epochs,train_windows=len(windows),
                    train_epoch_seconds_estimate=predicted,point_train_hours=predicted*epochs/3600,
                    low_train_hours=predicted*epochs*.7/3600,high_train_hours=predicted*epochs*1.5/3600,
                    measured_fold=fold=='hotel',ETH_original_CVAE_cap32=(fold=='eth' and stage=='CVAE')))
        csv_new(OUT/'compute_budget_by_fold.csv',estimates)
        measured_val={r['fold']:r['resource']['seconds'] for r in validation['cases']}
        val_counts={f:p['fold_counts'][f]['val_targets'] for f in p['fold_counts']}
        ref_val=sum(measured_val.values())/sum(val_counts[f] for f in measured_val)
        overhead=[]
        storage=0
        for fold in p['fold_counts']:
            val_seconds=measured_val.get(fold,ref_val*val_counts[fold])
            # Ten proposed validation passes per stage, not part of the original train.py loop.
            val_point=20*val_seconds/3600
            checkpoint_seconds=(10*cvae['checkpoint_save_seconds']+10*dlow['checkpoint_save_seconds'])/3600
            extra_weight_factor=(p['author_pairs'][0]['CVAE_parameters']/cvae['initial_state']['parameter_count'])
            # ETH has a larger original CVAE: allow its checkpoint storage using the known source parameter ratio.
            if fold=='eth':extra_weight_factor=6497922/cvae['initial_state']['parameter_count']
            bytes_estimate=int((10*cvae['checkpoint_bytes']+10*dlow['checkpoint_bytes'])*extra_weight_factor)
            storage+=bytes_estimate
            overhead.append(dict(fold=fold,validation_point_hours=val_point,validation_low_hours=val_point*.5,
                validation_high_hours=val_point*2,checkpoint_save_point_hours=checkpoint_seconds,
                checkpoint_storage_bytes=bytes_estimate,validation_is_assumed_cadence=True))
        csv_new(OUT/'compute_overhead.csv',overhead)
        point=sum(r['point_train_hours'] for r in estimates)+sum(r['validation_point_hours']+r['checkpoint_save_point_hours'] for r in overhead)
        low=sum(r['low_train_hours'] for r in estimates)+sum(r['validation_low_hours']+r['checkpoint_save_point_hours'] for r in overhead)
        high=sum(r['high_train_hours'] for r in estimates)+sum(r['validation_high_hours']+r['checkpoint_save_point_hours'] for r in overhead)
        hotel_high=sum(r['high_train_hours'] for r in estimates if r['fold']=='hotel')+next(r['validation_high_hours']+r['checkpoint_save_point_hours'] for r in overhead if r['fold']=='hotel')
        free=shutil.disk_usage(ROOT).free
        time_ok=high<=336 and hotel_high<=72
        disk_ok=free>=storage+128*1024**2
        if not time_ok:reasons.append('完整训练时间上界超过预设规划门槛。')
        if not disk_ok:reasons.append('当前磁盘不足以保留官方五折全部周期checkpoint及128MiB余量，正式训练启动NO-GO；不得删除历史成果解决。')
        budget.update(estimates_available=True,timing_fit=fit_info,fivefold_point_hours=point,fivefold_low_hours=low,
            fivefold_high_hours=high,HOTEL_high_hours=hotel_high,current_free_bytes=free,
            all_fivefold_periodic_checkpoint_bytes=storage,time_gate_passed=time_ok,storage_gate_passed=disk_ok,
            storage_excludes_validation_outputs_and_failure_checkpoints=True,
            caveats=['Only HOTEL two epochs measured; no fold wall-time equality assumed',
                'Shape-dependent CUDA allocator, CPU parser, thermal/throttling/background load can alter throughput',
                'Only epoch1 observed; training convergence/stability at100/50 epochs unverified',
                'ETH source CVAE original cap32 and additional modules differ; time sensitivity is planning, not confidence interval',
                'DLow measured using epoch1 CVAE; final trained CVAE may alter numerical behavior and loss convergence',
                'Original train.py has no validation loop; proposed validation cadence is overhead scenario only'])
    json_new(OUT/'compute_budget.json',budget)
    decision='GO' if not reasons else 'NO-GO'
    json_new(OUT/'research_decision.json',dict(decision=decision,reasons=reasons,
        full_epoch_and_resume_engineering_passed=success,formal_training_started=False,
        no_test_metrics_recomputed=True,tests=tests,budget=budget))
    # Validation tables separate oracle K20 and deterministic K1 budgets.
    if validation:
        lines=['|validation fold|窗口|目标窗口|minADE20|minFDE20|','|---|---:|---:|---:|---:|']
        cvlines=['|validation fold|确定性模型|ADE|FDE|','|---|---|---:|---:|']
        for r in validation['cases']:
            o=r['overall'];lines.append(f"|{r['fold']}|{r['windows']}|{r['targets']}|{o['minADE20']:.6f}|{o['minFDE20']:.6f}|")
            for name in ['CV-Last2','CV-OLS4','CV-OLS8']:
                cvlines.append(f"|{r['fold']}|{name}|{o[name+'_ADE']:.6f}|{o[name+'_FDE']:.6f}|")
            cvlines.append(f"|{r['fold']}|DLow第一头（非数学期望）|{o['K1_ADE']:.6f}|{o['K1_FDE']:.6f}|")
        validation_text='\n'.join(lines)+'\n\nK1预算单独报告：\n\n'+'\n'.join(cvlines)
    else:validation_text='未完成validation诊断，参见execution_failure.json；不填造指标。'
    write('02_VALIDATION_BASELINES.md','# 官方validation诊断\n\n'+validation_text+
        '\n\n全部为官方validation分析，并非独立测试确认。HOTEL/ZARA2是fold名称，val来自其余官方*_val序列，不等于HOTEL/ZARA2测试录像。两fold的validation存在共享录像/目标，不当作两份独立证据。\n\n'
        '同窗口同目标GT配对，目标窗口加权；重叠窗口不视为独立行人。K20 oracle与CV的K1预算不同，不能用二者差值证明同候选预算优劣。DLow第一头不代表分布期望。\n\n'
        '逐目标、逐时间步和全部预定分组见*_validation_targets.csv.gz、*_validation_timesteps.csv.gz与*_validation_groups.csv。无样本组保留count=0，指标留空。历史速度阈值0.05/0.4/1.8m/s，邻居分组仅使用历史完整8帧、末帧2m以内的人数，完整20帧模型人数另存。两种人数不混用。\n\n'
        '时序误差保留整轨迹ADE最优头、FDE最优头和第一头，前两者仅为事后oracle诊断，不是新候选打分器。终点RMS半径与190对平均距离只用预测，不用GT。原float32完整精度结果未作三位小数文本舍入，不与Stage7.3测试指标混排名。')
    for stage,row,file in [('CVAE',cvae,'03_CVAE_FULL_EPOCH.md'),('DLow',dlow,'04_DLOW_FULL_EPOCH.md')]:
        if row:
            text=json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False)
            body=f'实际执行状态：success={row["success"]}。\n\n```json\n{text}\n```\n\n'
        else:body='未执行。前置阶段或恢复检查未通过；没有补填loss、耗时或指标。\n\n'
        write(file,f'# {stage}完整epoch成本测量\n\n'+body+
            '每次有效窗口的原始loss分项、梯度范数、上下文人数、同步step时间及显存见同名_steps.csv.gz；含None候选的全部数据访问时间见_data_reads.csv.gz。原stdout、逐步JSONL及失败checkpoint存ignored cache，关键日志有版本化副本。\n\n'
            '仅1完整epoch；恢复探针是影子副本下一epoch首次有效窗口的1次工程更新，不计入完整epoch预算，不覆盖模型。不能据此声称100/50轮收敛、性能复现或与作者成绩可比。仪器同步与有限性检查开销已另列。DLow的CVAE参数不在优化器中，原requires_grad/eval行为保留，状态hash用于确认未更新。')
    write('05_COMPUTE_BUDGET.md','# 计算与存储预算\n\n```json\n'+json.dumps(budget,ensure_ascii=False,indent=2)+'\n```\n\n'
        '实测仅HOTEL的CVAE/DLow各一个完整epoch，推断不能替代其它fold的真实训练。细分折数与[1,N,N²]拟合见compute_budget_by_fold.csv；文件若不存在则该估算未执行。0.7—1.5倍训练时间、0.5—2倍validation开销只是预设工程敏感区间，不是统计置信区间。\n\n'
        '官方train.py不在epoch内评价validation，validation频率是未来计划的假设。周期checkpoint数量严格按原配置频率估计，不能删除历史或只保留有利模型来降低成本。存储预算还不含未来完整validation预测和失败现场，属于下界。RTX3080只在HOTEL出现的人数范围内实测，未进行其它fold的正式epoch。\n\n'
        'GO时间门槛是预注册的HOTEL上界72h、五折上界336h规划筛选；实际GPU可用时间与存储准备需下一阶段用户决策。本阶段不自动启动100/50轮。')
    reason_text='\n'.join('- '+r for r in reasons) if reasons else '全部预设工程准入条件通过；只能制定下一阶段计划，不能自动继续训练。'
    write('06_RESEARCH_DECISION.md',f'# Stage7.4决策：{decision}\n\n{reason_text}\n\n'
        f'完整epoch及恢复工程检查通过：{success}。测试：{tests}。\n\n'
        '若仅因存储被判NO-GO，不等于模型训练失败：应分开报告已运行的GPU事实与无法启动正式五折的资源条件。未测得的100/50epoch收敛、性能与其它fold稳定性均不得推断为已验证。禁止用已观察test结果选后续模型或结构，不新增GNN/注意力/评分模块，不重试或延长。')
    write('STAGE7P4_REPORT.md',f'# Stage7.4训练可行性与validation诊断\n\n最终决策：**{decision}**。\n\n'+reason_text+
        f'\n\n完整epoch及恢复工程检查：{success}。官方模型、loss、Adam和调度器保留原样；只测HOTEL两阶段各1轮与各1次独立影子恢复更新。无五场景test重复评价，无新模型。\n\n'+validation_text+
        f'\n\n完整测试：{tests}。保护历史文件{len(old["files"])}份与Stage7.3预测/GT/recon文件{pred_count}份，SHA变化0。\n\n'
        '预算与假设见05报告，完整阶段日志、逐步耗时、loss、checkpoint SHA、状态恢复和分组结果见results/tables/stage7p4。真实失败见execution_failure.json或对应_epoch.json，不存在的运行不填指标。\n\n'
        '只本地提交，不自动push：`git push -u origin feat/stage7p4-agentformer-training-feasibility`。完成后停止。')
    print('FINAL',decision,'engineering epochs/restore:',success,'reasons:',reasons,'tests:',tests,flush=True)


if __name__=='__main__':main()
