"""Post-hoc descriptive reporting only. No selection-rule/model/hyperparameter edits.
All formulas operate on finished paired results; no further GPU/model execution.
"""
from common import *

def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def find(rows,model,field=None,value=None):return next(r for r in rows if r['model']==model and (field is None or r[field]==value))
def append(name,text):
    with (ROOT/'reports/stage9'/name).open('a') as f:f.write(text)

def run():
    overall=read('overall_metrics.csv');groups=read('group_metrics.csv');sources=read('source_metrics.csv');a,b,c=[find(overall,k) for k in 'ABC']
    gain=float(b['window_ADE'])-float(c['window_ADE']);n=int(b['target_windows']);decomposition=[]
    for group in ['ambiguous','non_ambiguous']:
        bg=find(groups,'B','group',group);cg=find(groups,'C','group',group);delta=float(cg['window_ADE'])-float(bg['window_ADE'])
        weighted=int(bg['target_windows'])/n*delta
        decomposition.append(dict(group=group,target_windows=int(bg['target_windows']),C_minus_B_ADE=delta,weighted_contribution=weighted,fraction_of_net_improvement=-weighted/gain))
    assert abs(sum(r['weighted_contribution'] for r in decomposition)+gain)<1e-12
    csv_new(OUT/'ambiguity_gain_decomposition.csv',decomposition)
    source_delta=[]
    for source in sorted({r['source'] for r in sources}):
        bg=find(sources,'B','source',source);cg=find(sources,'C','source',source)
        source_delta.append(dict(source=source,target_windows=int(bg['target_windows']),source_persons=int(bg['source_persons']),
            **{f'C_minus_B_{m}':float(cg[m])-float(bg[m]) for m in ['window_ADE','window_FDE','person_ADE','person_FDE']},
            person_FDE_contribution=int(bg['source_persons'])/249*(float(cg['person_FDE'])-float(bg['person_FDE']))))
    assert abs(sum(r['person_FDE_contribution'] for r in source_delta)-(float(c['person_FDE'])-float(b['person_FDE'])))<1e-12
    csv_new(OUT/'source_paired_deltas.csv',source_delta)
    convergence=[]
    for k in 'ABC':
        curve=read(k+'_training_curve.csv');prev=np.mean([float(r['loss']) for r in curve[160:180]]);last=np.mean([float(r['loss']) for r in curve[180:200]])
        convergence.append(dict(model=k,epochs161_180_mean_loss=float(prev),epochs181_200_mean_loss=float(last),relative_change=float((last-prev)/prev),final_loss=float(curve[-1]['loss'])))
    csv_new(OUT/'late_training_loss_diagnostic.csv',convergence)
    charged=used();changed_person_ade=float(c['person_ADE'])-float(b['person_ADE']);changed_person_fde=float(c['person_FDE'])-float(b['person_FDE'])
    ambiguity_gain=-decomposition[0]['C_minus_B_ADE'];non_fraction=decomposition[1]['fraction_of_net_improvement']
    text='\n## 数据支持的最终解释\n\n'
    text+=f"C相对B的窗口ADE只降低{gain:.6f}米（{100*gain/float(b['window_ADE']):.3f}%），未达到预设0.005米门槛；行人等权ADE增加{changed_person_ade:.6f}米，FDE增加{changed_person_fde:.6f}米（{100*changed_person_fde/float(b['person_FDE']):.3f}%）。不能仅按窗口ADE的小幅最优宣布C更好。\n\n"
    text+=f"歧义组3,869个目标窗口中，C的ADE仅比B低{ambiguity_gain:.8f}米，约{ambiguity_gain*1000:.5f}毫米。布尔门槛ambiguity_ADE_improves虽为True，但这不是具有说服力的机制收益。非歧义组贡献约{100*non_fraction:.2f}%的净窗口ADE改善（分区互斥加权分解，非因果归因）。不支持收益特异性集中于配对歧义。\n\n"
    text+='C在5/7来源的窗口ADE较B低，但只有3/7来源的行人等权ADE更低；5/7来源的行人等权FDE更差。来源内及两种计权方向并不一致。biwi_eth的行人等权FDE由B的0.774807增至C的0.972702米，其26/249身份权重明显高于99/5203窗口权重，是汇总方向冲突的重要数值贡献之一；这不是对模型失败成因的因果证明。\n\n'
    text+='普通联合编码B相对A的窗口ADE/FDE也没有改善；行人等权ADE虽小幅更好，却不足以据此把B宣布为可靠改进。当前没有必要将B/C任何一项作为已验证新方法。若只比较结构复杂度可优先B，但本轮不建议以此启动追加训练。\n\n'
    text+='决定：停止本轮显式交叉统计改造，不追加模型、seed、epoch或测试集评价。这个结论没有否认原聚合非单射的数学事实，也不能证明所有位置—运动配对建模都无效；它表示本次具体实现未获得预设的独立预测收益证据。\n'
    append('08_FINAL_RESEARCH_DECISION.md',text)
    append('06_GROUP_AND_SOURCE_ANALYSIS.md','\n## 分组与计权冲突\n\n'+text.split('## 数据支持的最终解释\n\n')[1].split('普通联合编码B')[0]+'\n完整数值分解见ambiguity_gain_decomposition.csv与source_paired_deltas.csv。窗口分解仅针对互斥的歧义/非歧义分区；不能与0/1/多邻居分组重复相加。\n')
    conv='\n## 预算终点与收敛限制\n\n'
    for r in convergence:conv+=f"- {r['model']}：161–180轮平均loss {r['epochs161_180_mean_loss']:.6f}；181–200轮{r['epochs181_200_mean_loss']:.6f}，变化{r['relative_change']*100:.2f}%。\n"
    conv+='\n最后20轮训练损失仍在下降，因此不能宣称完全收敛。验证只在固定最终轮执行，没有逐epoch验证曲线，无法判断验证最优点或早期过拟合；不据此自动延长或选择中间checkpoint。该限制对A/B/C共同存在。\n'
    append('04_ACTUAL_TRAINING_RESULTS.md',conv)
    resource=f"\n## 实际执行预算\n\n三模型正式训练累计9,372.84秒（2.604小时，A含其完成后设备比较异常的墙钟）。GPU任务保守累计计费{charged:.3f}秒（{charged/3600:.4f}小时），包括预检、测速、恢复、最终验证及两次全量回归的整段墙钟，远低于43,200秒。账本可重复核算，不把GPU瞬时利用率当平均利用率；少量明确标记的失败探针/准备阶段采用保守上界。\n"
    append('03_TRAINING_BUDGET.md',resource)
    append('STAGE9_REPORT.md',f"\n实际结论：C相对B窗口ADE只改善{gain:.6f}米，行人等权ADE/FDE分别退化{changed_person_ade:.6f}/{changed_person_fde:.6f}米；歧义组ADE改善仅{ambiguity_gain:.8f}米，净收益主要来自非歧义组。B也没有在两个计权口径下一致超过A。故停止本轮配对改造，不能宣称独特交叉关系收益。\n\n完整单元测试458项通过；87,721项历史SHA检查通过；GPU任务保守累计{charged/3600:.4f}小时。模型参数、目标及GT/采样预算全部匹配，未触及test误差。\n")
    failures=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*failure*.json')) if p.name not in ['engineering_failure_catalogue.json','failure_resolution_summary.json']]
    json_new(OUT/'failure_resolution_summary.json',dict(real_failure_records=failures,original_unit_suite_passed=455,final_unit_suite_passed=458,
        formal_A_completed_before_device_check_error=True,A_updates_repeated=0,all_formal_runs_completed=True,OOM=False,nonfinite_formal_training=False,
        restoration_fix_commit='837da7b',additional_non_experimental_errors=[
            'Report command used system Python3.8 with Python3.10 NumPy binary; corrected environment, no experiment update',
            'Supplemental JSON export initially used numpy.int64 IDs; converted to native int, no model or data change',
            'One poll targeted an already-completed monitor cell; no training-process effect'],
        note='This is a descriptive resolution summary of observed tool/run failures, not newly executed tests. Raw structural logs and A completion/restore traceback retained.'))
    # Presentation rounding only; all CSV/NPZ raw precision remains unchanged.
    import re
    for name in ['04_ACTUAL_TRAINING_RESULTS.md','05_VALIDATION_METRICS.md','06_GROUP_AND_SOURCE_ANALYSIS.md','07_CAPACITY_AND_MECHANISM_AUDIT.md','STAGE9_REPORT.md']:
        p=ROOT/'reports/stage9'/name;s=p.read_text()
        lines=[re.sub(r'(?<![A-Za-z0-9_])([+-]?\d+\.\d{7,})(?![A-Za-z0-9_])',lambda m:f'{float(m.group(1)):.6f}',line) if line.startswith('|') else line for line in s.splitlines()]
        p.write_text('\n'.join(lines)+'\n')
    # CSV line-ending normalization only for append journals; no numeric edits.
    for p in [OUT/'gpu_usage.csv',OUT/'A_epoch_journal.csv']:
        content=p.read_bytes();p.write_bytes(content.replace(b'\r\n',b'\n'))
    print('Descriptive interpretation appended; selectors unchanged; total charged GPU hours',charged/3600)

if __name__=='__main__':run()
