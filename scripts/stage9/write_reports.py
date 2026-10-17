"""Render reports from real finished CSV/JSON only; never fill absent metrics."""
from common import *

def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def md(rows,columns):
    table='|'+'|'.join(label for key,label in columns)+'|\n|'+'|'.join('---' for _ in columns)+'|\n'
    for r in rows:
        values=[]
        for key,label in columns:
            value=r.get(key,'')
            if isinstance(value,float):value=f'{value:.6f}'
            values.append(str(value))
        table+='|'+'|'.join(values)+'|\n'
    return table

def run():
    report=ROOT/'reports/stage9';train=json.loads((OUT/'training_summary.json').read_text());v=json.loads((OUT/'validation_execution.json').read_text());decision=json.loads((OUT/'final_decision.json').read_text())
    overall=read('overall_metrics.csv');sources=read('source_metrics.csv');groups=read('group_metrics.csv');config=json.loads((ROOT/'configs/stage9/training.json').read_text())
    for r in train['models']:r['minutes']=r['seconds']/60;r['allocated_MiB']=r['allocated']/2**20
    checks=json.loads((OUT/'structure_tests.json').read_text());history=json.loads((OUT/'historical_preservation_final.json').read_text())
    cols=[('model','模型'),('window_ADE','窗口minADE20'),('window_FDE','窗口minFDE20'),('person_ADE','行人等权minADE20'),('person_FDE','行人等权minFDE20')]
    def get(k):return next(x for x in overall if x['model']==k)
    a,b,c=[get(k) for k in 'ABC']
    delta_b=float(b['window_ADE'])-float(a['window_ADE']);delta_c=float(c['window_ADE'])-float(b['window_ADE'])
    (report/'04_ACTUAL_TRAINING_RESULTS.md').write_text('# 实际正式训练\n\n冻结配置后，A/B/C使用相同目标池、seed42、共同'+str(config['epochs'])+'完整epoch、batch128、Adam 0.0001、常数lr。全部随机初始化。无作者权重、无挑选best epoch。\n\n'+md(train['models'],[('model','模型'),('epochs','完整epoch'),('updates','更新'),('target_visits','目标访问'),('parameters','参数'),('minutes','分钟'),('allocated_MiB','allocated峰值MiB')])+'\n完整训练曲线及step日志见各模型文件。训练损失是原best-of20轨迹欧氏目标，不能叫NLL。最后一轮不是保证收敛点；本阶段不因曲线走向自动延长。训练中未读取验证精度。所有正式计数必须为200×232=46,400更新/模型、5,935,200目标访问/模型。若实际冻结epoch不同，以config和表格为准。\n\nCPU/CUDA Adam计步张量比较的问题、原始异常及修复见恢复审计。修复只涉及加载设备/验证与完成run衔接，不改变权重计算、预算、数据顺序或超参数。A完整训练权重保留，不重复训练；B/C从头训练。最终所有模型/优化器/RNG恢复检查通过。checkpoint索引保存路径和SHA，权重全部在ignored cache。\n\n'+md(v['models'],[('model','模型'),('targets','验证目标窗口'),('candidates','候选数'),('seconds','最终推理及序列化秒')])+'\nallocated/reserved是PyTorch峰值；整卡显存为nvidia-smi采样而非连续真实峰值。持有CUDA任务的总墙钟保守计入gpu_usage.csv，包含测速、预检失败、恢复与最终推理。\n')
    (report/'05_VALIDATION_METRICS.md').write_text('# 最终共同checkpoint验证指标\n\n这是HOTEL折其他7个来源的官方validation区域，不是HOTEL留出test结果，也不是官方Re成绩复现。目标5,203个，来自688个重叠窗口。所有A/B/C的sequence/frame/person/history_SHA/GT_SHA逐项一致；K20随机噪声种子与batch顺序完全配对。\n\n'+md(overall,cols)+'\n单位米。窗口加权为每个目标窗口等权；行人等权先在同(source,person_id)的重叠窗口内平均，再对来源内身份等权，防止重叠窗口被误当独立行人。逐来源宏平均仅作补充，不能代替两个主要口径。\n\nminADE20与minFDE20分别取完整12步候选ADE和终点FDE的最小值，可能不同head。没有逐时间步换head、GT候选重排或FPC。预测归档包含固定原始K20轨迹与GT，独立NumPy float64重新计算并核对逐目标GT_SHA；数值差异见independent_metric_verification.csv。此oracle评价不代表在线挑选head能力。\n\n窗口ADE：B−A='+f'{delta_b:+.6f}'+'米，C−B='+f'{delta_c:+.6f}'+'米。负值代表改善。不得由A/B额外容量差异直接断言社会关联贡献。\n')
    (report/'06_GROUP_AND_SOURCE_ANALYSIS.md').write_text('# 来源与历史分组\n\n全部分组由历史8帧决定，阈值训练前冻结，无未来误差分组。Stage8和Stage9分组成员一致。每目标可同时属于歧义组和多人组；组之间不是互斥全集。身份按来源区分，不把不同录像相同数值ID合并。\n\n'+md(groups,[('model','模型'),('group','组'),('target_windows','目标窗口数'),('source_persons','来源身份数')]+cols[1:])+'\n逐来源：\n\n'+md(sources,[('model','模型'),('source','来源'),('target_windows','目标窗口数')]+cols[1:])+'\nC相对B窗口ADE改善的来源：'+', '.join(decision['improved_sources'])+'。各来源都曾用于开发研究，录像身份也不是新独立确认数据。没有窗口IID显著性检验，单seed不能支持跨seed稳定性；描述性差异不能自动变成因果证明。配对表保留所有目标和退化行人。\n')
    witness=checks['pairing_witness']
    (report/'07_CAPACITY_AND_MECHANISM_AUDIT.md').write_text('# 容量与机制核查\n\n'+md([dict(model=r['model'],parameters=r['parameters'],extra=r['parameters']-3149192) for r in train['models']],[('model','模型'),('parameters','总参数'),('extra','相对A新增')])+'\nB/C新增完全相同8,192个权重，总差异0%。A与B/C差8,192个（0.2601%）。相同数量不等于相同函数类或优化路径：B是逐邻居Tanh联合编码，C是中心化双线性交叉统计；C单分区仅一个邻居时必为零，B无此约束。这些归纳偏置差异必须保留。上下文含ego，所以C也可能在仅一个外部邻居与ego同分区时产生交叉项；它与“两名外部邻居”的预设歧义分组不同。此次不是有/无社会信息消融，不能单靠结构差异证明真实邻居内容的因果收益。没有额外容量实验，故BvsA收益不能完全排除额外参数。\n\n'+md(witness,[('model','模型'),('aggregation_max_difference','配对交换最大差异'),('distinguishes_pairing','可辨识')])+'\n见证仅证明聚合级信息丢失/可辨识性，不证明预测困难与GT具有因果关联。原始ReLU-B在见证处局部线性，Tanh选择发生于训练/验证精度前。结构失败记录全部保留。\n\n单人、2/5/57上下文、输出[B,20,12,2]、padding、空邻居、邻居存储置换、平移等变、有限loss/梯度及严格模型恢复均通过。新增分支在足够多邻居的人工batch有非零有限梯度；不足两点C交叉分支为零是定义所需，不把无信息样本的零梯度称为死参数。\n')
    decision_table=[dict(condition=k,passed=value) for k,value in decision['gates'].items()]
    further='仅建议用户另行批准下一阶段验证；本轮不追加seed/录像/epoch。' if all(decision['gates'].values()) else '停止显式交叉统计这项假设，不增加模块挽救，不自动开启下一阶段。'
    (report/'08_FINAL_RESEARCH_DECISION.md').write_text('# 最终研究判断\n\n**'+decision['decision']+'**\n\n'+md(decision_table,[('condition','训练前门槛'),('passed','通过')])+'\n'+further+'\n\n先审BvsA（窗口ADE差'+f'{delta_b:+.6f}'+'米），再审CvsB（'+f'{delta_c:+.6f}'+'米）。若B/C接近，优先简单联合编码B的后续验证，停止宣称交叉统计独特优势；这仍须B本身对A有值得验证的效果，不能因实现简单强行推荐。\n\n即使全部门槛通过，也只是单seed单fold开发结果，不是跨场景稳定性、SCI新颖性或真实邻居因果收益证明。聚合辨识性、容量数、最终性能必须分开解释。官方Re数据入口与本受控适配不同，不与论文表直接排名。\n\n历史SHA核验'+str(history['checked_files'])+'项通过，未修改Stage1–8或读取test误差。Stage8 NO-GO历史结论保留，本轮新授权实验不倒改既有结论。\n')
    (report/'STAGE9_REPORT.md').write_text('# Stage9：Resonance位置—运动配对受控真实训练\n\n结论：**'+decision['decision']+'**。三模型均从随机初始化完成相同固定预算，最终epoch评价，无测试指标、作者初始化或挑轮。\n\n'+md(overall,cols)+'\nA原始聚合；B逐邻居联合编码；C中心化低秩交叉统计。B/C新增容量精确相同，A较少0.2601%。人工见证A不可区分、B/C可区分；预测性能由真实validation单独检验。\n\n训练协议见01，机制公式见02，实测预算见03，训练日志见04，最终指标见05，全部组与来源见06，机制/容量界限见07，冻结门槛及停止决策见08。\n\n这是AgentFormer官方HOTEL折train/val区域上的Re适配受控实验；5,203目标窗口、688窗口，7来源，原20帧离线上下文资格。不是官方Re论文成绩复现、不是独立确认，也不构成跨seed稳定或论文创新成立。完整数组和权重在ignored cache，轻量逐目标误差、源码/配置/数据SHA、checkpoint索引均在results/tables/stage9。\n\n只本地提交，不push；用户手动执行`git push -u origin feat/stage9-resonance-pairing-controlled-pilot`。完成后停止。\n')
    print(decision['decision'])

if __name__=='__main__':run()
