"""Reports contain resource measurements only, never new forecast-accuracy claims."""
from .common import *
import pandas as pd,numpy as np
import xml.etree.ElementTree as ET

def main():
    diag=json.loads((OUT/'diagnostic_epoch1_execution.json').read_text());ledger=json.loads((OUT/'gpu_budget.json').read_text())
    dec=json.loads((OUT/'final_decision.json').read_text());bud=json.loads((OUT/'training_budget_scenarios.json').read_text())
    dev=json.loads((OUT/'device_execution_audit.json').read_text());eq=json.loads((OUT/'instrumentation_equivalence.json').read_text())
    raw=pd.read_csv(OUT/'kmeans_runtime.csv');pair=pd.read_csv(OUT/'paired_runtime_by_target.csv')
    summary=pd.read_csv(OUT/'state_summary.csv');cpu=pd.read_csv(OUT/'cpu_resident_diagnostic.csv')
    rows=['|状态|固定目标数|迭代中位数|迭代范围|KMeans中位秒|每迭代中位毫秒|完整K20中位秒|',
          '|---|---:|---:|---|---:|---:|---:|']
    for state,g in pair.groupby('state',sort=False):
        rows.append(f"|{state}|{len(g)}|{g.iterations.median():.0f}|{g.iterations.min():.0f}–{g.iterations.max():.0f}|"
            f"{g.wall_seconds.median():.6f}|{g.seconds_per_iteration.median()*1000:.4f}|{g.outer_prediction_seconds.median():.6f}|")
    table='\n'.join(rows)
    scenario_rows=['|情景|未来完整运行秒|未来小时|与当前剩余额差距秒|假设|','|---|---:|---:|---:|---|']
    for s in bud['scenarios']:
        scenario_rows.append(f"|{s['name']}|{s['total_seconds']:.2f}|{s['total_seconds']/3600:.3f}|{s['shortfall_seconds']:.2f}|{s['assumptions']}|")
    scenarios='\n'.join(scenario_rows)
    path_evidence="""原作者`utils/kmeans.py`固定在a345ada1557f53dc2acb4c07f71d340d366980da：

- 第50–58行：X.float()后.to(device)，初始化中心来自同设备X。
- 第43–44、73行：Euclidean分支调用pairwise_distance_function(X,initial_state)，未传第三个device。
- 第145–158行：pairwise_distance默认CPU，两个参数.to(cpu)，距离矩阵CPU；第75行argmin的choice_cluster也在CPU。
- 第79–84行：CPU选择结果索引CUDA X；selected.mean与center赋值在CUDA。隐式索引传输的具体实现和字节数未测。
- 第86–105行：CUDA center_shift参与Python停止条件，存在标量读取/同步路径；平方位移阈值0.001优先于1000次上限。
- 第16–17行：NumPy无放回初始化；空簇分支第81–82行使用CPU torch.randint，再在原设备X索引。

每轮明确的D2H有效载荷估算=(10000×2+19×2)×4=80,152字节；这是静态有效载荷核算，
不是硬件传输量或带宽实测。CPU mask/index的隐式H2D、内核启动、选择和归约同步未被定量拆分。
"""
    path_evidence+=r"""
固定数学语义（未修改）：

\[
q_i=\arg\min_c\sum_d(x_{id}-\mu_{cd})^2,\qquad
\mu_c=\frac{1}{|I_c|}\sum_{i\in I_c}x_i,\quad I_c=\{i:q_i=c\}.
\]

空簇使用原CPU `torch.randint` 选一个采样点；收敛条件为

\[
\left(\sum_{c=1}^{19}\|\mu_c^{new}-\mu_c^{old}\|_2\right)^2<0.001.
\]

随后才检查1000次上限。19中心用NumPy无放回初始化，TTST10000点仍为原CPU Generator有放回采样。
"""
    scope="""本阶段为RESOURCE_PROFILING_ONLY。Stage11C-1正式训练仍0epoch、正式验证0次。
本阶段获准的1epoch仅标记DIAGNOSTIC_ONLY，不能直接作为原正式实验起点；未来复用必须另行批准及核验严格续训。
没有新ADE/FDE、覆盖正确率、真实未来终点分析或测试集成绩。验证诊断只读取物理隔离的8帧history-only缓存和旧语义缓存。
训练未来标签仅用于原始训练Loss。15个目标不是998目标全量验证，也不是独立统计样本。
记录范围审计：360次主路径预测、90次独立原KMeans、180次CPU诊断均实际核查caller RNG不变；
额外6次原Predictor仅核验输出逐位相同，未单独测其RNG，已标NOT_SEPARATELY_MEASURED。
原日志副本保留，未改变预测、时序或判定。
"""
    rootcause=f"""本轮观察到迭代范围{raw.iterations.min():.0f}–{raw.iterations.max():.0f}次，
未监测原KMeans单迭代中位{raw.seconds_per_iteration.median()*1000:.4f}ms。
迭代数与wall的描述性Pearson={dev['iteration_wall_Pearson_descriptive']}（不作显著性检验，不等于因果分解）。
如果执行1000次，以本轮每迭代中位成本投影约{dev['hypothetical1000_iterations_seconds']:.4f}s，
与旧约5.2秒量级相容；但这是条件外推，旧临时预测状态没有迭代trace，**不能据此追认旧样本实际达到1000次**。
旧C1的2个训练目标、临时多步更新模型，与本轮15个验证历史及随机初始/Epoch1并非同一配对实验。
CPU常驻诊断中心逐位相同{dev['CPU_centers_bitwise_count']}/{dev['CPU_centers_total']}，
最大中心坐标差{dev['CPU_center_max_abs_difference']:.9g}网络坐标像素。
CPU路径和原路径使用完全相同的采样点及初始化种子，但归约、设备访问、内核调用和同步共同改变。
所以设备混合路径及其成本风险有源码和运行证据，**传输具体贡献多少秒、占比多少仍UNVERIFIED**。
不能把两条路径耗时之差全部算作传输。
"""
    extrap="""情景A来自旧语义完整30轮和7次全量验证的真实记录，迁移到无语义模型仍是假设。
情景B以本轮完整epoch耗时外推30轮，用epoch1后每录像3个固定目标的普通完整预测耗时均值，
按真实录像目标数投影998个目标和7次验证，另计重复检查、旧评分/保存开销、31个checkpoint、启动、异常和退出预留，增加10%时间裕度。
情景C对每录像取NoSemantic初始与epoch1的较慢P90，假设7次验证均保持该慢速，增加20%裕度。
这是条件性压力情景，不是总体统计上界，也不能假定实际epoch30仍为初始速度。
没有对未测epoch做插值；所有30轮和998目标的新增成本均是外推，未执行。
当前剩余额已扣除Stage11C-1和本轮全部GPU相关进程wall；诊断checkpoint未获准复用，因此成本从头计算。
"""
    acceleration="""|建议|是否本轮替换|严格等价风险|
|---|---|---|
|只读缓存场景尺寸、原始文件SHA、已核验语义张量|否，已有部分缓存复用|低，需校验值/dtype/device和失效规则|
|减少重复模型加载|否|同权重、eval、RNG及缓存生命周期需一致|
|缓存历史预处理|否|需保证每目标坐标、舍入、padding及模板完全一致|
|CPU常驻聚类或改变设备传输|仅隔离诊断|可能改变均值归约、聚类中心及后续Path；非现成严格等价优化|
|重写KMeans/融合循环|否|顺序、精度、空簇随机与停止时刻均可能变化|

未来D/E类替换最低验收：998目标固定权重、相同10000 proposal、19中心、20Goal、完整20Path均逐位一致；
同ID重复与逆序一致，原RNG完全隔离。15个样本即使全部相同也不够。
本轮未对作者源码或正式采样器做任何替换，不能宣称已实现严格等价加速。
"""
    before=pair[pair.state=='NoSemantic-initial'];after=pair[pair.state=='NoSemantic-epoch001']
    paired_change=json.loads((OUT/'paired_runtime_change.json').read_text())['differences']['wall_seconds']
    recommended_total=bud['conservative_stress_combined_budget_seconds']
    budget_recommendation=f"建议提交用户审查的累计预算为{recommended_total:.0f}秒（{recommended_total/3600:.2f}小时），比原18000秒增加{recommended_total-18000:.0f}秒；它覆盖本轮条件压力情景及已用预算，不是实际耗时保证，也不构成正式训练授权。"
    qa=[
        ('旧停止是否主要因早期成本估计偏高？',f"旧拒绝确实使用临时早期模型的约5.2秒聚类投影。当前随机初始/epoch1 KMeans中位分别{before.wall_seconds.median():.4f}/{after.wall_seconds.median():.4f}秒；这检验了早期估计的代表性，但不能把变化单因归于语义有无。"),
        ('实际迭代次数？',f"全部固定状态/目标范围{raw.iterations.min():.0f}–{raw.iterations.max():.0f}；逐样本见kmeans_iteration_counts.csv。"),
        ('有设备切换成本？','明确存在CUDA输入转CPU距离、CPU指派与CUDA更新的混合路径，见源码与设备事件。'),
        ('切换成本已量化？','仅明确D2H有效载荷和两条整体路径wall；切换秒数、占比UNVERIFIED。'),
        ('实际一epoch多久？',f"训练循环{diag.get('train_seconds',float('nan')):.6f}秒，完整GPU父进程另含构造、保存、恢复与退出。"),
        ('初始与epoch1成本变化？',f"初始/epoch1每15目标原KMeans总计{before.wall_seconds.sum():.4f}/{after.wall_seconds.sum():.4f}秒；{paired_change['lower']}/15变快，{paired_change['higher']}/15变慢，每ID配对保存，未删除退化样本。"),
        ('原语义epoch1/5/10/30有变化？','见相同15身份的逐样本及中位/P90表；不以不同样本组冒充配对趋势。'),
        ('有可靠完整预算？','具备完整epoch和配对采样实测锚点，但以后epoch及全998成本仍外推，三情景不是完整运行证据。'),
        ('5小时够吗？',f"预算准入={dec['BUDGET_ADMISSION']}，不能用情景A或B最乐观结果保证安全完成。"),
        ('额外预算多少？',budget_recommendation),
        ('有严格等价工程加速？','存在I/O与加载的候选机会，尚未验证完整998等价；CPU迁移不能直接认定等价。'),
        ('需改原数值协议？','本轮不需要且没有修改。预算不足应交用户审查，不通过换KMeans、减少K或跳验证解决。'),
        ('具备正式准入？','FORMAL_TRAINING=NOT_AUTHORIZED；本阶段不能自动启动。'),
        ('阻断条件？','完整30轮+7验证未实测；后续概率分布和聚类成本未获保证，预算情景及授权需审查；不能擅用诊断checkpoint续训。'),
        ('下一步唯一建议？','用户审查原样完整训练的预算与明确授权；保持数值协议，不自动替换聚类、不训练其他seed。')]
    qtext='\n'.join(f'**Q{i} {q}** {a}\n' for i,(q,a) in enumerate(qa,1))
    verdict='\n'.join(f'- {k}: **{dec[k]}**' for k in ['ENGINEERING_AUDIT','KMEANS_ROOT_CAUSE','BUDGET_ADMISSION','PROTOCOL_COMPATIBILITY','FORMAL_TRAINING'])
    tests='完整pytest待执行，不能标为通过。'
    if (OUT/'full_tests.xml').exists():
        root=ET.parse(OUT/'full_tests.xml').getroot();suite=root.find('testsuite')
        n=int(suite.get('tests'));skip=int(suite.get('skipped'));fail=int(suite.get('failures'));err=int(suite.get('errors'))
        tests=f'完整pytest：{n-skip-fail-err}通过，{skip}跳过，{fail}失败，{err}错误；原警告和SKIP保留。'
    hist=json.loads((OUT/'history_after.json').read_text()) if (OUT/'history_after.json').exists() else None
    integrity=f"历史{hist['files']}文件SHA复核passed={hist['passed']}。" if hist else '最终历史SHA复核尚待完成。'
    contents={
        '01_KMEANS_SOURCE_AUDIT.md':f'# 原KMeans源码审计\n\n{path_evidence}\n\n{scope}\n',
        '02_RUNTIME_PROFILING.md':f'# 运行成本与监测等价性\n\n{table}\n\n全部同15历史身份，普通运行两次取逐ID均值；原KMeans单独无监测重跑一次，计数在两个观察运行中一致。\n\n{rootcause}\n\n外部sys.setprofile回调只读Python事件和设备元数据，不执行tensor算术；退出后才读取终止center_shift。实际监测等价记录{eq["comparisons"]}次，passed={eq["passed"]}。\n\n所有索引、proposal、中心、Goal、Path SHA以及RNG隔离见equivalence_details.csv，原作者Predictor也作为每状态首目标对照。\n\n{scope}\n',
        '03_DIAGNOSTIC_EPOCH1.md':f'# 诊断完整Epoch1\n\n状态：{diag["status"]}；role=DIAGNOSTIC_ONLY。\n\n初始核心SHA `{INITIAL}`，新空Adam，seed17。7496唯一目标、1882更新、原场景/目标顺序SHA核对通过。语义六通道全画布为0，分割器不前向且SHA不变。\n\n训练循环实测{diag.get("train_seconds")}秒；峰值allocated={diag.get("peak_allocated_bytes")}，reserved={diag.get("peak_reserved_bytes")}字节。\n\n完整模型、Adam、Python/NumPy/Torch CPU+CUDA RNG、配置和源码SHA保存在独立diagnostic_epoch001.pt；保存后才开始采样。严格恢复模型/Adam/RNG身份验证={diag.get("strict_checkpoint_restore_model_optimizer_RNG")}。\n\n{scope}\n',
        '04_FULL_TRAINING_BUDGET_ESTIMATE.md':f'# 原样完整训练预算\n\n{scenarios}\n\n{extrap}\n\n{budget_recommendation}\n\n已用原C1={ledger["OriginalStage11C1Used"]:.6f}s，本轮={ledger["Stage11C1RUsed"]:.6f}s，合计={ledger["CombinedUsed"]:.6f}s，剩余={ledger["CombinedRemaining"]:.6f}s。\n',
        '05_ENGINEERING_ACCELERATION_RISKS.md':f'# 工程加速机会与边界\n\n{acceleration}\n\n{rootcause}\n',
        '06_FINAL_ADMISSION_DECISION.md':f'# 最终准入决策\n\n{verdict}\n\n{qtext}\n\n{scope}\n',
        'STAGE11C1R_REPORT.md':f'# Stage11C-1R：无语义训练成本与KMeans预算审计\n\n{verdict}\n\n{scope}\n\n{table}\n\n{path_evidence}\n\n{rootcause}\n\n诊断训练循环实测{diag.get("train_seconds")}秒。\n\n{scenarios}\n\n{extrap}\n\n原C1已用{ledger["OriginalStage11C1Used"]:.6f}s；本轮GPU相关进程{ledger["Stage11C1RUsed"]:.6f}s；合计{ledger["CombinedUsed"]:.6f}s；剩余{ledger["CombinedRemaining"]:.6f}s。失败进程若有也计入，无另扣CPU反事实时间。\n\n{tests}\n\n{integrity}\n\n{qtext}\n'}
    for name,body in contents.items():
        with (ROOT/'reports/stage11c1r'/name).open('x') as f:f.write(body.rstrip()+'\n')
    print('REPORTS_WRITTEN',len(contents),flush=True)
if __name__=='__main__':main()
