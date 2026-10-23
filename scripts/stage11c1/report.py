"""Evidence-grounded reports for the actual budget admission outcome."""
from .common import *
import xml.etree.ElementTree as ET

def main():
    b=json.loads((OUT/'budget_preestimate.json').read_text());p=json.loads((OUT/'preflight_results.json').read_text())
    init=json.loads((OUT/'initialization_audit.json').read_text());data=json.loads((OUT/'data_audit.json').read_text())
    old=json.loads((ROOT/'results/tables/stage11b1c/training_execution.json').read_text())
    oldm=old['best_metrics'];spent=parent_seconds()
    budget=f"实测GPU相关父进程累计 **{spent:.6f}s**；硬上限18000s，安全阈值16800s。"
    costs=f"""|成本|秒|性质|
|---|---:|---|
|临时模型预检完整进程|{spent:.6f}|实测，包含启动、构造、更新、K20、保存和退出|
|按25场景固定batch投影单轮|{p['projected_epoch_seconds']:.6f}|投影，未执行完整epoch|
|K20单目标，六次调用平均|{p['mean_K20_target_seconds']:.6f}|两训练目标，未评价验证GT|
|K20单目标最大值|{p['maximum_K20_target_seconds']:.6f}|同上|
|Checkpoint保存并校验|{p['checkpoint']['save_and_hash_seconds']:.6f}|实测临时模型|
|保守全计划|{b['conservative_all_validations_seconds']:.6f}|30轮及7次验证投影，非已发生耗时|
|仅第一次验证较慢，其余沿用旧验证上界|{b['first_validation_slow_rest_historical_high_seconds']:.6f}|情景估算，仍超过硬上限|
|全部沿用原语义模型历史耗时|{b['historical_only_optimistic_seconds']:.6f}|乐观参照，不能保证无语义模型相同|
"""
    caveat="""预算不准入不等于证明完整实验实际一定超时。预检仅2个训练目标、一个临时早期模型，
其KMeans耗时随概率分布和聚类收敛变化，不能证明第1轮正式验证也需要同样时间。
本轮没有计数KMeans实际迭代次数，没有证据把耗时归因于固定某次数迭代或无语义模型的固有缺陷。
决策依据是用户要求的**保守估算准入**无法满足预算，不能据乐观旧模型耗时直接启动。
没有改采样、KMeans、K、验证规模、epoch或学习率来绕过预算。"""
    limits="""这是已反复使用的5录像开发协议，非独立确认。998窗口共享背景及行人，不独立。
训练与验证目标来自冻结的25/5录像；测试17录像2829目标未加载或评分。
历史保护对旧文件做流式字节校验不等于读取测试轨迹评价。
全零输入仍保留绝对轨迹坐标及图像网格大小，不能称为完全无场景信息。
同为1641381名义预测器参数，但1728个语义输入连接梯度恒零，结构有效参数上界1639653；
这仍非精确函数容量匹配。六类语义编号映射未完整核实且无独立分割GT，语义质量因果作用未知。"""
    testsfile=OUT/'full_tests.xml'
    if testsfile.exists():
        root=ET.parse(testsfile).getroot();suites=root.findall('testsuite') if root.tag!='testsuite' else [root]
        totals={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        testing=f"完整pytest：{totals['tests']-totals['failures']-totals['errors']-totals['skipped']}通过，{totals['skipped']}跳过，{totals['failures']}失败，{totals['errors']}错误。详见full_tests.txt/xml。"
    else:testing='完整pytest记录尚未生成，不能宣称通过。'
    hist=OUT/'history_after.json'
    integrity=json.loads(hist.read_text()) if hist.exists() else None
    protecting=f"历史最终校验：{integrity['files']}文件，passed={integrity['passed']}。" if integrity else '历史最终SHA校验待执行；首次95657项通过。'
    ref=f"""冻结语义Epoch30仅作为只读历史参照：ADE1={oldm['ADE1']:.12f}px，FDE1={oldm['FDE1']:.12f}px，
minADE20={oldm['minADE20']:.12f}px，minFDE20={oldm['minFDE20']:.12f}px，GoalCover25=834/998。
这些值来源Stage11B-1C原始training_execution.json及配对结果，**不是本轮无语义成绩**。"""
    decision="""|验收项|结论|
|---|---|
|A 训练工程|临时模型预检GO；完整正式训练链尚未运行验收|
|B 预算|NO-GO，PRETRAINING_ADMISSION阶段STOPPED_BUDGET|
|C 开发集语义收益|UNKNOWN，没有训练匹配的无语义预测，不能认定收益或无收益|
|D 跨录像一致性|UNKNOWN|
|E 多seed确认|不启动；先解决固定单seed实验预算准入|
|F 语义质量因果作用|未证明|
|G 语义可靠性模块|NO-GO，未授权开发|
|H 唯一优先建议|先审查是否为原样30轮+7验证提供可承担预算；原设置不变，不搜索新变体|
"""
    qa=[
        ('完成30轮？','没有。正式0轮；仅临时模型工程预检。'),
        ('完成7次998目标验证？','没有，0次；未生成无语义验证预测。'),
        ('无语义最佳epoch？','不存在。'),('best minADE20？','未运行，未知。'),('best配对minFDE20？','未运行，未知。'),
        ('五录像macro minADE20差距？','未知，没有匹配无语义结果。'),('窗口加权minADE20差距？','未知。'),
        ('GoalCover25差异？','未知，不能使用旧推理置零结果替代从头训练对照。'),
        ('K1与K20何者受影响？','未知。预检K20只用于工程检查。'),('五录像方向一致？','未知；全部录像均保留。'),
        ('deathCircle是否主导？','未知，未进行该比较。'),('复杂运动是否更需语义？','未知，不能用相关性代替因果。'),
        ('原164失败目标如何变化？','已冻结原164/998身份；恢复、新增失败均未知。'),
        ('语义有开发集预测收益？','本轮未获得判断证据。'),('满足冻结工程科研门槛？','四项科研门槛均不可计算，NOT_EVALUATED；预算门槛不通过。'),
        ('多seed？','不建议现在启动，先解决原单seed预算准入。'),
        ('核查分割质量？','可保留为将来问题，当前没有独立GT且本轮未运行，不构成当前优先实验。'),
        ('开发可靠性门控？','没有充分证据；不批准、不开发。'),
        ('下一阶段具体问题？','现有冻结30轮及7次全量验证的预算能否获可靠保障？先交用户审查，不自动追加。'),
        ('硕士小论文方向调整？','本次只被预算阻断，不能凭未运行结果决定修改论文方向。')]
    qtext='\n'.join(f'**Q{i} {q}** {a}\n' for i,(q,a) in enumerate(qa,1))
    protocol="""新输入14通道=6全零语义+8原始历史距离热图。语义float32全画布含padding为零；
原Goal与Path decoder、softargmax、损失及GT waypoint训练teacher forcing保持。
验证接口只接收冻结history-only缓存，不接收GT；定义为先保存998条K20结果并校验完整性，再读取GT评分。
K20仍为原CPU multinomial每sample_id独立SHA种子、10000有放回proposal、原19中心KMeans与1个softargmax。
模型选择仍为完整验证中窗口加权minADE20最低，完全相同则较早epoch；配对FDE同一checkpoint。
科研主比较固定Epoch30对Epoch30，五录像等权macro；best比较单列次要分析，不能事后挑有利方案。
科研门槛分母统一NoSemantic：macro ADE改善≥2%，macro FDE语义退化≤1%，至少3/5录像ADE支持语义，
OTHER4合计语义ADE退化≤1%。最后一项把用户“明显反向”事前具体化为1%，未依据验证误差选择。
Delta_ADE/FDE=NoSemantic−Semantic；Delta_Cover=Semantic−NoSemantic，正值均支持语义。
"""
    contents={
        '01_INITIALIZATION_AUDIT.md':f"# 初始化与数据身份审计\n\n临时模型调用原build_model(seed=17)，新Adam状态为空。\n\n预测器初始SHA：`{init['random_predictor_state_SHA']}`，与原initial只读哈希核验一致。\n\n从未加载原Epoch30预测器为初始化；没有正式initial.pt。临时smoke权重未作为正式起点。\n\n参数总数53155339，冻结分割51513958，预测器名义1641381。1728输入连接梯度恒零，结构上界1639653，不宣称函数容量完全匹配。\n\n数据8494目标历史/GT SHA全部通过，30轮预定顺序与旧结果一致。见data_audit.json、expected_orders.csv。\n\n{limits}\n",
        '02_NO_SEMANTIC_TRAINING.md':f"# 无语义训练工程与预算\n\n**正式训练未启动。完整epoch=0，正式optimizer更新=0。**\n\n预检75个基准batch+2次严格恢复重放更新，均属于临时模型；不是完整epoch或性能实验。\n\n{protocol}\n\n{budget}\n\n{costs}\n\n{caveat}\n\n原损失finite、梯度finite、真实K20完整路径、重复与逆序、原始未来反事实、checkpoint严格恢复后下一步一致均通过。\n\n记录：preflight_results.json、preflight_steps.csv、preflight_gradients.csv、preflight_K20.csv。\n",
        '03_VALIDATION_RESULTS.md':f"# 验证状态\n\n7次固定998目标验证均**未执行**；没有NoSemantic ADE/FDE或GoalCover，也没有best或last正式checkpoint。\n\n原语义与新无语义没有形成数值对照。validation_metrics.csv仅表头，不能把空值当零。\n\n{ref}\n\n图1为冻结协议示意；其余11项需要真实训练/验证/配对预测，未生成。预算与预检耗时图仅显示实测与标注估算，见figure_manifest.json。\n",
        '04_PAIRED_SEMANTIC_COMPARISON.md':f"# 配对比较状态\n\n**NOT_EVALUATED**。没有无语义预测，不能计算固定Epoch30或best比较。\n\n{protocol}\n\n原语义失败164、成功834仅冻结为参照；无语义恢复、丢失、两者共同失败均未知。所需比较CSV为空schema或显式NOT_EVALUATED，不虚构样本行。\n\n{ref}\n",
        '05_SCENE_AND_MOTION_ANALYSIS.md':f"# 录像与运动分组\n\n五录像保持：bookstore_2=423，deathCircle_3=443，gates_7=24，hyang_7=45，nexus_4=63，总998。OTHER4=555。\n\n没有可比无语义输出，全部录像差值及ALL998、MACRO、deathCircle、OTHER4均未知。\n\n沿用Stage11B-2A历史速度、转向、变速、近静止分组；未执行组间收益对比，不删难例。\n\n按人ID等权、LORO与真实案例同样没有对照结果，不能生成支持某模型的案例图。\n\n{limits}\n",
        '06_RESEARCH_DECISION.md':f"# 研究决策\n\n**预算NO-GO；语义收益NOT_EVALUATED。**\n\n{decision}\n\n{caveat}\n\n{limits}\n\n{qtext}\n",
        'STAGE11C1_REPORT.md':f"# Stage11C-1：匹配无语义Y-Net对照——预算准入报告\n\n状态：**STOPPED_BUDGET / PRETRAINING_ADMISSION**。工程smoke通过；正式训练及验证均未启动。\n\n{budget}\n\n{costs}\n\n{caveat}\n\n{protocol}\n\n{decision}\n\n{testing}\n\n{protecting}\n\n原始测试断言失败被保留在validation_test_failure_record.json，浮点归约约2e−16差异只修正测试断言；未改变指标、K20或逐位重复性要求。\n\n{limits}\n\n{qtext}\n\n交付仅本阶段新目录；原模型、指标和冻结源码未改。大型临时checkpoint保留本地cache并被忽略。正式训练源码保留供审查，但未获资源准入，不能据此自动执行。\n"}
    directory=ROOT/'reports/stage11c1'
    for name,body in contents.items():
        with (directory/name).open('x') as f:f.write(body.rstrip()+'\n')
    print('REPORTS_WRITTEN',len(contents),flush=True)
if __name__=='__main__':main()
