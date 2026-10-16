"""Evidence-led Stage7 decision, never invent real-scene prediction scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,xml.etree.ElementTree as ET
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

ROOT=Path('results/tables/stage7');REPORT=Path('reports/stage7')
agent=json.load(open(ROOT/'AgentFormer_compat_admission.json'))
prior=json.load(open(ROOT/'AgentFormer_full_prior_checks.json'))
pec=json.load(open(ROOT/'PECNet_native_admission.json'))
resource=json.load(open(ROOT/'AgentFormer_max_context_resource_probe.json'))
cohort=json.load(open(ROOT/'minimal_cohort_audit.json'))
plan=json.load(open('configs/stage7/minimal_reproduction_plan.json'))
am=json.load(open(ROOT/'sources/AgentFormer.json'));pm=json.load(open(ROOT/'sources/PECNet.json'))
au=f"https://github.com/Khrylx/AgentFormer/blob/{am['commit']}";pu=f"https://github.com/HarshayuGirase/Human-Path-Prediction/blob/{pm['commit']}/PECNet"
tests=sum(int(r.attrib['tests']) for r in ET.parse(ROOT/'full_tests.xml').getroot().findall('testsuite'))


def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'


def save(name,text):
 with (REPORT/name).open('x') as f:f.write(text.rstrip()+'\n')


scope='''本阶段新正式训练=0、优化器更新=0、ETH评价=0。所有新神经输出均为固定随机权重的人工输入功能验证，不是HTraj真实ADE/FDE或论文性能复现。官方预训练权重没有下载、没有严格加载；严格恢复只指本轮随机state_dict往返。模型源码保持原始Git blob不变，依赖兼容别名、包装器与检查记录独立保存。

'''
save('01_HISTORICAL_RESEARCH_CONCLUSIONS.md','# 历史研究结论与正式关闭范围\n\n'+scope+'''Stage1至Stage6.1代码、协议、指标、报告及checkpoint全部保护；不改旧学习率、seed、选择规则或采样比例。下列关闭对象不因换公开主干而悄悄恢复。

|关闭对象|原始证据|本轮决策与边界|
|---|---|---|
|TOHI高阶/图模块|Stage2.6：H3对H2三seed ADE都差，H2独立收益不稳定|停止增加高阶/时间传播结构；不推出所有社会预测必然无效|
|G0/G1社会图原型|Stage4.2最终同预算100轮：ZARA2 G1−G0 ADE +0.091252m，HOTEL −0.046452m，均弱于同目标CV|关闭原型迭代；单录像好不代表稳定独立社会收益|
|SocialVAE社会注意力改造|Stage5.5 Full−Placebo：HOTEL窗口加权2/3seed改善，人等权仅1/3；ZARA2两权重3/3|未通过跨录像/seed门槛，停止社会注意力/关系GRU/GNN追加|
|容量与Placebo路线|Stage5.3 NoSocial结构有效上界比Full少585792，5.4控制函数类不完全相同，5.5内容对照不稳定|已有对照冻结，不再用Full略优NoSocial直接宣称社会信息有效|
|Stage6.1历史速度分层采样|新30轮U/S：HOTEL ADE .395194→.379526，ZARA2 .280626→.287247；HOTEL正常速度偏差下降30.10%，静止FDE .207669→.295994|停止比例/阈值/模型/seed/训练长度搜索；ZARA2-U best=30，有限预算不能证明更长训练必然无效|

来源：reports/stage2p6/STAGE2P6_REPORT.md；reports/stage4p2_final/05_FINAL_RESEARCH_DECISION.md；reports/stage5p5/STAGE5P5_REPORT.md；reports/stage6p1/06_RESEARCH_DECISION.md，各原始表与权重SHA仍保留。不把partial、offline完整20、严格历史上下文三类协议串成同一数值排行榜。

公开模型历史审计也不重复失败：Social-STGCNN维护版节点卷积排列敏感；原Full/NoSocial存在BN/PTN混杂，公平G0/G1不能挽救旧结构。EqMotion官方历史源码/权重不严格匹配，不能用strict=False补齐随机参数。HighGraph未定位到可冻结完整实现。SGCN空间mask沿节点索引卷积，Stage6已有换序反例和train累计边界风险；其TXP节点实际在batch轴的纠正保留，不能继续传播旧静态误判。SocialVAE已证明可训练但社会独立价值未稳定。SingularTrajectory仅源码/组件审查，地图/vector field与协议成本尚未解决。本轮不重新调试这些路径。

旧失败支持“本项目这些协议/预算下旧假设未获稳定支持”，不支持“任何GNN或邻居信息没有价值”。新公开模型首先接受技术准入，再看真实同目标基线，不以论文排名抵消旧负结果。
''')

save('02_PUBLIC_BASELINE_COMPARISON.md',f'''# 两个新公开候选：版本与选择

{scope}
只详细检查AgentFormer、PECNet。引用的ICLR2022归因论文仅为相关工作，不是第三套候选实现。核查日期使用项目客户端日期2026-10-17。

|维度|AgentFormer维护版|PECNet公开版本|
|---|---|---|
|正式来源|[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html)，[arXiv v3](https://arxiv.org/abs/2103.14023v3)|[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470749.pdf)，[arXiv v3](https://arxiv.org/abs/2004.02025v3)|
|作者源码commit|`{am['commit']}`|`{pm['commit']}`；旧PECNet URL重定向Human-Path-Prediction|
|许可|MIT/CMU，保留原许可|MIT，保留原许可|
|结构|共享时空Transformer，区分同/不同行人注意力，显式agent_mask|个体终点CVAE，显式N×N非局部关系池化，终点条件路径MLP|
|输入/输出|8帧完整位置/位移；12帧2D，K个联合预测|8帧展平位置、末观测位置、历史mask；11中间点+生成终点=12帧|
|已测参数|{agent['parameter_count']}|{pec['parameter_count']}|
|从头入口|[train.py]({au}/train.py)，ETH/UCY配置齐全；先CVAE再DLow|[training_loop.py]({pu}/scripts/training_loop.py)，公开主体主要是SDD|
|公开权重|作者提供Drive/Baidu链接，未下载/核验|saved_models文件可定位，未下载/核验|
|8/12及单位|ETH配置明确8/12，原位移不是m/s；0.4秒/米必须按数据源核验|SDD optimal.yaml使用scale1.86、initial_pos/1000，不能直接当ETH米制协议|
|实际RTX3080检查|N1/2/5/12及最大源上下文N57原始K20损失反向可运行|N1/2/5/12 float64原始损失反向可运行|
|主要剩余问题|官方完整20帧上下文使用未来存在性，不能直接称严格历史因果loader|官方评价先用GT终点选样再社会插值；跨代理oracle上下文与独立minADE不是同一评价|

**唯一保留的后续复现主干：AgentFormer-CVAE原公开pre阶段。** 选择依据是明确ETH/UCY入口、完整时空交互、排列测试和最大上下文反向通过，非论文成绩。不是已经完成其全部两阶段论文复现，也不是立即可开发创新算法的结论。PECNet核心可运行，但SDD单位/异步关系批次、外置ETH数据处理与GT选终点评价需要更多重建，本阶段不把它列作第二个推荐主干。

AgentFormer作者曾确认旧版以历史+未来共同求scene_orig的泄漏，[官方问题#5](https://github.com/Khrylx/AgentFormer/issues/5)说明已修正并更新结果。当前源码scene_orig只取历史；NaN/大幅改变未来标签未影响实际learned-prior预测。必须冻结维护版，不沿用修正前paper数字。官方test.py本身strict=False，不能以脚本未报错宣布预训练匹配；本轮没有加载作者权重。

环境：Python3.10、torch2.5.1+cu124、torchvision0.20.1、NumPy1.26.4、RTX3080。glob2==0.7、easydict==1.13仅安装到新cache/deps，不改旧环境；AgentFormer原生import因私有_LinearWithBias移除失败，运行时同bias线性类兼容别名后通过，未修改源码、层宽、损失。完整数据loader仍有np.int旧API，下一阶段需独立兼容入口smoke，不能声称整条官方train.py已端到端跑通。

源码53个目标文件（43+10）逐项Git blob/SHA核验；AgentFormer初次3个SSL失败仅重试一次后成功，原失败记录保留。git ls-tree --long会触发partial clone懒取blob，首尝试约44秒后停止，改为只读tree元数据。学术MCP未挂载，OpenAlex429、CVF直接403、OpenReview挑战均保留说明；采用作者/arXiv/正式出版元数据交叉核查，不宣称穷尽检索。
''')

rows=[]
for name,result in [('AgentFormer兼容',agent),('PECNet原生',pec)]:
 for r in result['cases']:
  rows.append([name,r['people'],r['passed'],f"{r['K1_fixed_latent_permutation_max']:.3e}",f"{r['K20_coupled_latent_permutation_max']:.3e}",
      r['poisoned_future_max_difference'],r['strict_random_checkpoint_restore'],f"{r['peak_allocated_bytes']/2**20:.2f}"])
save('03_BASELINE_TECHNICAL_ADMISSION.md','# 实际技术准入：模型、协议和失败分开\n\n'+scope+table(
 ['模型','N','有限前向/loss/梯度/恢复通过','固定z换序max','K20同z换序max','未来NaN差','严格随机权重恢复','峰值分配MiB'],rows)+f'''
上表第一轮AgentFormer使用显式固定z，不能单独代表learned-prior全路径。补充AgentFormer_full_prior_checks.json保留p(z|H)的mu/sigma并按ID同步epsilon：N1/2/5/12均通过；mu/logvar也同步置换；K1 prior-mean与K20最大预测差均≤4.77e-7，GT污染差与严格恢复差均0。原始Normal.sample临时只替换标准噪声，loc/scale/模型/decoder不变，退出即恢复。单人/多人数据适配单测验证了同步mask和逆置换，不用未配对随机数制造差异。

N57是既有非ETH源训练manifest的最大上下文人数；原始三项loss含K20、完整前向+反向一次均有限。实测allocated `{resource['peak_allocated_bytes']}` bytes、reserved `{resource['peak_reserved_bytes']}` bytes，总检查约{resource['elapsed_seconds']:.3f}秒（含初始化，非每训练窗口吞吐）。显存余量小，正式运行须串行；不能由N12的550MiB外推所有真实场景成本，不默默删除拥挤窗口。

## 未来信息隔离的两层结果

预测函数层：agentformer_packet只接受[N,8,2]，历史外的20帧包被拒绝；共享容器补零future及零mask以满足官方接口。训练后验才显式附加标签。原始inference(mode=infer)的先验/decoder不使用fut数值，改变为NaN或远移1000也不改变预测。`mode=recon`会使用未来后验，禁止作为可部署K1；官方test虽额外保存recon，但评价目录是samples，不能误报其官方samples就是GT重建。

数据选择层：直接执行原preprocessor.get_valid_id的方法体，保持全部历史相同、仅删除邻居某一未来帧，其上下文ID从[0,1]变成[0]。因此**官方loader未通过严格历史上下文准入**。这是完整20帧离线筛选与历史因果任务的差别，不能用模型层GT数值隔离掩盖。后续冻结离线A任务要明确承认未来存在性条件化，不能把它命名HTraj-Causal8x12-v1；若论文坚持严格历史可部署上下文，本轮整链路No-Go。

## 保留失败与限制

首次完整单测4失败/381通过，原因是测试自身用负stride NumPy反向索引Torch；改为连续copy后，最终{tests}项通过，first_full_tests.xml/txt均保留。AgentFormer原生import失败日志也保留；只是机械兼容修复，不宣称原环境无问题。未运行作者预训练、真实ADE/FDE、优化器更新或正式epoch。人工“loss有限”不等于真实收敛，也未验证DLow阶段。候选源码、脚本、依赖报告与原始JSON全部交付。
''')

save('04_PAPER_INNOVATION_FEASIBILITY.md',f'''# 论文创新可行性：有复现对象，尚无新算法证据

**唯一后续对象是AgentFormer-CVAE，不直接命名新的创新模块。** 总题目《面向复杂场景的行人轨迹预测关键技术研究》可以容纳动态人数/上下文变化、运动与交互协同，但题目适配不等于已具备论文贡献。

## 模型信息路径与已有成果

H∈R^(N×8×2)。维护配置c=所有历史点均值，v_t=p_t−p_(t−1)，首段位移复制使8帧对齐；这是米/采样步，不把输入命名真实m/s。拼接[p−c,v]后映射256维，加入时间编码，节点索引编码默认关闭。共享时空Transformer以I(agent_i=agent_j)区分同人/他人投影，历史agent_mask显式限定连接；当前配置默认近似全连接。每人历史token均值得context，learned prior p(z_i|H)，训练q(z|H,Y)，z维32。

自回归decoder逐步生成12帧，当前配置sn_out_type=norm将输出加最后历史位置；未来编码器只用于训练后验或recon。损失是作者MSE+min_clip KL+best-of20 sample项，不能称精确边缘NLL。完整论文还包括后续DLow sampler；本轮只测试并选择原CVAE/pre阶段，不能冒充完整两阶段复现。主要关系计算含O((TN)^2 d)，K20与自回归增加时间/显存成本。

来源：[AgentFormer正式论文](https://arxiv.org/abs/2103.14023v3)、[维护源码]({au}/model/agentformer.py)。时空联合、agent-aware attention、变分意图、概率采样、节点排列一致性均是已有成果。PECNet的终点条件CVAE/非局部池化也是既有结构，不重新命名为HTraj创新。

## 具体但尚待真实验证的问题

默认密集上下文与scene中心随人数改变：在相同两个目标历史下加入3个偏移100m的人工行人，固定随机网络的prior-mean预测最大变化{prior['distant_context_probe']['original_agent_prediction_max_difference_m']:.6f}m。证明存在上下文依赖，**不证明已训练模型预测退化、不证明远邻必然无效，也没有分离scene中心与注意力影响**。待验证假设：在真实训练权重、同目标GT、历史定义的远离/低相对交互条件下，不相关上下文增加是否引起不必要的预测变化和误差？若没有误差影响，这个候选问题立即停止。

对比应先固定目标/采样噪声，改变历史上下文而不打乱目标运动；同时区分固定checkpoint分布外干预和独立重训，记录人数/距离/关系预算。centering、有效参数和归一化不一致不能混成图独立收益。单纯调conn_dist、删邻居、加注意力或改归一化都是常见做法，不能直接作为新算法贡献；本阶段不实现任何这种改造。

## 文献覆盖与投稿风险

[You Mostly Walk Alone，ICLR2022](https://openreview.net/forum?id=POxF-LEqnF)已经研究轨迹模型的特征归因及社会信息贡献，[作者研究所记录](https://is.mpg.de/en/publications/makansietal21)与[作者预印本](https://arxiv.org/abs/2110.05304)交叉核实。因此“发现个体运动强、邻居贡献小”不能本身当新算法；更不能用本地负结果普遍否定交互预测。AgentFormer已解决跨时间/跨行人联合注意，SGCN/HighGraph/UniEdge的既有图与高阶问题继续占位。

能争取的贡献只有：先在可信公开主干上发现可重复的具体失效，再给出与已有方法有计算差异、超过同目标强CV和个体对照的解决方案，并排除容量/选择/上下文混杂。当前没有这样的完成证据，没有独立确认数据，两个开发录像已高度曝光；没有足够证据承诺一篇可投稿小论文。现阶段创新开发No-Go，基线复现有条件Go。
''')

count_table=table(['留出录像','split','窗口','目标','单人窗口','最大人数'],[
 [r['recording'],s,r[s]['windows'],r[s]['targets'],r[s]['single_person_windows'],r[s]['max_context_people']]
 for r in cohort['cohorts'] for s in ['train','validation']])
save('05_FINAL_GO_NO_GO.md',f'''# 最终Go/No-Go及最低正式实验

**Go仅限AgentFormer-CVAE维护版在明确离线任务上的最小复现准备；No-Go用于论文新算法开发和严格历史因果全链路复现的完成声明。** 本阶段不启动训练。不是强行宣布一个已获真实提升的正式论文主干；若“合格”要求Stage5.1严格历史上下文整链路，本轮没有完全合格对象，不能直接开训。

唯一保留对象：`{am['commit']}`，原ETH/UCY pre配置，所有原模型层和MSE/KL/sample目标不变。选择理由是来源/训练入口完整、维护版已隔离旧scene_orig未来值、单人与多人/先验/排列/反向/恢复及N57资源检查通过。PECNet不推荐为并行备选，旧TOHI/G0/G1/SocialVAE/采样路线继续关闭。

## 已冻结但未执行的最小方案

精确设计与SHA在configs/stage7/minimal_reproduction_plan.json，协议名HTraj-AgentFormer-A_offline_complete20-v1。依据Stage5.1冻结录像与时间支持，从原始五个非ETH文件生成新manifest，保留N1；同一监督目标集重新分组为同步完整20帧上下文。未来存在性用于离线资格，不使用GT坐标计算运动特征/连接，不冒称Causal8x12。单人/完整目标数如下：

'''+count_table+f'''
训练集排除整个验证录像，仅原来源train区域，旧validation不混训练；ETH封存。HOTEL、ZARA2两个曝光开发划分，不叫独立确认。需从原始数据重建packet，不沿用自定义模型缓存。数据source SHA/新manifest SHA已经保存；Stage7未计算新真实预测指标。

四run：原始CVAE与逐人独立同结构对照×两录像×seed17。均随机初始化，相同窗口/监督目标访问、每窗口一次优化（独立人loss先聚合），最多30epoch，前5轮包含在内；Adam1e-4、StepLR10/gamma0.5、原loss三项和K20，均不调参、不使用旧权重。逐人独立处理必须同时消除global scene中心及注意路径，不能只设对角mask后称无社会。

有效容量和单人KL clipping聚合与Full可能不同，scene中心也不同；前置审计必须说明，不能因名义参数相同宣布完全控制。这个对照足以检验个体预测能力，**不足以单独证明真实社会信息贡献**。若后续要求独立归因而不能控制这些差异，停止方法开发，不自动增加容量/Placebo实验。

同窗口/目标ID/GT重算CV-Last2、OLS4、OLS8，固定Last2为主，不事后逐录像挑CV。主指标K1 learned-prior-mean（不是recon、不是轨迹数学期望），选最低窗口ADE/早epoch，同checkpoint FDE；同ID噪声首样本、独立minADE20/minFDE20另报，所有联合候选生成完再对GT计算指标。窗口/录像内行人等权/录像宏平均全部报告，不将重叠窗口当独立样本。

开始前仍须通过真实数据入口smoke、np.int机械兼容、目标/GT/预算/损失聚合/有效容量审计；不默默运行原test.py的strict=False。串行RTX3080，N57保留显存已约9.66GB，遇OOM立即停止，不能删拥挤难例换可运行。当前没有真实吞吐，不能抄SocialVAE训练耗时估算AgentFormer整轮。

停止规则冻结：任何非有限/泄漏/目标错配/OOM即停该run不盲重试；30轮到期停，best=30标截断不延长。继续运动预测研究需两录像两权重相对Last2 ADE增益≥max(.005m,1%)、配对FDE退化≤max(.005m,1%)；独立社会研究还必须相对逐人对照达到同门槛且解决容量/中心混杂，否则不增加社会模块。K1不过但K20改善仅记录多模态条件收益，不自动转第五章。全部条件通过也只建议后续独立验证，不能宣称创新。

完整两阶段AgentFormer/DLow、官方权重真实性与官方原论文数值未复现；严格历史上下文适配未完成。以上是冻结的有限预算方案，不是当前训练授权或已完成结果。最终方向仍由下一条用户指令确定。
''')

decision=dict(sole_followup_backbone='AgentFormer-CVAE maintained original pre-stage',commit=am['commit'],
 component_technical_admission=True,offline_protocol_baseline_feasibility='conditional GO',
 strict_history_context_pipeline_admitted=False,official_end_to_end_reproduction_complete=False,
 new_paper_algorithm_development='NO-GO',new_algorithm_evidence=False,formal_training_started=False,
 official_pretrained_strictly_verified=False,other_new_candidate_selected=False,ETH_evaluated=False,
 reason='Model tests pass; offline context and strict causal context must stay separate; real paired prediction ability/independent interaction benefit not established',
 frozen_future_plan_sha256=sha256('configs/stage7/minimal_reproduction_plan.json'),automatic_push=False)
json_new(ROOT/'final_decision.json',decision)
save('STAGE7_REPORT.md',f'''# Stage7：论文技术主干决策

**唯一后续复现对象：AgentFormer-CVAE维护版；当前新算法开发No-Go。** 模型级技术准入通过不等于已完成真实预测性能复现。仅在明确的离线完整20帧任务上保留最小复现方案；严格历史因果loader未完成准入，不以它宣称已得到可部署因果模型。

{scope}
旧TOHI、G0/G1、SocialVAE社会改造、容量/Placebo与Stage6.1采样路线正式冻结，未改任何历史参数/指标/权重。Stage6.1正常运动改善与静止/另一录像退化不能构成跨场景成功，关闭结论不改。

AgentFormer冻结`{am['commit']}`，{agent['parameter_count']}参数。机械_LinearWithBias兼容、隔离deps后N1/2/5/12完整损失/梯度/恢复通过；保留learned prior的K1/K20换序恢复最大差4.77e-7，未来NaN污染差0。旧版future-centered归一化已在维护版修正，未使用旧paper成绩。N57原始K20反向分配约6.39GB、保留9.66GB，3080可运行但余量小。PECNet冻结`{pm['commit']}`，{pec['parameter_count']}参数，也通过人工结构测试，但SDD尺度/协议与GT终点先选再插值评价需更多重建，因此不作为第二主干。

原AgentFormer loader确实使用未来存在性筛上下文；独立原方法测试已记录，不能混称HTraj严格因果协议。已冻结新离线manifest（含单人）和同目标CV/逐人对照方案，详见05与minimal_reproduction_plan.json。没有正式训练，没有新算法或真实性能优势。未来一旦真实基线/对照/混杂/资源门槛失败，即停止，不新造模块延续旧路线。

研究问题仅暂留“训练后的模型是否受无关上下文变化影响”，人工随机权重远邻加入变化约{prior['distant_context_probe']['original_agent_prediction_max_difference_m']:.6f}m只表明计算依赖，不表明真实误差退化。特征归因/时空交互/概率预测已被公开工作覆盖；当前无经确认的新算法贡献或投稿证据。

六份报告01历史结论、02候选对比、03准入、04创新可行性、05最终决策及本总报告已生成。results/tables/stage7保存源码blob/SHA、3个首次下载失败与重试、原生import失败/兼容、首次4个索引测试失败与修复、实际模型检查、未来存在性反例、资源和协议记录。最终完整测试{tests}项通过；历史SHA最终复核见delivery_verification.json。原始外部源码/依赖/随机权重在忽略cache，Git只收包装器、元数据、测试和报告。

只本地commit，不自动push。用户自行：`git push -u origin feat/stage7-paper-baseline-decision`。完成后停止。
''')
print('Six Stage7 reports and explicit scoped Go/No-Go saved')
