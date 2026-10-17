"""Render evidence-linked Stage7.5 reports; no model training or test-set scoring."""
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from stage7p5_prepare import digest

OUT=ROOT/'results/tables/stage7p5'
REPORT=ROOT/'reports/stage7p5'


def table(headers,rows):
    def cell(x):
        if x is None:return '未定义/未取得'
        if isinstance(x,float):return f'{x:.6f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in rows])


def save(name,text):
    p=REPORT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text.strip()+'\n')


def main():
    g=json.load(open(OUT/'complete_group_statistics.json'))
    overlap=json.load(open(OUT/'cohort_overlap.json'))
    execution=json.load(open(OUT/'execution.json'))
    preservation=json.load(open(OUT/'history_preservation.json'))
    with gzip.open(OUT/'K_subset_curves.csv.gz','rt') as f:curves=list(csv.DictReader(f))
    with open(OUT/'head_imbalance_summary.csv') as f:imbalance=list(csv.DictReader(f))
    prefix='../../results/tables/stage7p5/'
    common=f'''
本报告仅分析Stage7.4保存的官方validation预测，不重新评价test。官方源码e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5，作者CVAE+DLow、mean=True、K20、8/12帧、0.4s、米制、完整20帧离线上下文资格均保持不变。协议在034727969d0892632889bf5e038c0fac5f892d53提交；原始文件数值列修复在350aceb提交后执行。未训练、未更新模型参数、未增加模块。

HOTEL/ZARA2是权重fold名称；validation分别来自其余7个来源录像的_val区域，并非对应held-out录像的test。两折共享3944个目标窗口，history/GT SHA完全相同，不能作为两个独立确认集。窗口加权、来源录像内行人ID等权和逐来源结果分别保存。相同行人可跨历史运动组出现，组内人数不可相加。旧test已经被观察过，未来不能称为完全未接触的独立确认集。

完整数值及有效方向样本数见[complete_group_statistics.json]({prefix}complete_group_statistics.json)，原始逐目标/逐head表见同目录。早期group_statistics.json可能漏掉首行未定义的方向列，已明确由complete版本替代；旧文件和失败记录保留。空值代表方向不可靠或无适用样本，不填0。
'''
    save('01_AGENTFORMER_ARCHITECTURE_AUDIT.md',f'''# AgentFormer实际信息路径审计
{common}
## 冻结实现，而非新模型

以[官方固定版本](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)为准。论文方法见[AgentFormer，§3及§4](https://arxiv.org/html/2103.14023v3)。论文与维护配置的具体差异以源码记录，不把2021文章描述等同所有后续默认值。

```mermaid
flowchart LR
  X[完整8帧多行人历史] --> O[历史均值scene_orig与scene_norm、位移vel]
  O --> C[ContextEncoder: 8N tokens]
  C --> H[逐行人时间均值 agent_context]
  H --> P[Learned Prior]
  H --> Q[DLow q_mlp与20个q_b输出]
  Q --> Z[z_k=b_k; mean=True]
  C --> D[原FutureDecoder自回归12步]
  Z --> D
  D --> Y[N × 20 × 12 × 2 米制位置]
  GT[训练未来GT] --> F[训练FutureEncoder后验]
  C --> F
```

源码定位均相对上述commit：`model/agentformer.py:103–168`、`171–256`、`259–427`、`515–583`；`model/agentformer_lib.py:278–330,377–378`；`model/dlow.py:92–137`；两折`cfg/eth_ucy/*/*_agentformer_pre.yml`。

1. 当前ContextEncoder输入是scene_norm和vel，线性映射4→256，8N个token进入2层、8头Transformer，FF维512；输出按每个行人跨8帧平均得到N×256。位置减去全部历史观测的场景均值；vel是每帧位移，诊断时除0.4才得到m/s。没有把未知速度伪造成新模块输入。
2. Agent-Aware Attention对同一行人token和其他行人token分别计算Q/K得分，再使用mask和softmax。当前默认`gaussian_kernel=False, sep_attn=True`。它并非未建模交互；可以通过学到的token表征隐式表达相对关系。
3. 两折配置未设置有限conn_dist，默认100000触发全连接mask。2m邻居只是本阶段的历史分组半径，绝不是模型原始attention半径。
4. 当前输入没有显式边特征`(x_j-x_i,v_j-v_i,closing)`。但“没有手工边特征”不等于“无法表达相对运动”，也不证明增加边模块有益。scene_orig本身依赖上下文成员，邻居删除同时改变坐标原点，干预会有混杂。
5. Learned Prior从历史agent_context生成潜变量分布；训练FutureEncoder使用有标签未来。DLow正式推理显式传入z=b，因而不是从该prior独立抽20次；q_A仍影响DLow训练KL，但mean=True前向的z不包含A乘噪声项。head0不是数学期望或最高概率候选。
6. FutureDecoder通过原跨agent attention使用历史与已预测未来token，自回归12步，不把未来GT作为推理输入。官方loader的完整20帧资格会利用未来存在性筛选上下文，本阶段接受并声明这一离线benchmark语义。

若N为行人数，历史attention约O((8N)²d)；K20解码包含12次增长的时空token attention及历史交叉attention。未新增参数或推理成本。本阶段不重新加载/推理作者模型，沿用已核验输出，来源字节SHA再核对。
''')
    motion=[]
    for fold in g:
        for name in ('stationary','low','normal','fast'):
            z=g[fold]['speed_group'][name];w=z['window'];p=z['person_equal']
            motion.append([fold,name,z['target_windows'],z['persons'],w['GT_speed_mps'],w['ADE_oracle_predicted_speed_mps'],w['ADE_oracle_speed_bias_mps'],w['ADE_oracle_ADE'],w['FDE_oracle_FDE'],w['ADE_oracle_ADE_over_GT_path_floor0.5'],p['ADE_oracle_ADE_over_GT_path_floor0.5']])
    source_speed=json.load(open(OUT/'source_by_speed_statistics.json'))
    local=[]
    for f in source_speed:
        for group in ('normal','fast'):
            z=source_speed[f]['biwi_eth_val'][group];w=z['window'];local.append([f,group,z['target_windows'],z['persons'],w['ADE_oracle_ADE'],w['ADE_oracle_ADE_over_GT_path_floor0.5'],w['ADE_oracle_speed_bias_mps']])
    turn=[]
    for f in g:
        for group,z in g[f]['turn_group'].items():
            w=z['window'];turn.append([f,group,z['target_windows'],w['ADE_oracle_ADE'],w['ADE_oracle_ADE_over_GT_path_floor0.5'],w.get('ADE_oracle_turn_magnitude_bias_rad')])
    save('02_MOTION_ERROR_ANALYSIS.md',f'''# 运动误差：区分尺度、候选口径与局部偏差
{common}
## 定义

历史分组仅用7段历史步速的均值：静止<0.05、低速[0.05,0.4)、正常[0.4,1.8)、高速≥1.8m/s。未来平均速度包含最后观测到首预测的步段，总计12段/4.8s。方向由最后观测到终点的位移定义，位移<0.24m时留空；历史方向用OLS4，速度<0.05m/s时留空。

速度偏差=预测平均速度−真实平均速度。终点沿向/横向误差相对历史OLS4方向计算。角度差使用atan2(cross,dot)，单位rad。方向列只对有效样本统计；圆周角跨±π的普通均值不能当全局航向偏置，结论优先依据绝对角误差及来源分解。

主要尺度控制为ADE/max(真实未来路程,0.5m)，并保留0.25/1m floor敏感性；FDE/真实净位移及历史隐含未来路程也保留。未来路程只作为结果尺度，不用于分组、模型输入或选候选。低速/往返目标净位移小，归一化值对floor敏感，不当作新统一排行榜。

## 四类历史运动结果

以下速度、ADE来自**GT选出的整条ADE最佳候选**；FDE列单独取FDE最佳，二者不拼成同一候选。oracle不是在线选择能力。逐head及20head等权诊断另存；后者也不是预测分布期望。

{table(['fold','历史组','窗口目标','行人ID','GT速度','oracle速度','速度偏差','minADE20','minFDE20','路程归一化ADE/窗口','路程归一化ADE/行人'],motion)}

## 高速假设的正反证据

汇总窗口口径下，高速归一化ADE约0.066/0.067，正常约0.071/0.071，不能仅凭原始ADE就宣称高速结构缺陷。但同来源分解显示：

{table(['fold','biwi_eth_val组','窗口','行人ID','ADE','归一化ADE','速度偏差'],local)}

高速51个窗口全部来自biwi_eth_val的同21人，且两折完全重合。该局部群体归一化误差仍大于同录像正常组，速度偏低，不能全归因于位移尺度；正常组也仅5人，两组构成不同。连续log(ADE+0.01)回归同时控制log路程及来源，仍留有正的高速系数，但类别与来源高度绑定、缺乏其它录像高速支持，不能外推跨录像规律。见[scale_adjusted_regressions.json]({prefix}scale_adjusted_regressions.json)。

20head等权高速速度偏差约−0.696/−0.675m/s，与oracle的−0.242/−0.219不同；这描述候选集合而不是一个已部署点预测器。整体验证oracle速度偏差仅−0.021/−0.022m/s，20head等权整体约+0.020/−0.001m/s；没有验证统一“所有运动状态都偏慢/偏快”的说法。历史静止组等权候选偏快而oracle偏差小，可能包含启动模式，未用未来静止标签证明这些候选无效。

## 转向与方向

{table(['fold','仅历史转向组','窗口','ADE','路程归一化ADE','oracle转向幅度偏差rad'],turn)}

历史转向组相对误差高于直行组，但转向幅度偏差在两折约+0.012/−0.003rad，未发现一致“转向不足”。正常运动组oracle绝对方向误差约0.093/0.095rad；高速约0.064/0.080rad。不能据此宣布方向系统性错误。低速/静止方向有效目标比例小，不能把未定义方向算成0。

逐时间步位置误差见timestep_errors.csv.gz，整条ADE/FDE赢家固定，不逐时间步换head。motion_speed_transitions.csv.gz保存真实/预测未来均速相对历史均速变化。person_cluster_intervals.json为来源内行人cluster重采样200次的**行人等权**诊断区间，不是窗口IID显著性检验，也不是新录像泛化置信区间。
''')
    kt=[]
    for fold in g:
        for k in (1,2,5,10,20):
            rr=[r for r in curves if r['fold']==fold and r['group_field']=='all' and int(r['K'])==k]
            p=[r for r in rr if int(r['order_id'])==0][0];a=[r for r in rr if int(r['order_id'])>0]
            kt.append([fold,k,float(p['window_ADE']),float(p['window_FDE']),float(np.mean([float(v['window_ADE']) for v in a])),float(np.mean([float(v['window_FDE']) for v in a]))])
    it=[]
    for r in imbalance:
        if r['group_field']=='all' and r['weight']=='window':it.append([r['fold'],r['metric'],r['top_head'],float(r['top_fraction']),float(r['top4_fraction']),float(r['entropy_effective_heads'])])
    du=[]
    for f in g:
        for group,z in g[f]['speed_group'].items():
            w=z['window'];du.append([f,group,w['near_pair_fraction_tight'],w['near_pair_fraction'],w['near_pair_fraction_loose'],w['endpoint_pair_mean_m']])
    save('03_MULTIMODAL_CANDIDATE_ANALYSIS.md',f'''# 原始DLow K20候选结构
{common}
## 事先固定的K子集

0347279提交已保存原head顺序和seed7501生成的20套排列；每套取前K=1/2/5/10/20，嵌套、不根据GT挑组合。随机顺序平均只对这20套固定顺序有效，不是全组合的精确期望。所有min误差均为GT oracle覆盖，非部署候选选择。

{table(['fold','K','原顺序ADE','原顺序FDE','固定排列平均ADE','固定排列平均FDE'],kt)}

固定排列平均K10→20使ADE约下降26.5%/26.9%，FDE约下降34.1%/35.1%。未发现“很快饱和”的支持证据；这也不表示K>20一定有益，本阶段没新增候选。head0在HOTEL很差，换固定顺序差别大；没有概率排序的20个head不能把前若干个当作最有可能的候选。

## 胜出频率

以1e-8范围内并列平分贡献，避免仅argmin索引偏好。head为0起始编号。有效head数=exp(胜出分布熵)，只是频率摘要，不是独立模式的估计量。

{table(['fold','赢家指标','最高head','最高频率','最高4个总频率','熵有效head数'],it)}

HOTEL最高head19约18%，ZARA2最高head9约11%；最优贡献并非跨fold相同的少数head。逐来源、历史运动组及行人等权频率均在head_contributions.json及head_imbalance_summary.csv。不同head无等频训练约束，胜出频率不均本身不证明训练缺陷，不能用这些GT频率在线排序。

## 相似性及终点分布

预设近重复判据为两候选平均路径距离<0.2m且终点距离<0.3m；另报告阈值半倍/双倍。共190对/目标，比例不等于重复目标数，也不等于聚类模式数。

{table(['fold','历史组','紧阈值比例','主阈值比例','松阈值比例','平均终点两两距离m'],du)}

全体主阈值比例约1.21%/1.15%，未发现“大量重复模式”。重复较集中于低速/静止组；两折整体终点平均两两距离约2.086/1.978m。宽终点分布不能证明概率校准正确，也不能单凭静止历史的分散候选认定模型错误。所有目标的RMS、协方差有效秩、路径两两距离均已保存，可复核绘图。

本阶段没有评估联合场景oracle、碰撞率或校准度；每个人独立选最佳head未必构成同一联合未来。这一评价缺口已被[Joint Metrics Matter，ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html)专门研究，不能改名作为本项目创新。
''')
    density=[];sources=[]
    for f in g:
        for group,z in g[f]['density_group'].items():
            w=z['window'];density.append([f,group,z['target_windows'],z['persons'],w['ADE_oracle_ADE'],w['FDE_oracle_FDE'],w['ADE_oracle_ADE_over_GT_path_floor0.5']])
        for seq,z in g[f]['sequence'].items():sources.append([f,seq,z['target_windows'],z['persons'],z['window']['ADE_oracle_ADE'],z['person_equal']['ADE_oracle_ADE']])
    save('04_SOCIAL_INTERACTION_DIAGNOSIS.md',f'''# 社会信息：结构事实与描述性证据
{common}
## 历史量定义

对原始_val数据仅取当前及之前8帧。邻居资格是历史8帧完整，2m半径由末观测位置确定；相对速度=(邻居末两帧速度−目标末两帧速度)，接近速度=−r·v_rel/max(||r||,1e-8)。接近阈值0.1m/s只作历史标志。无邻居时相关距离/速度留空。

该集合可能包含官方完整20帧上下文未纳入的行人。我们没有把这些额外历史邻居送入模型；这些数值是可观测场景描述，不能宣称模型确实attention到每一个这样的邻居。模型上下文人数另列context_people，绝不与2m邻居数量混用。

{table(['fold','历史2m邻居','窗口目标','行人ID','ADE','FDE','路程归一化ADE'],density)}

没有10+邻居组样本，不能填0或推断超密集场景。原始ADE随密度下降，而归一化后并非单调，说明运动构成/位移尺度会混杂“社会效果”。行人ID等权Spearman：邻居数量与ADE约−0.106/−0.067，与归一化ADE却约+0.169/+0.174。相对速度与原始ADE约+0.416/+0.330，归一化后约−0.021/−0.0003；接近速度与归一化ADE约+0.047/+0.110。这些仅为描述相关，不能当作关系内容的因果贡献。

## 逐来源敏感性

{table(['权重fold','验证来源','窗口目标','行人ID','窗口ADE','行人等权ADE'],sources)}

全部逐来源相关在history_error_associations.csv；不同来源运动状态与人数构成不同，没有窗口级IID p值。两折宏平均也不能消除共享3944个目标的依赖。

## 三个证据层次

- 描述性相关：本阶段实际执行，包含人数、距离、相对运动及误差。
- 固定权重敏感性：本阶段未新增邻居干预或模型推理，未以预测变化证明有益/有害。
- 真实因果收益：没有新从头训练的匹配NoSocial、容量对照与合法负对照，无法判断AgentFormer社会信息的独立价值。不能把SocialVAE旧结论迁移给AgentFormer。

当前agent_mask默认全连接，没有显式接近趋势边输入；但已有attention可隐式编码关系。增加相对运动特征是否有独立价值仍待验证。[SocialCircle，§3](https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf)已研究速度/距离/角度表示；[Inter-GraphFormer](https://doi.org/10.1016/j.physa.2026.131566)公开方法概要亦涉及有向稀疏交互。不能因原模型无手工边输入就宣布新的社会GNN必要。
''')
    papers=[
        dict(name='AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting',year=2021,venue='ICCV',paper='https://arxiv.org/html/2103.14023v3',code='https://github.com/Khrylx/AgentFormer',commit='e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5',read='全文§3/4与固定维护源码',innovation='agent-aware时空attention；CVAE联合潜在意图；DLow多候选',inputs='多行人历史；ETH/UCY无地图',outputs='K条多行人未来轨迹',objective='CVAE重建/KL/sample；DLow重建/KL/diversity',datasets='ETH/UCY、nuScenes',metrics='ETH/UCY 8/12帧、K20 marginal minADE/FDE；nuScenes另有K1/5/10',relation='本项目冻结基础',overlap='DLow条件化与多样性思想已存在，不能重新命名',anchors='paper§3.1/3.2/4；model/agentformer.py、dlow.py'),
        dict(name='Trajectory Unified Transformer for Pedestrian Trajectory Prediction (TUTR)',year=2023,venue='ICCV',paper='https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf',code='https://github.com/lssiair/TUTR',commit='655a756af8e961598a1a22a07cb48eb8534c722a',read='全文§3.2/4.1与model.py/train.py',innovation='训练轨迹归一化聚类mode；mode/social两级Transformer；轨迹与概率双输出',inputs='目标和邻居历史、来源训练集mode中心',outputs='轨迹与概率；推理按概率TopK',objective='最近训练mode监督的Huber回归+软标签交叉熵',datasets='ETH/UCY、SDD',metrics='8/12帧；best-of20 ADE/FDE与brier指标',relation='解决未排序多候选/后处理开销，相近问题而非AgentFormer简单增层',overlap='普通概率评分头与归一化mode想法已覆盖',anchors='paper§3.2 Eq1–6/§4.1；model.py:37–75；train.py:248–260有test选best需审慎'),
        dict(name='SocialTrans: Transformer based social intentions interaction for pedestrian trajectory prediction',year=2025,venue='Physica A 663:130435',paper='https://doi.org/10.1016/j.physa.2025.130435',code='https://github.com/XiaodZhao/SocialTrans',commit='未取得',read='出版社方法/实验概要；全文未取得，不能称精读完成',innovation='概要：SIE/NPIE并行意图状态、终点优化及聚类',inputs='概要：目标与邻居观测运动；准确特征/资格待核实',outputs='多未来轨迹；精确候选数待核实',objective='完整公式/权重待核实',datasets='ETH/UCY、SDD',metrics='ADE/FDE；K、筛选、选模协议待全文核实',relation='目标/邻居意图协同、终点修正为相近思路',overlap='尚未核实，不能作为不存在某机制的证据',anchors='publisher Proposed approach/Experiments；OpenReview PDF遇浏览器验证，不绕过'),
        dict(name='MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation',year=2025,venue='CVPR',paper='https://arxiv.org/html/2503.09950v2',code='https://github.com/DSL-Lab/MoFlow',commit='f1b89b0e80ce95646214923e271e3c280d305cfa',read='全文§4/5与官方README/模型源码静态检查',innovation='相关K-shot flow matching回归+分类；IMLE/Chamfer一步蒸馏',inputs='历史多行人位置/相对运动上下文、噪声',outputs='K个相关未来；一步student',objective='最佳候选回归+CE；student匹配teacher样本的Chamfer',datasets='ETH/UCY、SDD、NBA',metrics='ETH/UCY 8/12、min20ADE/FDE；NBA10/20帧；SDD像素',relation='多候选多样性/推理效率相近；生成机制不同',overlap='候选覆盖、相关头和概率辅助已覆盖；本阶段不采用flow/扩散',anchors='paper§4 Eq4.3/4.4,§5.1/5.2；README明确ETH/UCY有LED与original不同版本'),
        dict(name='Inter-GraphFormer: A spatial–temporal graph transformer-based method for pedestrian trajectory prediction',year=2026,venue='Physica A 694:131566 (July2026)',paper='https://doi.org/10.1016/j.physa.2026.131566',code='未定位可验证作者源码',commit='未取得',read='CrossRef身份及出版社Introduction/方法概要；全文未取得',innovation='概要：稀疏有向图、并行时空Transformer、dual-GLU、CVAE',inputs='历史运动与社会上下文；精确状态/资格待核实',outputs='多个未来轨迹',objective='CVAE提及；精确loss/权重待核实',datasets='ETH/UCY、SDD',metrics='完整8/12、K及选模细节未核实，不推定',relation='显式运动/非对称交互是相近思路',overlap='相近思路已公开；与具体草案是否相同尚未核实',anchors='publisher Introduction贡献段；CrossRef DOI元数据；text-mining全文接口429'),
        dict(name='Joint Metrics Matter: A Better Standard for Trajectory Forecasting',year=2023,venue='ICCV',paper='https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf',code='https://github.com/ericaweng/joint-metrics-matter',commit='894182b5ea7dac3c93bbe0bcc18c156c932398b7',read='全文§3/4/5与官方README',innovation='JADE/JFDE及collision系统评价；Joint AgentFormer训练项',inputs='历史多行人；联合候选',outputs='原预测器联合轨迹',objective='原AgentFormer基础上增加joint sample reconstruction',datasets='ETH/UCY、TrajNet SDD',metrics='marginal与joint K20、collision；8/12、stride1、包含单人',relation='直接改进AgentFormer联合度量',overlap='仅增加联合oracle指标/损失属于高度重合工作',anchors='paper§3/§4.1/§5；README标准保存格式与数据差异说明'),
        dict(name='SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction',year=2024,venue='CVPR',paper='https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf',code='https://github.com/cocoon2wong/SocialCircle',commit='383ff0671f011806a8da9422d4a3839206847b82',read='全文§3/4；作者仓库身份/版本',innovation='角分区聚合速度(历史净位移)、距离、方向，作为虚拟时间序列接入backbone',inputs='历史目标/邻居运动',outputs='沿用所接backbone的K未来',objective='沿用backbone原loss，不新增统一loss',datasets='ETH/UCY、SDD',metrics='8/12帧、minADE20/FDE20；ETH/UCY米、SDD像素',relation='显式相对运动表示近邻先例',overlap='直接添加距离/速度/角度社会特征不是新机制',anchors='paper§3 Eq1–11,§4；官方仓库TensorFlow与beta PyTorch版本须分开'),
        dict(name='Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting',year=2025,venue='CVPR',paper='https://openaccess.thecvf.com/content/CVPR2025/papers/Messaoud_Towards_Generalizable_Trajectory_Prediction_using_Dual-Level_Representation_Learning_and_Adaptive_CVPR_2025_paper.pdf',code='本次未定位作者独立实现，不声称已复现',commit='未取得',read='全文§3/4',innovation='PerReg+自蒸馏/段重建、register mode queries与冻结主干prompt适应',inputs='车辆历史与车道地图',outputs='多模态GMM未来轨迹及概率',objective='自蒸馏+GMM+重建，多任务权重',datasets='nuScenes、AV2、WOMD',metrics='2s历史/6s未来、K6、minADE/FDE/B-FDE等',relation='跨域运动泛化相关，但属于不同输入/协议问题',overlap='泛化/模式query已被研究，不能直接迁移其指标到行人8/12',anchors='paper§3.1–3.5/§4.1；车辆/地图输入限制')]
    json_new(OUT/'literature_comparison_matrix.json',papers)
    csv_new(OUT/'literature_comparison_matrix.csv',papers)
    sections=[]
    for p in papers:
        sections.append(f'''### {p['name']}

{p['year']}，{p['venue']}。[论文]({p['paper']})；作者代码：{p['code']}；版本`{p['commit']}`。阅读范围：{p['read']}。

- 方法：{p['innovation']}。
- 输入/输出：{p['inputs']} → {p['outputs']}。
- 训练与评价：{p['objective']}；{p['datasets']}；{p['metrics']}。
- 与本项目：{p['relation']}。重合判定：{p['overlap']}。
- 证据定位：{p['anchors']}。
''')
    save('05_RECENT_LITERATURE_REVIEW.md','''# 2023—2026方法与协议核查

检索日期2026-10-17。使用nature-academic-search多源流程及nature-downloader开放全文核验；这是针对本轮问题的可追溯检索，不是穷尽所有文献的系统综述。学术MCP未挂载；OpenAlex备用脚本遇HTTP429，CrossRef与出版社/作者/arXiv/CVF互补。GitHub API限流，改用公开git ls-remote冻结版本并读取固定commit文件。所有失败均留档，不把检索失败等同论文/代码不存在。

AgentFormer、TUTR、MoFlow、Joint Metrics Matter、SocialCircle及PerReg+已取得合法PDF并读方法和实验，SHA在literature_sources.json。SocialTrans与Inter-GraphFormer身份已定位，但全文审阅未完成：出版社入口403；OpenReview请求浏览器验证，不绕过；Inter-GraphFormer出版社声明的文本接口429。未取得个人图书馆会话或作者全文，相关loss/协议保留待核实。**因此指定五篇中有两篇未达到完整方法精读，不能宣布新颖性检索全部通过。**

分类严格区分：原方法直接覆盖、相近思路、不同任务、尚未核实。相近思路并不证明技术完全相同，缺少公式也不证明不同。

'''+'\n'.join(sections)+'''
## 与当前证据最接近的先例

TUTR §3.2已将训练mode中心、概率和轨迹联合建模：训练用GT形成监督标签是正常监督，不是部署oracle；在线TopK来自模型概率。官方train.py却逐轮调用名为test的数据loader并按ADE+FDE保存best，不能把这种入口直接移植为本项目严格validation选模方式。

MoFlow §4.2/4.3直接研究相关多候选及覆盖/多样性，代码声明ETH/UCY存在两种预处理版本，不能仅因同为8/12、K20就直接公平排名。Joint Metrics Matter已经把marginal最优不等于joint最优作为核心问题；只把该指标加进HTraj不构成新算法。

SocialCircle的“速度”meta分量实际用历史端点净位移长度，和本阶段平均相邻步速不完全相同。可区别具体定义，不能将常规运动特征不同定义本身宣布为创新。SocialTrans与Inter-GraphFormer的未核实项仍是新颖性判断风险，不由摘要补全。

本报告不使用任何这些论文的test数值来挑选本项目改进，不新建跨方法性能排行榜。
''')
    save('06_INNOVATION_CANDIDATES.md',f'''# 创新候选筛选：没有批准进入训练的方案
{common}
## 一项被暂缓的最小草案，而非新论文方法

**历史速度摘要条件化原DLow采样器**仅列为可证伪草案，最终未通过GO；没有实施，也没有命名新网络。不存在第二项为了凑数的候选。

1. 具体问题：biwi_eth_val高速群体的候选速度覆盖偏低，且同来源相对误差高于正常组；但全来源归一化并未验证普遍高速劣势。
2. 数据证据：每折相同51窗口/21人；ADE oracle速度偏差−0.242/−0.219m/s；同来源归一化ADE约0.066/0.067，对比正常0.037/0.035。详见运动报告。它是条件性现象，并非跨录像重复。
3. 先例：原DLow已条件化历史context并有多样性loss；TUTR用运动mode及概率，MoFlow生成相关多候选，SocialCircle显式运动特征。未取得两篇指定论文全文，不能断言不存在相同机制。
4. 最小技术区别设想：仅将原q_mlp输入c_i扩展为[c_i, log(1+s_i/1m/s), std(step_speed_i)/1m/s]，得到原20个b_i,k，保留原CVAE/decoder。s及std只用8帧历史。不新增attention、GNN、gate或评分头。**这很可能只是常规特征增强，现阶段不具备可信论文创新性。**
5. 涉及模块：若将来另行批准，只改DLow q_mlp首层256→258，其余模块不变；当前官方源码和权重均未改。
6. loss：设想不改原DLow重建、KL、diversity目标；GT仍只用于原训练监督，不在部署选head。
7. 成本：首层512单元，理论增加2×512=1024参数，相对DLow591872约0.173%；计算增量预期小于1%但未实测。仍需完整DLow重训，HOTEL50轮仅按Stage7.4实测307.35s/epoch约4.27GPU小时，加validation/保存开销；其它fold不能假定相同。没有运行任何候选训练。
8. 公平消融：冻结同一CVAE，原DLow、仅速度、仅方差、两者；近容量对照保持256输入而首隐藏512→514，增加1026参数（函数类仍非完全等价）。相同train池/seed/预算/validation规则；不按已见test误差选方案。必须有其它来源高速支持，不能将同51窗口重复算两次。
9. 否定条件：若尺度/来源控制后现象消失、其它录像不重复、改进只依赖GT oracle后选子集、原DLow同预算追平、容量对照解释收益、正常/静止组明显退化，或完整文献已实现同样机制，则否定。将来若验证，建议预注册每来源minADE至少下降max(0.005m,1%)且minFDE退化≤max(0.005m,1%)，行人等权同方向；这是尚未执行的计划门槛，绝非当前提升。
10. SCI三区/四区审稿风险：手工特征缺少新机制；单来源和小样本；GT best-of-K选择偏差；作者validation重复使用；近容量函数类不同；未证明部署效果；缺少严格多seed/来源外验证；两篇全文缺失。投稿分区不降低这些要求，也不构成可接收保证。

## 为什么不启动

局部速度信号存在，但机制新颖性不足、跨来源证据不足。候选重复及快速饱和假设未得到支持；社会相关性也不支持立刻加入交互模块。因此推荐训练方案数为0，当前未找到值得启动训练的创新方案。
''')
    save('07_FINAL_RESEARCH_DECISION.md',f'''# 最终科研决策：NO-GO（创新开发）
{common}
{table(['准入条件','本轮证据','判断'],[
 ['具体问题','同一biwi_eth_val高速群体速度偏低；历史转向组相对误差高','局部现象成立，不等于普遍缺陷'],
 ['排除纯尺度解释','整体高速归一化不劣；同来源高速相对误差仍较高；回归仍有残差','部分支持，但来源/人群混杂未排除'],
 ['候选重复/饱和','近重复对约1.2%；K10到K20仍约26%-27%ADE收益','不支持作为当前改进动机'],
 ['独立社会收益','仅观察相关，无匹配训练消融；关系量与误差相关受尺度影响','无法判断'],
 ['文献未覆盖机制','原DLow已有历史条件化；TUTR/MoFlow/SC有相近思想；两篇全文未取得','未通过'],
 ['最小可实施新机制','有q_mlp输入小改草案，但更像特征工程','工程可行，不代表创新成立'],
 ['对照/否定条件','草案列明同预算、近容量、来源及失败规则','已设计，不执行'],
 ['最终','没有方案同时通过问题、区别与证据门槛','NO-GO']])}

NO-GO针对启动新创新模型训练，不否定已经建立的AgentFormer公开基线。SSD空间恢复只是工程条件改善，不能替代科研证据。

## 下一步替代建议（本轮不执行）

优先补齐SocialTrans/Inter-GraphFormer合法作者全文与公式核查，再判断是否有机制差异；如果仍只是显式速度特征、常规候选概率或普通相对边编码，应停止包装为新方法。若用户决定继续问题核查，可另行冻结来源级验证计划，确认局部速度/转向现象是否在不同录像支持；不得把当前两个高度重叠fold当两次独立成功。

若要研究联合场景一致性，必须先承认Joint Metrics Matter已直接解决部分相同问题，重新找可验证不足，不能复用其joint损失作为“创新”。不建议现在增加GNN、Transformer层、gate、评分头或扩散模型。

没有提出或验证任何实验提升；没有新增正式训练、没有重评test、没有自动push。Stage1—7.4旧文件保护核验{preservation['total_SHA_checks']}项SHA检查通过；完整单元测试433项通过。最终结果交付后停止。
''')
    save('STAGE7P5_REPORT.md',f'''# Stage7.5 AgentFormer缺陷与创新可行性总报告

**结论：本轮创新开发NO-GO。当前未找到值得启动训练的创新方案。** 保留可靠AgentFormer基线；本轮没有训练新模型、没有修改官方模型/权重、没有重算test成绩。
{common}
## 真实执行范围

重读并核验Stage7.4全部1278个validation预测文件，HOTEL 688窗口/5203目标、249个来源行人ID；ZARA2 590窗口/4262目标、237个来源行人ID。共有3944目标窗口；使用7个来源录像分解，不能以9465个窗口目标宣称9465个独立行人。CPU统计实测{execution['duration_seconds']:.2f}秒，本阶段未使用GPU执行正式推理或训练。模型人工前向仅限完整单元测试。

## 三个研究问题

- 运动：整体高速路程归一化ADE未劣于正常组；同一biwi_eth_val中高速仍有局部相对误差和速度偏低。全部高速只有51个共享窗口/21人，不能泛化。没有发现跨运动状态统一速度偏差，也未验证一致转向不足。
- 多候选：近重复对约1.21%/1.15%；K10到K20仍带来约26.5%/26.9%的固定排列平均ADE下降。最高head胜出占比约18%/11%，不是相同少数head垄断。oracle覆盖不是在线排序/选择能力。
- 社会：原网络已有全时空agent-aware attention与历史条件化DLow；没有手工相对边特征不构成新模块必要性。邻居统计只有描述性相关，独立社会贡献无法判断。

## 文献与候选

核查8篇，6篇取得全文并阅读方法/实验；SocialTrans2025、Inter-GraphFormer2026已核实身份和公开方法概要，但完整公式/协议未取得，不能冒充精读完成。全文/接口受限均已留档。TUTR、MoFlow、SocialCircle、Joint Metrics Matter已覆盖候选概率、多样性、显式社会特征、联合度量等相近方向。

最多列出1项未获批准的q_mlp历史速度摘要草案；它工程改动小，但更像普通特征增强，来源证据和新颖性不足，推荐启动训练的方案数为0。

## 交付索引

1. 01_AGENTFORMER_ARCHITECTURE_AUDIT.md：原始信息流、配置、源码定位。
2. 02_MOTION_ERROR_ANALYSIS.md：速度/转向/尺度及来源敏感性。
3. 03_MULTIMODAL_CANDIDATE_ANALYSIS.md：固定K子集、head贡献、相似性。
4. 04_SOCIAL_INTERACTION_DIAGNOSIS.md：历史关系描述与归因边界。
5. 05_RECENT_LITERATURE_REVIEW.md：论文方法、输入输出、loss、协议和重合风险。
6. 06_INNOVATION_CANDIDATES.md：1项暂缓草案、消融及否定条件。
7. 07_FINAL_RESEARCH_DECISION.md：NO-GO和替代建议。

结果位于[results/tables/stage7p5/]({prefix})：逐目标/逐head/逐时间步误差，历史组、来源、行人等权，固定K曲线，文献矩阵，来源/权重SHA及全部失败与测试记录。canonical统计为complete_group_statistics.json。图表数据均可从这些真实表复核，本轮未制作虚构或示意性能图。

保护检查{preservation['total_SHA_checks']}次SHA通过，其中12924项旧文件保护、73656个旧test输出仅按不透明字节验SHA，不解析旧test误差。单元测试433通过。初次数值列解析及可空字段导出失败均修复并保留现场，不改既定组、阈值、模型或历史指标。

分支feat/stage7p5-agentformer-innovation-study。本地提交后停止，不自动push。用户可执行：

```bash
git push -u origin feat/stage7p5-agentformer-innovation-study
```
''')


if __name__=='__main__':main()
