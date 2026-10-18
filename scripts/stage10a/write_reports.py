"""Generate evidence-linked Stage10A reports; no model invocation."""
from common import *


def save(name,text):
    path=ROOT/'reports/stage10a'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(text.strip()+'\n')


def main():
    data=json.loads((OUT/'data_audit.json').read_text())
    res=json.loads((OUT/'resources.json').read_text())
    sel=json.loads((OUT/'selection_summary.json').read_text())
    save('01_HISTORY_AUDIT.md',r'''
# Stage10A：历史研究审计

本轮基于已推送Stage9 `d8e753f718647db49202da3ac6d43e4b37cfc59b`；本地与origin无历史差异，工作区原本干净。冻结旧结论，重新授权可控图关系实验并不倒改旧阶段NO-GO。不把失败解释为网络“不够复杂”。

## 已完成过的动态图与失败边界

|证据|已做的结构/实验|真实结论与限制|
|---|---|---|
|[Stage2.6](../stage2p6/STAGE2P6_REPORT.md)、`src/models/temporal_graph.py:select_candidates`|4m/8邻居，历史相对位置/相对速度、有限4.8s CPA、按`DCA+.05*distance`排序；MLP社会权重；Dynamic-GNN-RN、独立H1/H2/H3及容量控制|3seed，1024训练/96开发窗口，30轮；Dynamic-RN ADE .440075±.002073，PedBase .439398±.003578，未建立稳定ADE增益。H3弱于H2；干预改变预测说明计算依赖，不能证明独立泛化价值。此处数值仅引用旧任务，不是新任务成绩。|
|[Stage3](../stage3/STAGE3_REPORT.md)、`src/models/relation_routing.py`|分离receiver/carry连续性，未知速度标记、位置回退；逐帧关系、关系GRU、均匀/随机关系控制|动态关系历史、缺失运动处理、关系门控均非本轮首次出现。旧种子/小预算/重用开发集结论不外推全部社会模型。|
|[Stage4.2最终审计](../stage4p2_final/FINAL_AUDIT_REPORT.md)|消除节点索引卷积，匹配逐人时间归一化PTN；G0/G1同预算从30延长到100轮，配对三CV|ZARA2 G0/G1 ADE .514221/.605474，CV-Last2 .325740；HOTEL .980086/.933634，CV-OLS8 .241865。两个神经模型均未超过三CV，G1只HOTEL改善G0。不是所有GNN的否定。|
|[PTN审查](../stage4p2_final/04_PTN_SENSITIVITY_AUDIT.md)|全体目标、固定权重1mm扰动，静止/低速/正常运动分组|局部PTN高增益及静止目标重复敏感性成立；G0也有，不能归咎社会边或证明PTN唯一因果责任。窗口重复会放大少数长轨迹影响。|
|[Stage5.1](../stage5p1/01_CAUSAL_DATA_PROTOCOL.md)、[原始输入反事实](../stage5p1/02_RAW_INPUT_LEAKAGE_AUDIT.md)|严格8帧历史；后向速度差分、显式有效性；未来邻居padding；训练/验证录像隔离|旧官方SocialVAE loader首出现速度补值确有未来数值污染反例；新B适配阻断该路径。未来监督目标资格与输入泄漏必须分开。|
|[Stage6](../stage6/STAGE6_REPORT.md)|Full/Placebo多seed结果、HOTEL运动误差、SGCN代码检查|Full社会内容优势未跨录像/seed通过；HOTEL持续运动的预测速度偏低是描述性证据，不证明唯一原因。SGCN空间mask索引问题成立，TXP节点卷积猜测已被纠正。|
|[Stage7](../stage7/STAGE7_REPORT.md)、后续AgentFormer协议|换序、梯度、单/多人和完整20帧上下文审查|官方离线资格可依赖未来存在性，不能冒充严格历史协议。排列等变与准入测试是工程性质，不是创新。AgentFormer公开基准成功复现继续保留，本轮不读取其test误差。|
|[Stage8](../stage8/STAGE8_REPORT.md)|Resonance/MoFlow协议、配对歧义及公开研究查重|有表示歧义反例，但没有预测损害与新颖性证据；当时NO-GO，不改写。|
|[Stage9](../stage9/08_FINAL_RESEARCH_DECISION.md)、07容量/机制审计|A原Re、B同容量联合MLP、C位置—运动交叉统计，各200轮，seed42；离线Complete20|C对B窗口ADE改善.001763m，小于.005m门槛；行人等权ADE/FDE退化.006571/.030109m；歧义组仅.00002085m收益。增加配对表达能力并未通过机制/预测门槛。不是“所有位置运动关系无价值”的证明。|

## 本轮实质差异与重复部分

重复部分：相对速度、CPA/DCA、风险排序、动态图、GRU、消息MLP、残差/门控/稀疏关系都不是新的原理。尤其旧TOHI已实现CPA选边，不能称本轮第一次从距离切换到风险。

本轮重新设计的是**归因条件**：新独立命名空间、严格历史完整目标协议、平移等变的速度GRU、无PTN/节点卷积、单跳当前ego消息、无学习关系门控；D/R候选池/每接收者边数/消息特征/参数完全相同，只有排序规则不同。S是单独静态时间规则控制，E有效容量较小。本轮不是复活高阶/关系GRU，也不能把这些工程修订本身当创新。

尚未验证：在该特定受控骨干和Causal8x12任务内，DCA风险排序能否比距离排序提升同目标预测；是否超过强CV及容量控制；跨seed/录像是否稳定；风险较高是否对应真正行为影响。这些均需未来独立授权训练，当前没有性能证据。本轮风险是恒速几何代理，不是行为意图、真实碰撞概率或社会影响因果标签。
''')
    table='|留出录像|split|窗口/有监督窗口|目标窗口|source×person ID|每人最多窗口|上下文N最大|\n|---|---|---|---|---|---|---|\n'
    for r in data['summary']:
        table+=f"|{r['fold']}|{r['split']}|{r['windows']}/{r['supervised_windows']}|{r['targets']}|{r['source_person_IDs']}|{r['max_windows_per_person']}|{int(r['context_N_quantiles'][-1])}|\n"
    save('02_DATA_PROTOCOL.md',r'''
# HTraj-RiskGraph-Causal8x12-v1：冻结数据契约

监督目标集合逐项复用Stage5.1的`HTraj-SocialVAE-Causal8x12-v1`，模型输入接口独立。8历史/12未来，0.4s（2.5Hz）、米制，神经输出K1确定性回归。本轮没有训练，也没有新神经模型验证ADE/FDE。未来若做K20基准须重新立项，不能拿本轮K1框架与公开minADE20直接排名。

唯一manifest为`configs/stage5p1/manifests/{biwi_hotel,crowds_zara02}/B.json`，hash在`results/tables/stage10a/data_audit.json`。它们来自Stage3.5录像隔离及旧时间支持区域：训练排除验证录像全部时段，其他录像只用旧train区域；验证只用被留出录像旧validation区域，未偷加旧validation到train。每窗口完整20帧支持为start..start+190 tick，历史为start..start+70，每步10tick。物理0.4s来自既有数据约定，不由frame数字自行推算。

HOTEL/ZARA2已经多次开发曝光，不是新的独立确认数据。ETH本阶段继续封存；历史官方test文件仅作为保护清单做不解释内容的字节SHA，不计算其指标。Stage9使用AgentFormer HOTEL折的其他来源train/val及OfflineComplete20，数据和目标完全不同，禁止新旧数字混排。

## 固定目标与历史上下文

目标须完整8帧历史和12帧未来，未来只定义监督资格/GT。上下文C只含历史8帧出现过的ID，部分历史保留、未来是否存在不影响入选。历史位置mask为真即允许接收/发送**位置信息**；相对速度只有两端各自连续两帧可见才有效。此规则区别于Stage5.1原SocialVAE固定6维接口屏蔽未知运动packet：本轮显式known通道允许位置-only消息，所以目标协议兼容不等于模型输入完全相同。

缺失速度以数值0占位并附显式标记，绝不解释成真实静止；风险退回当前距离，不能推断未知者正在接近。重新进入者不继承旧GRU或社会carry；没有插值或窗口前速度。`inference_inputs(recording,start)`只查询8个原始frame，函数没有未来参数。标签函数独立，监督mask不筛掉上下文节点。

## 集合核对与统计单位

'''+table+r'''
四份`*_targets.csv.gz`逐行保存`sample_id/source/frame_id/person_id/history_SHA/GT_SHA`，还保留完整上下文位置/mask/ID SHA；与Stage5.1全部历史/GT哈希完全相同。`sample_id=recording:start:person`，frame_id是窗口起点，last_observed_frame另存。哈希算法沿用`array_sha`：dtype+shape+连续字节；上下文ID和mask另散列。训练来源交集只在不同fold之间存在，单fold内train/val录像相交为0；两个fold不能当完全独立重复实验。

同一人的窗口高度重叠。未来必须同时报告窗口加权、source×person先平均后的等权、两录像宏平均，配对误差按同sample/ID/GT计算；不能把50,690训练目标窗口称为50,690独立行人。Stage5.1监督集合是当前冻结基准，不按新图规则删除难例。

阈值审计只用来源训练窗口的历史位置，不用验证误差/未来轨迹。对重复fold窗口按source+start去重，取每时刻最近可见邻居距离的q95，向上取0.5m网格并限制[2,8]m；邻居数为该半径下度数q75向上取整并限制[1,8]；风险距离尺度取最近距离中位数按0.5m取整并限制[.5,2]。结果radius=3m、K=8、sigma=.5m，H=4.8s按任务时域固定，速度数值阈值1e-4m/s。详细分位数/原始5文件SHA及旧manifest SHA见data_audit，不从验证ADE选择参数。

因果支持结论局限于本接口和测试覆盖：原始数据反事实删除/替换所有12帧未来、插入纯未来ID，HOTEL/ZARA2真实窗口的输入与三图逐位相同；人工原始记录反事实还验证随机权重预测逐位不变。它不证明完整实际社会影响，也不把监督资格使用未来标签错误称为输入泄漏。
''')
    save('03_MODEL_DESIGN.md',r'''
# 共同模型与S/D/R：结构定义

独立源码`src/risk_dynamic_graph/`；没有修改任何历史模型或第三方源码。来源/接口/许可审查见`scripts/stage10a/REUSE_AND_LICENSE.md`，本阶段不导入旧模型权重。时序GRU、均值消息传递和常规回归都属于已有技术。

## 历史运动与风险几何

输入位置P为[B,T=8,N,2]，有效性M为[B,T,N]，稳定整数ID为[B,N]。图输出candidate/edge为[B,T,N,N]，方向A[i,j]代表j→i。速度v=(p_t-p_{t-1})/.4仅在连续观测有效；r_ij=p_j-p_i，w_ij=v_j-v_i。两端速度已知为k_ij，否则相对速度占位0且k=False。

令H=4.8s、epsilon_v=1e-4m/s：

$$\tau_{ij}=\begin{cases}\operatorname{clip}(-r_{ij}^\top w_{ij}/\|w_{ij}\|^2,0,H),&k_{ij}\land\|w_{ij}\|>\epsilon_v,\\0,&\text{otherwise},\end{cases}$$
$$d^{CPA}_{ij}=\|r_{ij}+\tau_{ij}w_{ij}\|,\qquad q_{ij}=\exp[-(d^{CPA}_{ij}/\sigma)^2/2-\tau_{ij}/H],\quad\sigma=.5m.$$

这是有限时域最近接近时间/距离，而非圆盘碰撞TTC；两人并不一定相撞。远离/同速时最近点为当前，q也未必为0；未知速度时同样退回当前距离核但保留不同known标记。0或极低速度不除以0，未给未知历史伪造速度。常规CPA与固定指数核均不主张创新。

## 三种选边

共同：只在同一scene内、当前共同可见、排除自己；最多8邻居；确定性排序的次关键字为稳定source ID，不用存储索引。输入ID不能重复，padding使用独立ID。身份仅用作精确同分的可复核tie-break，不进入神经特征；因此保持带身份的存储置换等变，但不宣称任意ID重命名后精确同分选择也不变。

S：每对首次共同可见时记录距离。该次距离≤3m才有静态资格；同一时刻新出现的合法对按首次距离/ID排序，填充接收者尚未占用的永久K槽。已接纳关系永不因后续距离改变或离开而换人；离开仅mask，返回恢复旧关系。超预算或首次在3m外的对不重新考虑。这个**前缀因果静态扩展规则**避免用后续才出现的邻居决定较早图；初始所有人可见时等价于初帧topK距离静态图。它有先到者偏置，明确作为静态控制局限。

D：每帧从当前3m共同可见候选池选择距离最小K个。

R：与D逐位相同的当前3m候选池、相同每接收者K预算，选择q最大的K个。R不能看到3m之外的对象，是本轮公平预算约束；不以扩大候选范围伪造收益。

三种图在定义关系时使用相同3m阈值。S旧边可以在后续3m外仍有效，不能又声称S每帧有与D相同候选池，否则违反“静态关系仅visibility失效”的定义。S/D是整套静态/动态规则比较；**D/R才是主要的同池同边数排序对照**。S可能更稀疏，应报告实际消息预算，不把差异全部归因风险。

## 共享编码、消息和解码

GRUCell(3,64)逐节点共享，输入[vx,vy,velocity_known]，不含绝对坐标；无效/重新进入时carry重置。全部历史运算只用前缀，没有按最终位置回看归一化早期token。图模块在每帧使用当前ego隐状态，单跳消息不递归传播社会隐状态：

$$h_i^t=\operatorname{GRU}([v_i^t,k_i^t],\ \mathbf1_{\text{continuous}}h_i^{t-1}),$$
$$m_{ij}^t=W_2\operatorname{ReLU}(W_1[h_j^t,r_{ij}^t,w_{ij}^t,k_i^t,k_j^t,d_{ij}^t,d^{CPA}_{ij},\tau_{ij}]+b_1).$$

消息MLP为73→64→64，最后层无bias；三社会模型全部接收相同9维关系特征，风险量不只给R。选边后先按每帧真实入度均值，再在接收者当前连续片段的有消息帧均值：s_i=mean_t mean_j m_ij。无消息精确为0。E跳过整个社会路径。

解码器[最终h_i,s_i]为128→64→24，12步二维增量累积，再加最后有效历史位置。输出[B,N,12,2]，最终帧不可见节点输出padding0，不进入监督。无邻居且共同权重时社会模型逐位等于E；不是已验证CV轨迹。无额外时间注意力、归一化、关系GRU、高阶、社会gate、新loss或多模态头。

|模型|名义参数|结构有效上界|本轮有效路径|
|---|---:|---:|---|
|E|31,896|18,968|encoder、decoder自身列；消息8,832和decoder社会列4,096无有效输入贡献|
|S/D/R|31,896|31,896|相同encoder/message/decoder，只有图规则变化|

`parameters.csv`同时保留loss关联参数张量元素数和一次合成反向的非零梯度元素数，后者不代表长期训练活性。E的loss关联张量元素23,064仍包括4,096个恒零输入列，不能当有效容量。E对社会模型的性能差不能独立归因邻居信息；后续若比较必须另冻结合理ego容量控制或限制结论。S/D/R同参数量也不等于同函数类或同优化难度。

## 复杂度与边界

每帧密集候选几何O(BN²)，稳定排序O(BN²logN)，构图中间存储O(BTN²)；消息只处理选中边O(BTNK h²)，GRU O(BTNh²)，decoder O(BNh²+BN·12h)。没有宣称整套算法是O(NK)：当前仍显式计算所有候选对。S历史状态O(BN²)。硬topK有边界不连续，恒速估计对观测噪声/转弯敏感，风险不是校准概率。平移等变由速度、相对位置及末点锚定提供；未宣称任意旋转等变。
''')
    save('04_STRUCTURE_TESTS.md',r'''
# 结构验证与真实检查记录

最终全仓库测试命令及JUnit：`results/tables/stage10a/final_unit_tests_stdout.txt`、`final_unit_tests.xml`。旧测试458项与本阶段43项共501项通过。此前497项通过后补充同向/相向/交叉及动态入半径的4个明确测试，再运行全套。没有以结构通过冒充预测收益。

|要求|检查与证据|
|---|---|
|排列等变与梯度|E/S/D/R，N=1/2/5/12，输入/图随ID同步换序并逆换输出；float64预测及输入梯度atol1e-10/1e-9；同分按稳定ID确定|
|平移等变|全局+[8,-11]m，预测相应平移；原点无绝对坐标编码|
|padding/mask|padding坐标NaN由mask在运算前清除；不同padding宽度不影响有效预测；无效输出0；缺失与重入carry重置|
|零/单邻居与密集|N1、N2，GPU B8/N79无裁剪；无消息s严格0，与同权重E逐位相同|
|未知/静止速度|连续静止valid=True；首次观测velocity_known=False，两端已知才用相对速度；位置-only合法，missing标记显式|
|前缀因果|完整8帧前4步图等于只输入4帧；改后4帧不影响前4步图和encoder；raw帧查询Guard禁止窗口外访问|
|未来GT隔离|人工原始数据改变全部未来，图、输入、随机权重预测逐位相同；HOTEL/ZARA2真实原始记录反事实图与输入逐位相同，未计算验证预测或误差|
|DCA/tCPA独立核对|100对独立NumPy标量实现，包括未知、零/极小速度；与Torch tau/DCA/q一致atol1e-12|
|人工运动|同向相对速度0→tau0；相向和交叉同速到达→tau1s、DCA0；进入3m后动态图新增边，静态不追认|
|R/D不同选择|固定末点：target0；近邻1m远离、另一邻居3m以1m/s接近；K1结构fixture下D选ID20，R选ID30；见synthetic_edge_witness.json。实验K8另由真实训练图统计验证|
|确定性/排序|重复建图逐位相同；shuffle存储顺序、保持ID关联，恢复后风险排序/边相同|
|梯度/跨scene|合成forward/backward无NaN/Inf；E其他行人Jacobian=0；D合法邻居梯度非零；不同batch scene无交叉依赖|
|恢复|临时合成state_dict strict保存/加载，恢复预测逐位相同，未使用旧权重|

预固定的64个来源训练窗口（每fold均匀取32，可能跨fold重叠）中D/R都有59,924条时空有向边，2,909条D边被R替换，约4.85%；23/64窗口有差异。全部窗口逐接收者预算、候选mask、消息特征一致。其余窗口可能度数≤K或排序一致，不将相同图伪称无机制。记录在train_graph_selections.csv、selection_summary.json，不使用GT误差挑反例。

首次结构检查10失败/23通过：9项由PyTorch stable argsort要求dim为关键字引起，1项为人工风险例的sigma=1期望错误；第二次仅后者失败。修复API参数，并让人工例显式使用已冻结的sigma=.5，未修改风险公式或数据阈值。风险排序并不保证所有“接近者”都压过当前近邻，指数时间/距离权衡须按公式计算，不能先写期望再调验证参数。失败文本及最终通过记录全部保留；为git空白检查对文本尾空格规范化，原始gzip和SHA另存。

Stage10A GPU smoke仅作合成平方输出的反向，优化器更新0，权重前后逐位一致；不是正式预测loss或小规模训练。全仓库旧单元测试只为回归验证，不据其人工输出形成新性能指标。
''')
    gpu_table='|模型|B/N|前后向均值ms|PyTorch allocated峰MiB|reserved峰MiB|整卡观测MiB|\n|---|---|---|---|---|---|\n'
    for r in res['rows']:
        if r['device']=='cuda' and r['N']==79:
            gpu_table+=f"|{r['model']}|8/79|{r['mean_seconds']*1000:.3f}|{r['allocated_peak_bytes']/2**20:.2f}|{r['reserved_peak_bytes']/2**20:.2f}|{r['whole_card_after'].split(',')[1].strip()}|\n"
    hours=res['total_planning_GPU_hours']
    save('05_TRAINING_BUDGET.md',r'''
# 资源实测与后续预算估算

执行前配置/核心源码已提交`61ee113cdfd9110c644a72d46ed261d2985168b4`，实际GPU程序核对该commit和protocol SHA。Python3.10.21、PyTorch2.5.1+cu124、NumPy1.26.4，RTX3080 10GB，CPU单线程；依赖只读复用Stage8/4隔离环境，没有改环境。

唯一GPU任务为合成随机轨迹前向/平方输出反向：B8，N=1/12/32/79，E/S/D/R；2次warmup+8次计时，不创建优化器，不更新参数，不读取验证/测试GT。CPU另外B1/N12运行四模型。N79来自本阶段完整历史可见上下文计数最大值，不删除拥挤窗口，也不沿用AgentFormer完整8历史邻居N73限制。真实数据仅做历史统计/图核对，未调用新模型计算验证ADE/FDE。

'''+gpu_table+r'''
GPU无OOM、无NaN/Inf。整卡数字为nvidia-smi各case前后观测，包含桌面/驱动开销，并非高频采样的真实整卡峰值；精确allocator峰值为reset_peak_memory_stats后的allocated/reserved。整个smoke约8.09秒，包括计时外检查/CPU部分；不是正式epoch耗时。无训练checkpoint，只有单元测试临时合成恢复。

## 尚未授权的统一正式预算

仅作为成本估算：每模型30轮、seed17、batch8个scene；一个scene内全部合法目标参与统一损失，每epoch遍历所有有监督窗口。HOTEL 2,121窗口→266步/轮、7,980步；ZARA2 1,684→211步/轮、6,330步。四条件共57,240步；每epoch目标27,145/23,545，最后batch不丢弃。这不是历史target-centric batch128预算，必须明确统计单位改变，未来不把两者叫等价优化步数。

取N12及所有batch都padding至N79的实测成本范围，另乘1.5–4覆盖Adam、数据读取、验证和保存开销；两fold×四模型总计'''+f' **{hours[0]:.2f}–{hours[1]:.2f} GPU小时**'+r'''。这是规划估计而非已完成训练；密度变化、Python构图/排序同步、DataLoader、磁盘、验证频率、保存策略、系统负载均会改变实际值，不提供硬上界保证。资源预留建议约3GPU小时，若未来授权先固定一轮实测再审预算，不能无止境延长或删难例。

显存测量未包含Adam动量：按31,896个float32参数两份状态约.24MiB，远小于当前余量，但将来改变batch/模型仍需重新测量。Ego容量控制、额外seed或完整五折不在此估算内。逐case每步原始时间/P50/P90/P99、设备状态和每run预算见resources.json、budget_estimate.csv；没有用测试误差估算/选择预算。

本轮结论是硬件可承担此轻量工程，不是研究路线已GO。不得据剩余GPU时间自动开始训练。
''')
    save('06_NOVELTY_AUDIT.md',r'''
# 风险动态图创新性审计

结论：**风险量和风险选邻居不新；目前不支持把该框架作为新论文算法启动训练。** 文献部分采用nature-academic-search的multi-source-search；学术MCP未挂载，改用网页定位、CVF/作者PDF和官方源码。六篇原文方法均读取，PDF只在忽略cache保存，URL/下载SHA/text状态见paper_access.json。网页CVF fetch403后直接公开PDF下载200；未取得内容的搜索条目及不相干arXiv定位不作为方法证据。按标题/来源合并重复结果。技能附带OpenAlex检索脚本另返回HTTP429，失败日志保留，未反复请求；结论依靠已取得的原文与官方代码。本次是定向查重，不宣称穷尽2026全部论文。

## 最接近的六项工作

|论文/时间/来源|原结构及关系机制（概括）|与本轮的差别及重合|
|---|---|---|
|[Snapshot: Towards Application-centered Models for Pedestrian Trajectory Prediction in Urban Traffic Environments](https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Uhlemann_Snapshot_Towards_Application-centered_Models_for_Pedestrian_Trajectory_Prediction_in_Urban_WACVW_2025_paper.pdf)，WACV Workshops/HAVI 2025，预印2024|§3.3/Fig3/Tab3：社会Transformer+地图cross-attention+CNN。相对运动给tTCA/DCA及bearing变化，风险为三项平方尺度之和的负指数；比较top7距离/风险/无邻居|**直接覆盖本轮研究问题和核心选边思想**。AV2异构道路对象、1s→6s及地图不同于本轮ETH/UCY逐帧图/GRU；差异不足以自动成立新颖性。原文Tab3已报告风险选择改善，不能写“前人未比较风险与距离”。|
|[HighGraph: Higher-order Relational Reasoning for Pedestrian Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)，CVPR2024|§4.1 Eq2–3：运动射线交点C存在时Aij=1/[当前距离·两人到C的距离和]；多阶邻接幂及融合插件|运动/碰撞关系图已有；其射线交点不等于同步有限时域CPA。本轮没有高阶幂，但不能将“不用高阶”当创新。|
|[SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction](https://www.chengjianglong.com/publications/SGCN_CVPR.pdf)，CVPR2021|§3.1 Eq1–5：自注意力密集分数、非对称卷积、阈值稀疏mask、ZeroSoftmax、时空GCN及高斯输出；M=1[sigmoid(F)≥ξ]，A=(M+I)⊙R|稀疏/动态/自适应交互不是空白；本轮为解析选边，没有学习mask。旧SGCN代码结构问题不能证明其研究思想未存在。|
|[Social-STGCNN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf)，CVPR2020|§4/6.1 Eq2：图顶点差的逆L2核；ST-GCN+TXP、高斯NLL。维护版loader用相邻位移建图，不能一概叫纯世界位置距离|本轮D为明确位置距离，R为CPA；没有照搬其归一化/节点卷积，结构简化不构成创新。|
|[Trajectron++](https://arxiv.org/abs/2001.03093)，ECCV2020|§4.1：距离≤接收者类别感知半径建立有向图；按类型聚合邻居状态、edge LSTM、注意力、CVAE与运动学积分|动态节点、相对运动输入、距离筛选早已存在；本轮固定单跳MLP/CPA，不声称首个动态图或方向关系网络。|
|[SocialCircle](https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf)，CVPR2024|§3 Eq1–9：按相对角分区，汇总邻居净位移幅度、距离及方向，嵌入为虚拟时间序列与原骨干融合|位置、速度幅度和方向已有；本轮保留关系配对用于CPA，不重做Stage9交叉统计，也不从配对区别推导预测收益。|

独立CSV保存正式名称、结构、关系式、方法位置、代码及核查层级。HighGraph本轮不提供未经核实的官方repo URL；论文是当前方法证据。SGCN/Social-STGCNN版本沿用旧审计，不声称再次运行。

## 直接查重：Snapshot官方代码

[官方冻结版本](https://github.com/TUMFTM/Snapshot/tree/704e93339c58e15740c786881add06de8a397ba3)，Apache-2.0。`utils/social_features.py`真实实现了相对位置/速度、tTCA=-r·v/||v||²、DCA、collision risk及按risk排序截取邻居；config还有距离/风险选边开关。本轮通过git blob逐位核对文件SHA，未运行模型或移植其预处理。

论文Fig3风险公式为exp{-1/2[(tTCA/δt)²+(DCA/δd)²+(bearing_rate/δα)²]}。本轮为有限H裁剪、DCA平方加线性时间惩罚、未知速度距离回退。没有bearing-rate、使用不同常数或把kernel写成topK图，均不足以宣称新机制。源码部分常数/低相对速度回退与论文不同（如发布代码18/10和论文1.8/.3尺度），原因未核实，不以它们为本项目阈值，更不宣称本轮复现其论文性能。未审计完整AV2调用链，不能由局部代码猜测其未来泄漏。

## 三层结论

已发表/旧项目已有：TTC/TCA/DCA、碰撞几何、相对运动、风险选邻居、稀疏图、动态关系、共享GRU/MLP。准备复用：上述几何与简单消息框架、显式mask、稳定ID排序。尚需验证的研究问题：**在同一严格历史目标协议、相同候选/边数/容量下，解析风险排序究竟是否带来额外预测价值**，以及收益是否超越强CV/ego容量。

这可以成为复核/归因实验问题，但目前没有风险关系的真实预测收益证据，也没有区别充分的数学贡献。不是声称整个端到端网络与某文逐字等价，而是核心思想和比较问题已被明确覆盖。仅换核常数、GRU替换Transformer或更换数据集不支持SCI新颖性。若以后用户只授权机制复核，可另冻结小规模失败门槛；当前不推荐以“新模型创新”名义启动Stage10B。
''')
    save('STAGE10A_REPORT.md',r'''
# Stage10A：风险感知动态图可行性总报告

**工程机制通过；作为创新模型进入Stage10B正式训练，当前NO-GO。** 未训练新模型，未计算新模型验证ADE/FDE，没有测试集评价。没有设计高阶、关系GRU、注意力或新loss。风险选邻居已见Snapshot2025，旧TOHI也已用DCA排序，不能包装为首次方法。

## 交付与证据

|内容|报告/证据|
|---|---|
|历史失败审计|[01_HISTORY_AUDIT.md](01_HISTORY_AUDIT.md)：Dynamic-RN/高阶、G0/G1/PTN、SocialVAE、Stage9具体负结论及边界|
|因果数据冻结|[02_DATA_PROTOCOL.md](02_DATA_PROTOCOL.md)：四目标集合逐ID/history/GT SHA匹配Stage5.1；raw反事实/Guard；不与Stage9 Complete20混排|
|共同模型及公式|[03_MODEL_DESIGN.md](03_MODEL_DESIGN.md)：E/S/D/R，GRU64、单跳MLP、12步确定性回归，有限H的CPA/DCA|
|结构与失败记录|[04_STRUCTURE_TESTS.md](04_STRUCTURE_TESTS.md)：全仓库501通过，初次API/人工期望失败保留；R/D选边确实不同|
|真实资源/预算|[05_TRAINING_BUDGET.md](05_TRAINING_BUDGET.md)：RTX3080合成CPU/GPU前后向，优化器更新0，最多79人未裁剪|
|查重|[06_NOVELTY_AUDIT.md](06_NOVELTY_AUDIT.md)：六篇原文方法及Snapshot冻结官方源码，未声称穷尽检索或已有方法等于新创新|

数据协议：8/12、.4s、米制，HOTEL/ZARA2旧曝光开发区域；训练排除对应验证录像，其他旧validation区域不混入训练，ETH继续封存。训练目标27,145/23,545，验证318/1,092；28/35个验证source×person ID，重叠窗口不能当独立行人。邻居资格只看历史出现性、允许部分历史和位置-only消息；真实未来只定义监督目标及GT，未进入选边/风险/预测输入。

冻结图radius3m、K8、sigma.5m来自来源训练历史统计，H4.8s。S首次共同可见距离定义、永久槽/visibility；D当前距离；R同D候选池及边数、按历史相对运动外推CPA风险排序。S的首次到达偏置和后续可能在3m外的旧关系已披露；D/R才是严格同候选排序对照。风险不是碰撞概率/真实意图，常速度近似及硬topK均有限制。

四模型名义31,896参数，S/D/R结构有效上界相同；E上界18,968，不能将E与社会分支差异直接认定邻居信息价值。64个预定来源训练窗口中D/R候选及每接收者边数/特征完全一致，23窗口选边不同，2,909/59,924条边替换；仅为机制证据。synthetic K1 fixture选ID20/30的可复核反例另存，不把它当实验K8配置。

GPU B8/N79前后向E/S/D/R约16.47/38.30/30.97/29.97ms，allocated峰值最高约99MiB、整卡前后观测最高744MiB，无OOM。仅估算两录像×四条件×30轮约.65–1.86GPU小时（建议资源预留约3h），未实际执行完整epoch，不保证此为上界。真实GPU smoke总约8.09s、权重逐位未变。完整每步/分位数/参数/源与数据SHA见`results/tables/stage10a/`。

## Stage10B判断

1. 历史协议、mask/前缀/未来反事实、排列和平移等变、有效梯度及资源准入通过。当前检查未发现新的输入未来泄漏；资格用未来标签不等于输入泄漏，结论不超出测试范围。
2. 核心新颖性未通过：风险对距离选邻居已被Snapshot明确提出并比较；旧TOHI已有几何风险排名。现有随机权重测试不能证明风险图改善预测，更不能替代因果社会价值证据。
3. **不建议自动启动Stage10B创新训练**。若以后仅以严格归因/重复检验为目标，仍需单独授权并冻结强CV、ego容量控制、同目标逐人加权、边预算及跨录像失败门槛；不能为论文需要继续堆模块。本轮无新ADE/FDE可报告。

所有Stage1–9研究文件、权重及官方预测按SHA保护；最终结果以history_after.json为准。旧test预测仅不解释字节的SHA保护，没有重算/读取其误差作为选型依据。新阶段代码/报告/轻量证据独立，数据/PDF/cache/checkpoint不入Git。先冻结配置commit61ee113，最终提交与远程SHA以Git交付记录和最终反馈为准。本阶段用户明确授权push新分支，不修改main、不force。

完成后停止，不进入Stage10B。
''')


if __name__=='__main__':
    main()
