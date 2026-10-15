"""Evidence-grounded qualifications after complete result and diagnostic audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
R=Path('results/tables/stage5p2');D=Path('reports/stage5p2')
audit=json.load(open(R/'convergence_audit.json'));metrics=read_csv(R/'model_metrics.csv')

def append(file,text):
    path=D/file;marker='## 最终验收补充（真实数据）'
    assert marker not in path.read_text()
    with path.open('a',encoding='utf-8') as f:f.write('\n'+marker+'\n\n'+text.strip()+'\n')

def m(rec,model,sel,family,weight):return next(r for r in metrics if (r['recording'],r['model'],r['selection'],r['metric_family'],r['weighting'])==(rec,model,sel,family,weight))

cuda=json.load(open(R/'CUDA_sampling_invariance_diagnostic.json'))
failed=[r for r in cuda if not r['passed']]
assert len(failed)==4 and all(r['test'].startswith('batch_partition') for r in failed)
warning='''完整单元测试302项通过（`final_tests.xml`）。额外固定权重CUDA诊断8项中：同batch置换K1/K20的4项预测最大差异均为0；跨batch拆分4项均未通过预设atol/rtol=1e-5，失败标记保留，未放宽阈值。ZARA2 K1/K20最大坐标差为0.000204086/0.001112938m，HOTEL为0.000275135/0.000776291m。标准高斯epsilon按目标ID完全一致；这说明控制随机数并不消除浮点批处理与padding布局带来的数值差异。官方网络/epsilon/主评价batch=128均未修改，没有重训或重选checkpoint。原始失败现场与完整诊断见`CUDA_initial_tolerance_failure.txt`、`CUDA_sampling_invariance_diagnostic.json`。这些只是各录像16个固定索引的功能诊断，不是全验证集最大误差上界，也不证明差异完全由某一GPU算子造成。'''
append('STAGE5P2_REPORT.md', '''主指标按目标窗口加权：ZARA2 K1相对Last2改善，HOTEL退化；K20两录像均优于三种CV，但这是20样本GT最优口径。**按行人等权时HOTEL的K20为0.4212/0.8036m，仍差于OLS4的0.3546/0.6914m和OLS8的0.3816/0.7696m**。不能将窗口加权下的结论推广至所有权重或所有行人。

两个录像后期训练误差下降而验证恶化，存在明显过拟合/跨录像泛化退化证据；best位于30/3轮，未由100轮边界截断。训练目标仍下降，不等于数值收敛，但没有证据支持本阶段自动延长。SocialVAE可暂作为多模态开发基线，尚不支持独立社会收益或第四章创新成立；下一步仅建议预注册完整NoSocial，当前不执行。

'''+warning)
append('04_CV_COMPARISON.md','''窗口加权：HOTEL K1相对Last2的ADE退化0.0577m，相对OLS8退化0.1670m；ZARA2 K1相对Last2改善0.0237m。K20窗口加权HOTEL相对OLS8改善0.0177m，ZARA2相对Last2改善0.1388m。

行人等权：HOTEL K1为0.7361/1.4786m，差于全部三个CV；HOTEL K20为0.4212/0.8036m，优于Last2但弱于OLS4、OLS8。说明该录像较长轨迹的窗口权重显著影响总体结论。ZARA2 K1行人等权为0.3598/0.7009m，也优于Last2的0.4191/0.8453m；K20为0.1871/0.3097m。

两个录像窗口口径宏K1为0.3299/0.6795m，弱于Last2宏0.3129/0.6393m；宏K20为0.1800/0.3151m。宏平均不抹去HOTEL K1退化和行人等权强CV反例。''')
append('05_ERROR_AND_GENERALIZATION.md','''K1：HOTEL历史静止106窗口/4ID退化0.0808m，正常行走91窗口/24ID退化0.2659m，低速121窗口/5ID改善0.1191m。转弯组仅3窗口/2ID，改善0.1548m不宜推广。ZARA2静止579窗口/9ID退化0.0159m，低速91窗口/10ID ADE基本持平（+0.0008m），正常行走422窗口/33ID改善0.0834m，转弯15窗口/5ID改善0.1758m。两个录像高速组均空；ZARA2方向突变组也为空，不能据此宣布该类问题被解决。

K1社会条件：HOTEL有/无合法邻居分别退化0.0617/0.0251m，同向/相向退化，交叉组11窗口/8ID改善；新进入和邻居速度不可观测组改善但并非整体优势。ZARA2有/无邻居都改善（−0.0243/−0.0177m），因此不能将收益直接归因社会信息。ZARA2交叉组7窗口/3ID ADE略改善0.0183m、FDE却退化0.0141m，必须配对报告。

K20相对Last2在两录像三个非空速度组及有/无邻居组均改善，但它依赖20候选GT最优评分；HOTEL行人等权强CV反例说明窗口层面覆盖收益不等于每类行人均优。所有CV、所有组与两种权重的完整行均已保留。''')
for rec,file in [('crowds_zara02','02_ZARA2_TRAINING_RESULTS.md'),('biwi_hotel','03_HOTEL_TRAINING_RESULTS.md')]:
    a=audit[rec];best=a['best_validation'];last=a['last_epoch'];tr=a['best_train_K1']
    text=f"""同一训练cohort的确定性K1 ADE从best轮{tr['train_K1_prior_ADE_m']:.4f}降至第100轮{last['train_K1_prior_ADE_m']:.4f}m，验证却从{best['K1_prior_ADE_m']:.4f}升至{last['K1_prior_ADE_m']:.4f}m；验证posterior目标从{best['posterior_loss']:.6f}升至{last['posterior_loss']:.6f}。这支持后期过拟合/录像迁移下泛化退化，不能仅用best轮训练误差与验证误差的大小关系判断。

末20轮训练loss仍下降；HOTEL验证K1虽末段回落但远弱于早期best，ZARA2末段K1基本平台且K20继续恶化。最优轮不在预算边界，未发现必须自动延长的依据；不宣称所有优化量已收敛。

"""
    append(file,text+warning)
append('06_RESEARCH_DECISION.md','''对第2–4问的明确限定：主窗口加权下，K1只在ZARA2获益，两个录像宏K1弱于Last2；K20在两个录像优于三种CV，但HOTEL行人等权K20不如OLS4/OLS8。故既不能宣布整体失败，也不能宣称跨录像、跨权重全面胜出。首随机样本在两录像均弱于Last2，20候选收益应解释为覆盖/多模态评分优势。

对第5、8问：后期过拟合证据明确。ZARA2训练K1 ADE从best轮0.4559降至last0.1156m，验证从0.2726升至0.3290m；HOTEL训练从0.5135降至0.1146m，验证从0.3872升至0.6470m。训练loss与验证posterior变化也支持泛化退化。两录像训练难度不同，不能把某一轮训练/验证误差的绝对大小作为唯一依据。两个最优点远离100轮边界，预算不是当前best选择的明显截断原因；后20轮训练目标仍下降，所以不声称充分优化收敛，也不自动追加。

对第6–7问：HOTEL低速和少量转弯/方向变化组改善，静止/正常行走及有/无邻居整体退化；ZARA2正常行走/转弯改善，静止退化，交叉组FDE略退化。分组不同可能存在混杂，不以未来错误挑选模型输入，不作社会机制因果解释。

对第9–10问：可保留为**有限可用的公开多模态开发基础**，值得下一阶段预注册完整NoSocial归因，优先保证相同因果协议、容量、loss、选择及采样预算。暂不确认第四章创新路线；独立社会收益、多seed稳定性、HOTEL按行人公平性与后期泛化仍未解决。停止添加GNN/关系GRU或更改loss，等待新指令。

'''+warning)
print('Supplemented six new stage reports; frozen preregistration unchanged')
