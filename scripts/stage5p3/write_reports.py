"""Actual trained Full/NoSocial results only, capacity-control stays unrun."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv

ROOT=Path('results/tables/stage5p3');DOC=Path('reports/stage5p3')
REC=['biwi_hotel','crowds_zara02'];LABEL={'biwi_hotel':'HOTEL','crowds_zara02':'ZARA2'}


def n(x,digits=4):return '无目标/未运行' if x is None or x=='' else f'{float(x):.{digits}f}'


def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def write(name,text):
    with (DOC/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    metric=read_csv(ROOT/'model_metrics.csv');effects=read_csv(ROOT/'social_effects_all_groups.csv')
    ns_audit=json.load(open(ROOT/'convergence_audit.json'))
    full_audit=json.load(open('results/tables/stage5p2/convergence_audit.json'))
    def get(rec,model,sel='best',family='K1-prior',weight='target-window'):
        return next(r for r in metric if (r['recording'],r['model'],r['selection'],r['metric_family'],r['weighting'])==(rec,model,sel,family,weight))
    rows=[]
    for rec in REC:
        for model in ['CV-Last2','CV-OLS4','CV-OLS8']:
            r=get(rec,model,'analytic','deterministic');p=get(rec,model,'analytic','deterministic','equal-person-ID')
            rows.append([LABEL[rec],model,'解析','—','确定性',n(r['ADE_m']),n(r['FDE_m']),n(p['ADE_m']),n(p['FDE_m'])])
        for model,audit in [('Full',full_audit),('NoSocial',ns_audit)]:
            for sel in ['best','last']:
                for family in ['K1-prior','K1-first-sample','K20-independent-minima']:
                    r=get(rec,model,sel,family);p=get(rec,model,sel,family,'equal-person-ID')
                    rows.append([LABEL[rec],model,sel,audit[rec]['best_epoch'] if sel=='best' else 100,family,n(r['ADE_m']),n(r['FDE_m']),n(p['ADE_m']),n(p['FDE_m'])])
    details=[]
    for rec in REC:
        for model,audit in [('Full',full_audit),('NoSocial',ns_audit)]:
            a=audit[rec];last=a['last_epoch'];best=a['best_validation']
            details.append(f"### {LABEL[rec]} {model}\n\n参数名义2144002；best第{a['best_epoch']}轮，last第100轮。实际墙钟{a['actual_run_wall_seconds']:.2f}s（{a['actual_run_wall_seconds']/60:.2f}分钟），峰值allocated {a['peak_allocated_bytes']/2**30:.3f}GiB。训练源码提交`{a['code_commit']}`。\n\n第1轮训练loss {a['first_epoch']['train_loss']:.6f} → last {last['train_loss']:.6f}（rec {last['train_rec']:.6f} + MC KL {last['train_kl']:.6f}）；best验证posterior loss {best['posterior_loss']:.6f} → last {last['posterior_loss']:.6f}。best固定权重完整训练K1 ADE {a['best_train_K1']['train_K1_prior_ADE_m']:.4f} → last训练 {last['train_K1_prior_ADE_m']:.4f}m；验证 {best['K1_prior_ADE_m']:.4f} → {last['K1_prior_ADE_m']:.4f}m。\n\n"+table(['末20轮指标','81–90均值','91–100均值','后段−前段'],[[field,*[n(v[k],6) for k in ['epoch81_90','epoch91_100','difference']]] for field,v in a['last20'].items()]))
    life=[]
    for rec in REC:
        rr=read_csv(ROOT/rec/'lifetime_parameter_gradients.csv')
        life.append([LABEL[rec],sum(int(r['nominal']) for r in rr),sum(int(r['nominal']) for r in rr if r['gradient_connected_ever']=='True'),sum(int(r['ever_nonzero_gradient_scalars']) for r in rr),sum(int(r['never_nonzero_gradient_scalars']) for r in rr),sum(int(r['structural_effective_upper_bound']) for r in rr)])
    write('04_TRAINING_RESULTS.md', '''# 正式训练与冻结Full对照

两个NoSocial run各自seed17从头训练100完整epoch，未加载Full训练权重，未调参、未追加seed/epoch。Full直接引用Stage5.2冻结正式结果，两个模型均按验证K1-prior ADE最低选择best、FDE同checkpoint。原始数据、训练池、目标ID、GT SHA、评价函数、目标噪声和K20预算相同，核查见`cohort_verification.json`。这仍是历史曝光开发录像，非独立确认；ETH未评价。

'''+table(['录像','模型','选择','epoch','口径','窗口ADE','窗口FDE','行人等权ADE','行人等权FDE'],rows)+'''

K1-prior不是分布数学期望；首样本为同一K20中的候选0；minADE/minFDE分别选最优候选，不保证候选相同。全部模型FPC关闭。CV三者保留，Last2为固定主参考，不按录像挑弱对手。Ego-Capacity-Control未实现/未训练，不填写性能。原始loss为重建+MC KL，非精确边缘NLL。

## NoSocial生命周期容量实测

'''+table(['录像','名义','曾连接loss图的张量参数','全程曾非零梯度标量','全程从未非零标量','结构有效上界'],life)+'''

梯度并集记录了该run全部optimizer步骤；它依赖有限训练数据和数值精度，不等于函数类维数。结构死参数逐轮保持原始值。Full只有旧Adam二阶矩代理，不伪造其逐步梯度历史。

## 曲线与泛化

'''+ '\n'.join(details)+'''

以上耗时为实际执行，NoSocial额外逐步梯度审计而Full没有该仪器开销，不能据两者墙钟差直接宣称架构效率优势。训练loss是更新中batch加权平均，验证posterior是固定权重、固定目标噪声诊断；不可称精确边缘似然比较。

100轮并不自动证明收敛。最优点、末20轮和同一checkpoint训练/验证差距必须一起看；任何末段改善都不触发额外训练。完整每轮/每batch曲线、逐目标误差、定期模型/optimizer/RNG严格恢复、checkpoint SHA和执行日志已保存。所有NoSocial模型在`experiments/stage5p3/checkpoints/`；best/last由`selection.json`给出，不覆盖旧权重。
''')
    sections=[]
    for rec in REC:
        for weight in ['target-window','equal-person-ID']:
            for family in ['K1-prior','K20-independent-minima']:
                rr=[r for r in effects if r['recording']==rec and r['selection']=='best' and r['metric_family']==family and r['weighting']==weight]
                sections.append(f'## {LABEL[rec]} / {weight} / {family}\n\n'+table(['历史组','目标窗口','行人ID','Full ADE/FDE','NoSocial ADE/FDE','Full−NoSocial ADE/FDE'],[
                    [r['group'],r['target_windows'],r['people'],n(r['Full_ADE_m'])+'/'+n(r['Full_FDE_m']),n(r['NoSocial_ADE_m'])+'/'+n(r['NoSocial_FDE_m']),n(r['Full_minus_NoSocial_ADE_m'])+'/'+n(r['Full_minus_NoSocial_FDE_m'])] for r in rr]))
    write('05_INTERACTION_GROUP_ANALYSIS.md','''# 全部分组社会收益分析

阈值完整沿用Stage5.2，全部组只由历史八帧定义，不用未来GT挑输入或组。负的Full−NoSocial表示Full误差更小。best按各模型同一K1选择规则独立确定，配对的是相同目标而非相同epoch；last固定100轮另保存。

NoSocial不读取邻居，但仍按相同的历史邻居条件分组，故组成员完全配对。多邻居方向标签可重叠；空组不填零误差。窗口加权、先ID内平均再ID等权均完整报告，录像宏平均见`recording_macro_metrics.csv`。重叠窗口不是独立行人，也不把行人ID等权当作独立显著性检验。

'''+ '\n'.join(sections)+'''

全部三种预测口径、best/last、所有组、所有模型含CV的原始行分别见`all_group_metrics.csv`、`social_effects_all_groups.csv`、`paired_model_errors.csv.gz`与`paired_social_effects.csv.gz`。不能因某个小组改善就宣称新机制有效。有效容量差异仍未排除。
''')


if __name__=='__main__':main()
