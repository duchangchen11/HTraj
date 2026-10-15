"""Freeze final audit and the single uniform continuation budget before execution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,json,subprocess,time
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.evaluation.graph_final_audit import diagnose_curve


def main():
    conf=Path('configs/stage4p2_final');root=Path('results/tables/stage4p2_final');root.mkdir(parents=True,exist_ok=True)
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert base=='55515d0ef5d50c36f4f148206d07f062644f92ff'
    old=json.loads(Path('configs/stage4p2/protocol.json').read_text());paths=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))-{''}
    paths|={r['path'] for r in json.loads(Path('configs/stage4p2/protected_artifacts.json').read_text())['files']}
    paths|={r['path'] for r in csv.DictReader(Path('results/tables/stage4p2/checkpoint_inventory.csv').open())};paths.add(old['cache'])
    write_json_new(conf/'protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
    audit=[];resume={};seconds=0
    for fold in old['folds']:
        for kind in ('G0','G1'):
            folder=Path('results/tables/stage4p2')/fold/kind/'deterministic_scan'
            rows=list(csv.DictReader((folder/'curves.csv').open()));row=dict(fold=fold,model=kind,**diagnose_curve(rows));audit.append(row)
            summary=json.loads((folder/'summary.json').read_text());resume[fold+'__'+kind]=dict(path=summary['last']['checkpoint'],sha256=summary['last']['checkpoint_sha256'],epoch=30)
            seconds+=row['median_train_eval_seconds']*70
    csv_new(root/'budget30_diagnosis.csv',audit)
    extend=all(r['clearly_unfinished'] for r in audit)
    config=dict(stage='4.2-final',base_commit=base,old_protocol_path='configs/stage4p2/protocol.json',old_protocol_sha256=sha256('configs/stage4p2/protocol.json'),
        cache=old['cache'],cache_sha256=old['cache_sha256'],manifests=old['manifests'],folds=old['folds'],models=['G0','G1'],seed=17,
        data_protocol='A_offline_complete20',ETH_evaluated=False,no_new_network=True,
        comparison=dict(models=['G0','G1','CV-Last2','CV-OLS4','CV-OLS8'],metrics='K1 Gaussian mean ADE/FDE, mean of12 Euclidean positions and last Euclidean position, m',
            arithmetic='same float32 frozen positions and neural predictions promoted to float64 for distance sums; check old metric differences<=1e-6',
            paired_keys='fold,window_id,agent_id,per-target future SHA256; no filtering; exact original frozen6886 targets',
            CV='Last2=(X7-X6)/.4; OLS4/8 slope using centered true .4s time; all anchor atX7 and extrapolate12*.4s; no learned parameters'),
        extend_to100=extend,extension=dict(start=31,end=100,initialization='resume exact epoch30 model+optimizer+CPU/CUDA RNG, not restart',
            rule='all four finite; trainNLL mean20:24 minus26:30>=.01, ADE reduction>=max(.005,.01*earlierADE), bestADE epoch>=28; all four must pass',
            frozen_optimizer='unchanged SGD.01, no momentum/decay/scheduler, clip1, batch16, accumulate64; same seed17+epoch ordering',
            selection='min validation Gaussian-mean ADE across1..100, exact tie earlier; same checkpoint FDE; separately save original budget30 and epoch100',
            endpoint='100 once, no further extension, no learning rate/structure/epsilon change; nonfinite stops affected run and reports incomplete',
            estimated_train_eval_seconds=seconds,estimated_minutes_with30percent_IO_margin=seconds*1.3/60,
            resume=resume),
        sensitivity=dict(checkpoints=['budget30_best','budget100_best_if_completed'],epsilon_m=[1e-5,1e-4,1e-3],replicates=[0,1],seed=271828,
            groups='historical mean speed<=.2 stationary,(.2,.8] low,(.8,1.6] normal walking,>1.6 fast; extra exact-stationary stepabs<=1e-8 and strict-stationary maxspeed<=.05+net<=.05m',
            noise='stable windowID+agentID keyed independent unit2D direction at each of8 points; same direction scaled byepsilon; all observed people jittered; no future GT',
            modes=['fixed_history_graph','recomputed_history_graph'],primary='fixed_history_graph for feature/PTN sensitivity; recompute secondary includes edge threshold changes',
            output='absolute position and relative-to-last forecast displacement shifts; actual inputRMS, gain P50/P95/P99/max; local PTN gain/lowvariance fraction; same jitter CV-Last2 calibration',
            risk_rule='at1mm, per-target max over2rep relative-final-shift>.1m and gain>100; group rate>=5% and>=30 unique people indicates repeated risk evidence, not iid significance or all-target claim',
            epsilon_unchanged=1e-5,interpretation='full fixed-network sensitivity plus local PTN hooks; no PTN-removal causal proof; overlapping windows dependent'),
        decisions=dict(worthwhile='G1 ADE lower by>=max(.005,.01*G0), FDE increase<=max(.01,.02*G0_FDE), both recordings required; also compare all three CVs',
            no_additional_seeds=True,no_G2_training=True,no_complex_modules=True,stop_after_audit=True),auto_push=False)
    write_json_new(conf/'protocol.json',config)
    write_json_new(root/'preparation.json',dict(base_commit=base,protocol_sha256=sha256(conf/'protocol.json'),data_sha256=old['cache_sha256'],
        protected_files=len(paths),audit= audit,extend_to100=extend,estimated_additional_seconds=seconds,
        source_files={str(p):sha256(p) for p in sorted(Path('src/graph_attribution').glob('*.py'))}))
    report=Path('reports/stage4p2_final');report.mkdir(parents=True,exist_ok=True)
    (report/'01_FROZEN_PROTOCOL.md').write_text('''# 最终验收：执行前冻结

基础55515d0，Stage1–4.2逐文件SHA保护。仅新增验收代码/解析CV/统计；旧网络、PTN epsilon、损失、图、数据和30轮指标均不修改。

完全配对：直接读Stage4.2冻结cache/manifest的HOTEL和ZARA2全部目标，8历史/12未来、0.4秒、米制，同目标future哈希；CV-Last2/OLS4/OLS8重新计算，绝不使用Stage3.8指标。神经输出/GT同提升float64计算距离，检查与旧float32指标差<=1e-6。

唯一预算延长：若四组同时满足20–24到26–30轮训练NLL降低>=.01，验证ADE降低>=max(.005m,1%早期ADE)，bestADE位于28–30轮且全有限，则全部从30恢复到100轮。学习率/优化器/梯度裁剪/累积/种子和模型均不变；不只延长G1，不再第二次延长。选1–100最低均值ADE、完全同值更早轮，保留原30预算与last100。ETH从未进入预算判定。实际检查四组均满足，估算见budget30_diagnosis.csv与protocol.json；此方案必须先提交后训练。

固定权重PTN审查：全体目标，30预算与100预算best均检查；历史速度固定四组另列精确/严格静止。1e-5/1e-4/1e-3m，每幅度两个ID键控相同方向扰动；主固定图、辅重算图，记录绝对与相对最后观测的预测位移变化、实际输入幅度、分位数、全部样本数和同扰动CV-Last2标定。不调epsilon，不用未来GT选病例。>=5%目标窗口达到1mm扰动引起>.1m且>100倍变化、组内>=30不同ID，才称重复风险迹象；仍不做窗口iid统计或宣称所有目标不稳。

所有模型仍seed17，仅两个已曝光录像，无多种子/ETH/新网络/G2训练。协议与完整hash、预估预算、停止条件见configs/stage4p2_final/protocol.json。完成本地commit，不push。
''')
    print(json.dumps(dict(extend=extend,extra_train_eval_seconds=seconds,minutes_with_margin=seconds*1.3/60,protected=len(paths))))


if __name__=='__main__':main()
