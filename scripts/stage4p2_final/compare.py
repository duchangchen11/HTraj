"""All-target, identical-GT five-model comparison for frozen30/extended100 budgets."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,gzip,json,time,subprocess,hashlib,itertools
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel
from src.evaluation.graph_final_audit import observed_batch,mean_positions,errors

ROOT=Path('results/tables/stage4p2_final');MODELS=('G0','G1','CV-Last2','CV-OLS4','CV-OLS8')


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    config=json.loads(Path('configs/stage4p2_final/protocol.json').read_text());assert sha256(config['cache'])==config['cache_sha256']
    cache=torch.load(config['cache'],weights_only=False);cv=list(csv.DictReader(gzip.open(ROOT/'fresh_cv_targets.csv.gz','rt')))
    cv_index={(r['fold'],r['window_id'],r['agent_id']):r for r in cv};metrics=[];groups=[];people=[];comparisons=[];runtimes=[];checkpoints=[]
    group_names=sorted(set(itertools.chain.from_iterable(r['groups'].split('|') for r in cv))|{'exact_stationary','strict_stationary','ptn_stationary','ptn_low_speed','ptn_normal_walking','ptn_fast'})
    for role in ('budget30_best','budget100_best','budget100_last'):
        combined=[]
        for fold in config['folds']:
            samples=[cache['records'][r][i] for r,i in cache['folds'][fold]['validation']]
            records={k:dict(v,role=role) for k,v in cv_index.items() if k[0]==fold}
            for kind in ('G0','G1'):
                if role=='budget30_best':summary=json.loads((Path('results/tables/stage4p2')/fold/kind/'deterministic_scan/summary.json').read_text());selected=summary['best']
                else:
                    summary=json.loads((ROOT/'extension'/fold/kind/'summary.json').read_text());selected=summary['last' if role.endswith('last') else 'best']
                assert sha256(selected['checkpoint'])==selected['checkpoint_sha256']
                model=GraphAttributionModel(kind).cuda().eval();model.load_state_dict(torch.load(selected['checkpoint'],weights_only=False)['model'])
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.no_grad():
                    for offset in range(0,len(samples),16):
                        part=samples[offset:offset+16];b=observed_batch(part);parameters=model(b['v'],b['relation'],b['valid'])
                        if not torch.isfinite(parameters).all():raise FloatingPointError('Nonfinite prediction')
                        prediction=mean_positions(parameters,b['last']).cpu()
                        for j,s in enumerate(part):
                            aa,ff=errors(prediction[j,:,:len(s['agent_ids'])],s['future'])
                            for i,pid in enumerate(s['agent_ids']):
                                row=records[fold,s['window_id'],str(pid)]
                                assert hashlib.sha256(s['future'][:,i].contiguous().numpy().tobytes()).hexdigest()==row['future_sha256']
                                row[kind+'_ADE_m']=float(aa[i]);row[kind+'_FDE_m']=float(ff[i])
                torch.cuda.synchronize();runtimes.append(dict(role=role,fold=fold,model=kind,evaluation_wall_seconds=time.perf_counter()-start))
                checkpoints.append(dict(role=role,fold=fold,model=kind,epoch=selected['epoch'],path=selected['checkpoint'],sha256=sha256(selected['checkpoint'])))
                avg=np.mean([float(r[kind+'_ADE_m']) for r in records.values()]);assert abs(avg-selected['K1_distribution_mean_ADE'])<1e-6
            rows=list(records.values());combined.extend(rows)
            for model in MODELS:
                aa=np.array([float(r[model+'_ADE_m']) for r in rows]);ff=np.array([float(r[model+'_FDE_m']) for r in rows])
                metrics.append(dict(role=role,fold=fold,model=model,target_windows=len(rows),unique_people=len({r['agent_id'] for r in rows}),ADE_m=float(aa.mean()),FDE_m=float(ff.mean())))
                for group in group_names:
                    selected=[r for r in rows if group in r['groups'].split('|')];row=dict(role=role,fold=fold,model=model,group=group,target_windows=len(selected),unique_people=len({r['agent_id'] for r in selected}))
                    row['ADE_m']=float(np.mean([float(r[model+'_ADE_m']) for r in selected])) if selected else None
                    row['FDE_m']=float(np.mean([float(r[model+'_FDE_m']) for r in selected])) if selected else None
                    for base in ('G0','CV-Last2','CV-OLS4','CV-OLS8'):
                        row['delta_ADE_vs_'+base]=float(np.mean([float(r[model+'_ADE_m'])-float(r[base+'_ADE_m']) for r in selected])) if selected else None
                        row['delta_FDE_vs_'+base]=float(np.mean([float(r[model+'_FDE_m'])-float(r[base+'_FDE_m']) for r in selected])) if selected else None
                    groups.append(row)
                for pid in sorted({r['agent_id'] for r in rows},key=int):
                    selected=[r for r in rows if r['agent_id']==pid]
                    people.append(dict(role=role,fold=fold,model=model,agent_id=pid,target_windows=len(selected),ADE_m=float(np.mean([float(r[model+'_ADE_m']) for r in selected])),FDE_m=float(np.mean([float(r[model+'_FDE_m']) for r in selected]))))
                for base in ('G0','CV-Last2','CV-OLS4','CV-OLS8'):
                    differences=aa-np.array([float(r[base+'_ADE_m']) for r in rows]);fdelta=ff-np.array([float(r[base+'_FDE_m']) for r in rows])
                    comparisons.append(dict(role=role,fold=fold,model=model,baseline=base,target_windows=len(rows),delta_ADE_m=float(differences.mean()),delta_FDE_m=float(fdelta.mean()),fraction_ADE_improved=float((differences<0).mean())))
        with gzip.open(ROOT/(role+'_paired_targets.csv.gz'),'xt',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    csv_new(ROOT/'model_comparison.csv',metrics);csv_new(ROOT/'all_group_results.csv',groups);csv_new(ROOT/'per_person_errors.csv',people)
    csv_new(ROOT/'paired_comparisons.csv',comparisons);csv_new(ROOT/'evaluation_times.csv',runtimes);csv_new(ROOT/'evaluated_checkpoint_inventory.csv',checkpoints)
    aggregate=[]
    for role in ('budget30_best','budget100_best','budget100_last'):
        for model in MODELS:
            rr=[r for r in metrics if r['role']==role and r['model']==model]
            for scheme in ('recording_macro','target_window_weighted'):
                w=np.array([1 if scheme=='recording_macro' else r['target_windows'] for r in rr],dtype=float);w/=w.sum()
                aggregate.append(dict(role=role,model=model,scheme=scheme,ADE_m=float(np.dot(w,[r['ADE_m'] for r in rr])),FDE_m=float(np.dot(w,[r['FDE_m'] for r in rr]))))
    csv_new(ROOT/'aggregate_comparison.csv',aggregate)
    write_json_new(ROOT/'comparison_execution.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        cache_sha256=sha256(config['cache']),cv_table_sha256=sha256(ROOT/'fresh_cv_targets.csv.gz'),GT_identity_sha256=sha256(ROOT/'target_identity_hashes.csv'),
        all_models_same_targets=True,neural_metrics_match_frozen_within_1e_minus6=True,overlap='window targets dependent; per-person and recording means descriptive; no iid p-values',ETH_evaluated=False))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
