"""Paired all-group analysis; supplementary checkpoint interventions are not retraining."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,gzip,json,subprocess,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel,gaussian_predictions
from src.graph_attribution.data import pack
from src.graph_attribution.engine import evaluate,MODES
ROOT=Path('results/tables/stage4p2');FOLDS=('crowds_zara02','biwi_hotel');ATTEMPT='deterministic_scan'
GROUPS=('all','context_N2to4','context_N5to8','context_N9plus','legal_neighbors_none','legal_neighbors_present',
 'legal_degree_low','legal_degree_mid','legal_degree_high','speed_stationary','speed_slow','speed_walking','speed_fast',
 'motion_smooth','motion_variable','path_turning_or_return','path_straight_or_low_speed','observed_neighbors_none',
 'observed_neighbors_present','observed_neighbors_but_no_model_edge','same_direction','opposing_direction','crossing_direction',
 'relations_changing','relations_stable')


def read(path):return list(csv.DictReader(Path(path).open()))


def csv_gz(path,rows):
    with gzip.open(path,'xt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    config=json.loads(Path('configs/stage4p2/protocol.json').read_text());cache=torch.load(config['cache'],weights_only=False)
    summaries={};comparison=[];curves=[];paired=[];groups=[];person=[];permutation=[];dependency=[];interventions=[];intervention_targets=[];gate=[]
    started=time.perf_counter()
    for fold in FOLDS:
        val=[cache['records'][r][i] for r,i in cache['folds'][fold]['validation']]
        for kind in ('G0','G1'):
            directory=ROOT/fold/kind/ATTEMPT;summary=json.loads((directory/'summary.json').read_text());summaries[fold,kind]=summary
            curve=read(directory/'curves.csv');curves.extend(dict(fold=fold,model=kind,**r) for r in curve)
            for label in ('best','last'):
                row=dict(fold=fold,model=kind,checkpoint_role=label,parameters=summary['parameters'],training_seconds=summary['training_seconds'],
                    total_evaluation_seconds=summary['evaluation_seconds'],source_commit=summary['source_commit'],**summary[label]);comparison.append(row)
                # Keep original uncompressed output locally; track compact exact archives.
                source=directory/(label+'_targets.csv');dest=directory/(label+'_targets.csv.gz')
                with source.open('rb') as src,gzip.open(dest,'xb') as out:out.write(src.read())
            model=GraphAttributionModel(kind,config['width'],config['graph']['gamma']).cuda().eval()
            model.load_state_dict(torch.load(summary['best']['checkpoint'],weights_only=False)['model'])
            generator=torch.Generator().manual_seed(9981)
            for sample in val[:64]:
                n=len(sample['agent_ids']);batch=pack([sample],kind);p=torch.randperm(n,generator=generator).cuda();inv=p.argsort()
                with torch.no_grad():
                    y=model(batch['v'],batch['relation'],batch['valid']);yp=model(batch['v'][...,p],batch['relation'][:,:,p][:,:,:,p],batch['valid'][:,p])[:,:,inv]
                    zero=torch.zeros(1,20,12,n,2,device='cuda');ym,_=gaussian_predictions(y,batch['last'],zero);pm,_=gaussian_predictions(yp,batch['last'],zero)
                    delta=(ym-pm).norm(dim=-1)
                permutation.append(dict(fold=fold,model=kind,window_id=sample['window_id'],people=n,
                    max_parameter_difference=float((y-yp).abs().max()),mean_position_difference_m=float(delta.mean()),max_position_difference_m=float(delta.max())))
                assert float((y-yp).abs().max())<2e-5 and float(delta.max())<2e-4
            # An actual trained-checkpoint graph/Jacobian example, nearest available edge.
            sample=next(s for s in val if bool(s['relation'].any()));batch=pack([sample],kind);x=batch['v'].detach().requires_grad_();r=batch['relation']
            receiver,source=(r[0].sum(0)>0).nonzero()[0].tolist()
            y=model(x,r,batch['valid']);gradient=torch.autograd.grad(y[:,:,receiver].sum(),x)[0]
            with torch.no_grad():noedge=model(x,torch.zeros_like(r),batch['valid'])
            dependency.append(dict(fold=fold,model=kind,window_id=sample['window_id'],receiver=receiver,source=source,
                sender_jacobian_max=float(gradient[...,source].abs().max()),zero_graph_distribution_difference=float((y-noedge).abs().max())))
            if kind=='G0':assert float(gradient[...,source].abs().max())==0 and torch.equal(y,noedge)
            # Hold G1 checkpoint fixed. These OOD interventions are auxiliary only.
            if kind=='G1':
                for name in ('real','control','zero'):
                    modified=[dict(s,relation=s['control'] if name=='control' else torch.zeros_like(s['relation']) if name=='zero' else s['relation']) for s in val]
                    metrics,targets=evaluate(model,modified,config['batch_windows'],True)
                    interventions.append(dict(fold=fold,checkpoint_sha256=summary['best']['checkpoint_sha256'],intervention=name,
                        evidence='fixed_checkpoint_inference_not_retraining',**metrics))
                    intervention_targets.extend(dict(fold=fold,intervention=name,**r) for r in targets)
        ego,social=summaries[fold,'G0'],summaries[fold,'G1'];assert ego['initial_state_sha256']==social['initial_state_sha256']
        ade0,ade1=ego['best']['K1_distribution_mean_ADE'],social['best']['K1_distribution_mean_ADE'];fde0,fde1=ego['best']['K1_distribution_mean_FDE'],social['best']['K1_distribution_mean_FDE']
        gate.append(dict(fold=fold,ADE_reduction_m=ade0-ade1,ADE_required_m=max(.005,.01*ade0),ADE_worthwhile=ade0-ade1>=max(.005,.01*ade0),
            FDE_increase_m=fde1-fde0,FDE_allowed_m=max(.01,.02*fde0),FDE_safe=fde1-fde0<=max(.01,.02*fde0)))
        for label in ('best','last'):
            data={kind:read(ROOT/fold/kind/ATTEMPT/(label+'_targets.csv')) for kind in ('G0','G1')}
            key=lambda r:(r['window_id'],r['agent_id'],r['mode'])
            e0={key(r):r for r in data['G0']};e1={key(r):r for r in data['G1']};assert e0.keys()==e1.keys()
            rows=[]
            for k,a in e0.items():
                b=e1[k];row=dict(fold=fold,checkpoint_role=label,window_id=k[0],agent_id=k[1],mode=k[2],
                    G0_ADE_m=float(a['ADE_m']),G0_FDE_m=float(a['FDE_m']),G1_ADE_m=float(b['ADE_m']),G1_FDE_m=float(b['FDE_m']),
                    delta_ADE_m=float(b['ADE_m'])-float(a['ADE_m']),delta_FDE_m=float(b['FDE_m'])-float(a['FDE_m']),
                    **{s:v for s,v in a.items() if s not in ('recording','window_id','agent_id','mode','ADE_m','FDE_m')})
                rows.append(row);paired.append(row)
            for mode in MODES:
                subset=[r for r in rows if r['mode']==mode]
                for group in GROUPS:
                    selected=[r for r in subset if group in r['groups'].split('|')]
                    row=dict(fold=fold,checkpoint_role=label,mode=mode,group=group,target_windows=len(selected),unique_people=len({r['agent_id'] for r in selected}))
                    for metric in ('G0_ADE_m','G0_FDE_m','G1_ADE_m','G1_FDE_m','delta_ADE_m','delta_FDE_m'):
                        row[metric]=float(np.mean([r[metric] for r in selected])) if selected else None
                    row['improved_ADE_fraction']=float(np.mean([r['delta_ADE_m']<0 for r in selected])) if selected else None
                    row['history_frames']=8
                    for feature in ('mean_speed_mps','mean_acceleration_mps2','model_mean_degree','historical_visible_mean_degree','historical_complete8_mean_degree','offline_context_people'):
                        row[feature]=float(np.mean([float(r[feature]) for r in selected])) if selected else None
                    groups.append(row)
                byid=defaultdict(list)
                for r in subset:byid[r['agent_id']].append(r)
                for pid,rr in byid.items():
                    person.append(dict(fold=fold,checkpoint_role=label,mode=mode,agent_id=pid,target_windows=len(rr),
                        **{metric:float(np.mean([r[metric] for r in rr])) for metric in ('G0_ADE_m','G0_FDE_m','G1_ADE_m','G1_FDE_m','delta_ADE_m','delta_FDE_m')}))
    csv_new(ROOT/'model_comparison.csv',comparison);csv_new(ROOT/'all_training_curves.csv',curves)
    csv_gz(ROOT/'paired_target_errors.csv.gz',paired);csv_new(ROOT/'paired_person_means.csv',person);csv_new(ROOT/'group_comparison.csv',groups)
    csv_new(ROOT/'trained_permutation.csv',permutation);csv_new(ROOT/'trained_jacobian.csv',dependency)
    csv_new(ROOT/'checkpoint_graph_interventions.csv',interventions);csv_gz(ROOT/'checkpoint_intervention_targets.csv.gz',intervention_targets)
    controls=[r for r in read(ROOT/'graph_control_statistics.csv') if r['split']=='validation']
    relation_gate=all(float(r['edge_replacement_fraction'])>=.25 and float(r['changed_graph_fraction'])>=.5 for r in controls)
    both=all(r['ADE_worthwhile'] and r['FDE_safe'] for r in gate)
    write_json_new(ROOT/'first_round_decision.json',dict(folds=gate,both_worthwhile=both,relation_control_gate=relation_gate,run_G2=bool(both and relation_gate),
        recommend_seeds42_2026=both,additional_seeds_executed=False,G2_training_status='not_run_predefined_gate_failed' if not (both and relation_gate) else 'required_by_preregistration'))
    aggregates=[]
    for label in ('best','last'):
        for mode in MODES:
            rr=[r for r in groups if r['checkpoint_role']==label and r['mode']==mode and r['group']=='all']
            for scheme in ('recording_macro','target_window_weighted'):
                weights=np.array([1 if scheme=='recording_macro' else r['target_windows'] for r in rr],dtype=float);weights/=weights.sum()
                aggregates.append(dict(checkpoint_role=label,mode=mode,scheme=scheme,recordings=len(rr),target_windows=sum(r['target_windows'] for r in rr),
                    **{metric:float(np.dot(weights,[r[metric] for r in rr])) for metric in ('G0_ADE_m','G0_FDE_m','G1_ADE_m','G1_FDE_m','delta_ADE_m','delta_FDE_m')}))
    csv_new(ROOT/'aggregate_comparison.csv',aggregates)
    checkpoints=[]
    for path in sorted(Path('experiments/stage4p2').rglob('epoch*.pt')):
        checkpoints.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
    assert len(checkpoints)==120;csv_new(ROOT/'checkpoint_inventory.csv',checkpoints)
    env=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    write_json_new(ROOT/'environment.json',dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),pip_freeze=env.splitlines(),deterministic=True,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG']))
    logs=[]
    with gzip.open(ROOT/'execution_logs.txt.gz','xt') as out:
        for path in sorted(Path('experiments/stage4p2').glob('*.log')):
            out.write('\nFILE '+str(path)+'\n'+path.read_text());logs.append(dict(path=str(path),sha256=sha256(path)))
    write_json_new(ROOT/'analysis_execution.json',dict(seconds=time.perf_counter()-started,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_script_sha256=sha256(__file__),training_commit=summaries[FOLDS[0],'G0']['source_commit'],protocol_sha256=sha256('configs/stage4p2/protocol.json'),
        cache_sha256=sha256(config['cache']),logs=logs,ETH_evaluated=False,overlap_warning='target windows are dependent; recording/person aggregates descriptive only; no iid-window p-values'))
    print(json.dumps(dict(comparison=[{k:r[k] for k in ('fold','model','checkpoint_role','epoch','K1_distribution_mean_ADE','K1_distribution_mean_FDE','NLL')} for r in comparison],gate=gate,run_G2=bool(both and relation_gate))))


if __name__=='__main__':main()
