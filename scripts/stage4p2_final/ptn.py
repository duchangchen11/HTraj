"""Whole-cohort fixed-checkpoint stress test, unchanged epsilon and no future GT."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,gzip,json,time,subprocess
from collections import defaultdict
from contextlib import ExitStack
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel,PersonTemporalNorm
from src.graph_attribution.data import real_relation
from src.evaluation.graph_final_audit import observed_batch,mean_positions,jitter_history,motion_group,cv_prediction

ROOT=Path('results/tables/stage4p2_final');MODES=('fixed_history_graph','recomputed_history_graph')


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    protocol=json.loads(Path('configs/stage4p2_final/protocol.json').read_text());config=protocol['sensitivity']
    assert sha256(protocol['cache'])==protocol['cache_sha256'];cache=torch.load(protocol['cache'],weights_only=False)
    worst={};summaries=[];provenance=[];start=time.perf_counter()
    for fold in protocol['folds']:
        samples=[cache['records'][r][i] for r,i in cache['folds'][fold]['validation']]
        variants={};metadata={}
        for s in samples:
            names,extra=motion_group(s['observed'])
            for i,pid in enumerate(s['agent_ids']):
                groups=['all',names[i]]+[k for k,v in extra.items() if v[i]]
                metadata[s['window_id'],pid]=groups
        for epsilon in config['epsilon_m']:
            for rep in config['replicates']:
                changed=[]
                for s in samples:
                    obs=jitter_history(s['observed'],s['window_id'],s['agent_ids'],epsilon,rep)
                    relative=cv_prediction(obs,'CV-Last2')-obs[-1]-cv_prediction(s['observed'],'CV-Last2')+s['observed'][-1]
                    changed.append(dict(observed=obs,relation=real_relation(obs),original_relation=s['relation'],
                        input_rms=(obs.double()-s['observed'].double()).square().sum(-1).mean(0).sqrt(),cv_final_relative_shift=relative[-1].norm(dim=-1)))
                variants[epsilon,rep]=changed
        for role in ('budget30_best','budget100_best'):
            for kind in ('G0','G1'):
                path=(Path('results/tables/stage4p2')/fold/kind/'deterministic_scan/summary.json') if role=='budget30_best' else ROOT/'extension'/fold/kind/'summary.json'
                chosen=json.loads(path.read_text())['best'];assert sha256(chosen['checkpoint'])==chosen['checkpoint_sha256']
                model=GraphAttributionModel(kind).cuda().eval();model.load_state_dict(torch.load(chosen['checkpoint'],weights_only=False)['model']);model.requires_grad_(False)
                captured={}
                def capture(name):
                    def hook(module,args,output):captured[name]=(args[0].detach(),output.detach())
                    return hook
                for name,module in model.named_modules():
                    if isinstance(module,PersonTemporalNorm):
                        assert module.eps==1e-5;module.register_forward_hook(capture(name))
                folder=ROOT/'ptn'/role/fold;folder.mkdir(parents=True,exist_ok=True);writers={};checkpoint_start=time.perf_counter()
                with ExitStack() as stack,torch.no_grad():
                    for mode in MODES:
                        for epsilon in config['epsilon_m']:
                            stream=stack.enter_context(gzip.open(folder/f'{kind}_{mode}_eps{epsilon:g}.csv.gz','xt',newline=''));writers[mode,epsilon]=[stream,None]
                    for offset in range(0,len(samples),16):
                        part=samples[offset:offset+16];base_batch=observed_batch(part)
                        p=model(base_batch['v'],base_batch['relation'],base_batch['valid']);base=mean_positions(p,base_batch['last']);base_hooks=captured.copy()
                        base_relative=mean_positions(p,torch.zeros_like(base_batch['last']))
                        for epsilon in config['epsilon_m']:
                            for rep in config['replicates']:
                                perturbations=variants[epsilon,rep][offset:offset+16]
                                for mode in MODES:
                                    changed=[dict(s,relation=s['original_relation'] if mode=='fixed_history_graph' else s['relation']) for s in perturbations]
                                    b=observed_batch(changed);parameters=model(b['v'],b['relation'],b['valid']);forecast=mean_positions(parameters,b['last'])
                                    if not torch.isfinite(forecast).all():raise FloatingPointError('Nonfinite PTN stress output')
                                    sigma=parameters[...,2:4].exp();remaining=1-parameters[...,4].tanh().square()
                                    gaussian_valid=(((sigma>0)&torch.isfinite(sigma)).all(-1)&(remaining>0)&torch.isfinite(parameters).all(-1)).all(1).cpu()
                                    position=(forecast-base).norm(dim=-1)
                                    relative=(mean_positions(parameters,torch.zeros_like(b['last']))-base_relative).norm(dim=-1)
                                    layer={}
                                    for name,(before,after) in captured.items():
                                        old_in,old_out=base_hooks[name]
                                        ratio=(after-old_out).square().sum((1,2)).sqrt()/(before-old_in).square().sum((1,2)).sqrt().clamp_min(1e-20)
                                        low=(old_in.var(2,unbiased=False)<1e-5).float().mean(1)
                                        layer[name.replace('.','_')]=(ratio.cpu(),low.cpu())
                                    position=position.cpu();relative=relative.cpu()
                                    edge_changes=((b['relation']>0)!=(base_batch['relation']>0)).sum(1).sum(-1).cpu()
                                    for j,s in enumerate(part):
                                        for i,pid in enumerate(s['agent_ids']):
                                            inp=float(perturbations[j]['input_rms'][i]);final=float(relative[j,-1,i]);gain=final/max(inp,1e-20)
                                            row=dict(role=role,fold=fold,model=kind,graph_mode=mode,epsilon_m=epsilon,replicate=rep,window_id=s['window_id'],agent_id=pid,
                                                groups='|'.join(metadata[s['window_id'],pid]),input_rms_m=inp,mean_position_shift_m=float(position[j,:,i].mean()),
                                                final_position_shift_m=float(position[j,-1,i]),mean_relative_displacement_shift_m=float(relative[j,:,i].mean()),
                                                final_relative_displacement_shift_m=final,final_gain=gain,CV_Last2_final_relative_shift_m=float(perturbations[j]['cv_final_relative_shift'][i]),
                                                binary_receiver_edge_changes=int(edge_changes[j,i]),
                                                high_risk=bool(final>.1 and gain>100),gaussian_valid=bool(gaussian_valid[j,i]))
                                            for name,(ratios,low) in layer.items():row[name+'_local_gain']=float(ratios[j,i]);row[name+'_low_variance_fraction']=float(low[j,i])
                                            handle=writers[mode,epsilon]
                                            if handle[1] is None:
                                                handle[1]=csv.DictWriter(handle[0],fieldnames=list(row));handle[1].writeheader()
                                            handle[1].writerow(row)
                                            key=(role,fold,kind,mode,epsilon,s['window_id'],pid)
                                            if key not in worst:worst[key]=dict(row)
                                            else:
                                                previous=worst[key]
                                                for field in ('final_gain','mean_position_shift_m','final_position_shift_m','mean_relative_displacement_shift_m','final_relative_displacement_shift_m','CV_Last2_final_relative_shift_m'):
                                                    previous[field]=max(previous[field],row[field])
                                                for field in row:
                                                    if field.endswith('_local_gain'):previous[field]=max(previous[field],row[field])
                                                previous['high_risk']=previous['high_risk'] or row['high_risk']
                                                previous['gaussian_valid']=previous['gaussian_valid'] and row['gaussian_valid']
                provenance.append(dict(role=role,fold=fold,model=kind,checkpoint=chosen['checkpoint'],checkpoint_sha256=sha256(chosen['checkpoint']),
                    seconds=time.perf_counter()-checkpoint_start,targets=sum(len(s['agent_ids']) for s in samples),epsilon_unchanged=True))
                print(json.dumps(provenance[-1]),flush=True)
    buckets=defaultdict(list)
    for key,row in worst.items():
        for group in row['groups'].split('|'):buckets[key[:5]+(group,)].append(row)
    for (role,fold,kind,mode,epsilon,group),rows in sorted(buckets.items()):
        byid=defaultdict(list)
        for r in rows:byid[r['agent_id']].append(r['high_risk'])
        risk=float(np.mean([r['high_risk'] for r in rows]));summary=dict(role=role,fold=fold,model=kind,graph_mode=mode,epsilon_m=epsilon,group=group,
            target_windows=len(rows),unique_people=len(byid),replicates=2,aggregation='per-target maximum over2 fixed directions',high_risk_window_fraction=risk,
            person_macro_risk_fraction=float(np.mean([np.mean(v) for v in byid.values()])),people_with_any_risk=sum(any(v) for v in byid.values()),
            invalid_Gaussian_fraction=float(np.mean([not r['gaussian_valid'] for r in rows])),
            repeated_risk_evidence=bool(epsilon==.001 and risk>=.05 and len(byid)>=30))
        for field in ('final_gain','mean_relative_displacement_shift_m','final_relative_displacement_shift_m','CV_Last2_final_relative_shift_m',
                      'motion_norm_local_gain','post_norm_local_gain','decoder_norm_local_gain','motion_norm_low_variance_fraction','post_norm_low_variance_fraction'):
            values=[r[field] for r in rows]
            for label,quantile in (('P50',.5),('P95',.95),('P99',.99),('max',1.)):
                summary[field+'_'+label]=float(np.quantile(values,quantile))
        summaries.append(summary)
    present={(r['role'],r['fold'],r['model'],r['graph_mode'],r['epsilon_m'],r['group']) for r in summaries}
    for role in ('budget30_best','budget100_best'):
        for fold in protocol['folds']:
            for kind in ('G0','G1'):
                for mode in MODES:
                    for epsilon in config['epsilon_m']:
                        for group in ('all','stationary','low_speed','normal_walking','fast','exact_stationary','strict_stationary'):
                            if (role,fold,kind,mode,epsilon,group) not in present:
                                empty={k:None for k in summaries[0]};empty.update(role=role,fold=fold,model=kind,graph_mode=mode,epsilon_m=epsilon,group=group,target_windows=0,unique_people=0,replicates=2,repeated_risk_evidence=False)
                                summaries.append(empty)
    csv_new(ROOT/'ptn_group_summary.csv',summaries);csv_new(ROOT/'ptn_checkpoint_execution.csv',provenance)
    write_json_new(ROOT/'ptn_execution.json',dict(seconds=time.perf_counter()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256(__file__),cache_sha256=sha256(protocol['cache']),target_cases=len(worst),raw_rows=2*len(worst),future_GT_used=False,
        epsilon_modified=False,scope='full fixed-network response and local PTN amplification; not PTN-removal causal comparison; no iid-window inference',ETH_evaluated=False))


if __name__=='__main__':main()
