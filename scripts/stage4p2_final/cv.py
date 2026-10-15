"""Fresh analytical evaluation on exactly the Stage4.2 frozen validation targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,gzip,json,time,hashlib,subprocess
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.evaluation.graph_final_audit import cv_prediction,errors,motion_group


def main():
    torch.set_num_threads(1);root=Path('results/tables/stage4p2_final');config=json.loads(Path('configs/stage4p2_final/protocol.json').read_text())
    assert sha256(config['cache'])==config['cache_sha256'];cache=torch.load(config['cache'],weights_only=False);rows=[];metrics=[];identities=[]
    for fold in config['folds']:
        samples=[cache['records'][r][i] for r,i in cache['folds'][fold]['validation']]
        observed=torch.cat([s['observed'] for s in samples],1);future=torch.cat([s['future'] for s in samples],1);scores={}
        for kind in ('CV-Last2','CV-OLS4','CV-OLS8'):
            start=time.perf_counter();prediction=cv_prediction(observed,kind);inference=time.perf_counter()-start
            start=time.perf_counter();aa,ff=errors(prediction,future);evaluation=time.perf_counter()-start
            scores[kind]=(aa.tolist(),ff.tolist());metrics.append(dict(fold=fold,model=kind,target_windows=observed.shape[1],windows=len(samples),ADE_m=float(aa.mean()),FDE_m=float(ff.mean()),
                parameters=0,inference_seconds=inference,evaluation_seconds=evaluation,device='cpu',dtype='float64',computed_fresh=True))
        offset=0
        for s in samples:
            names,extra=motion_group(s['observed'])
            for i,pid in enumerate(s['agent_ids']):
                history_sha=hashlib.sha256(s['observed'][:,i].contiguous().numpy().tobytes()).hexdigest();future_sha=hashlib.sha256(s['future'][:,i].contiguous().numpy().tobytes()).hexdigest()
                groups=list(s['features'][i]['groups'])+['ptn_'+names[i]]
                for name,mask in extra.items():
                    if mask[i]:groups.append(name)
                row=dict(fold=fold,recording=s['recording'],window_id=s['window_id'],agent_id=pid,history_sha256=history_sha,future_sha256=future_sha,
                    groups='|'.join(groups),**{k:v for k,v in s['features'][i].items() if k!='groups'})
                for kind,(aa,ff) in scores.items():row[kind+'_ADE_m']=aa[offset+i];row[kind+'_FDE_m']=ff[offset+i]
                rows.append(row);identities.append({k:row[k] for k in ('fold','window_id','agent_id','history_sha256','future_sha256')})
            offset+=len(s['agent_ids'])
        assert offset==observed.shape[1]
    with gzip.open(root/'fresh_cv_targets.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_new(root/'fresh_cv_metrics.csv',metrics);csv_new(root/'target_identity_hashes.csv',identities)
    write_json_new(root/'cv_execution.json',dict(cache_sha256=sha256(config['cache']),target_identity_table_sha256=sha256(root/'target_identity_hashes.csv'),
        rows=len(rows),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        helper_sha256=sha256('src/evaluation/graph_final_audit.py'),no_stage3p8_metrics_used=True,ETH_evaluated=False))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
