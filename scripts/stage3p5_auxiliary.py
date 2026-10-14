"""Frozen pilot graph dependence only; no extra training or ETH evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
import torch
from src.training.benchmark_audit import data_load,evaluate
from src.training.social_diagnostics import csv_new
from src.models.social_diagnostics_models import DIAGNOSTIC_MODELS
from src.data.preprocessing import sha256,write_json_new


@torch.no_grad()
def main():
    torch.set_num_threads(1);root=Path('results/tables/stage3p5');p=json.loads(Path('configs/stage3p5/protocol.json').read_text());data=data_load(p);rows=[]
    for view,(_,batches) in data.items():
        path=Path('experiments/stage3p5/pilot')/view/'Dynamic-GNN-RN/checkpoints/best.pt';model=DIAGNOSTIC_MODELS['Dynamic-GNN-RN']();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model.eval()
        ref=evaluate(model,batches,torch.device('cpu'),rows=True);rmap={(r['sample_id'],r['target_id']):r for r in ref['targets']}
        for case,options in [('full',None),('no_messages',dict(remove_messages=True)),('radius_random',dict(shuffle_graph=True,shuffle_seed=1735))]:
            result=evaluate(model,batches,torch.device('cpu'),intervention=options,rows=True);changed=total=0
            for b in batches:
                _,d=model(b['observed'],b['observed_valid'],b['timestamps'],return_debug=True,intervention=options)
                n=b['observed'].shape[0]*b['observed'].shape[2]
                for old,new in zip(d['original_graph_frames'],d['graph_frames']):
                    oldkey=old.target*n+old.source;newkey=new.target*n+new.source;total+=len(oldkey);changed+=int((~torch.isin(newkey,oldkey)).sum()) if case!='no_messages' else len(oldkey)
            diffs=[r['ADE_m']-rmap[(r['sample_id'],r['target_id'])]['ADE_m'] for r in result['targets']]
            rows.append(dict(protocol=view,model='Dynamic-GNN-RN',case=case,ADE_m=result['ADE_m'],FDE_m=result['FDE_m'],delta_ADE_m=result['ADE_m']-ref['ADE_m'],
                delta_FDE_m=result['FDE_m']-ref['FDE_m'],mean_same_target_ADE_delta=sum(diffs)/len(diffs),original_edges=total,changed_or_removed_edges=changed,
                changed_or_removed_fraction=changed/total if total else 0,independent_training=False,checkpoint_sha256=sha256(path),ETH_evaluated=False))
    csv_new(root/'pilot_fixed_graph_interventions.csv',rows);write_json_new(root/'auxiliary_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        scope='fixed best checkpoint CPU OOD/coadaptation dependence; not independent social-value or capacity proof',ETH_evaluated=False))


if __name__=='__main__':main()
