"""Read-only history analysis and previous-checkpoint response probes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
from collections import Counter,defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training.order_verification import datasets
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256,write_json_new
from src.models.ego_encoder import motion_inputs
from src.models.relation_routing import receiver_candidates
from src.models.temporal_graph import select_candidates
from src.models.order_verification import build_order_model
from src.training.social_diagnostics import csv_new


def analyze(data,split):
    counts=Counter();targets=[];relations=[];runs=[]
    for batch in DataLoader(data,batch_size=8,collate_fn=collate_trajectories):
        encoded=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps'])
        new=[];old=[]
        for t in range(8):
            a,f=receiver_candidates(encoded,t);new.append(a)
            old.append(select_candidates(encoded['positions'][:,t],encoded['velocity'][:,t],encoded['continuity'][:,t])[0])
        new,old=torch.stack(new,1),torch.stack(old,1)
        length=encoded['valid'].sum(1)
        counts.update(visible_targets=int(encoded['valid'][:,-1].sum()),old_edges_all_history=int(old.sum()),new_edges_all_history=int(new.sum()),blocked_receives_all_history=int((new & ~old).sum()))
        for b,sample in enumerate(batch['sample_id']):
            for i in encoded['valid'][b,-1].nonzero().flatten().tolist():
                sources=new[b,-1,i].nonzero().flatten(); one=int(length[b,i])==1
                row=dict(split=split,sample_id=sample,target_id=int(batch['agent_ids'][b,i]),history_frames=int(length[b,i]),
                    new_entry=not bool(encoded['continuity'][b,-1,i]),reliable_neighbors=len(sources),
                    full8_neighbors=int((length[b,sources]==8).sum()),blocked_edges=int((new[b,-1,i] & ~old[b,-1,i]).sum()))
                targets.append(row)
                if one:
                    counts.update(one_history_targets=1,one_history_with_neighbors=int(len(sources)>0),one_history_without_neighbors=int(len(sources)==0),
                        one_history_neighbor_edges=len(sources),one_history_full8_neighbor_edges=row['full8_neighbors'],one_history_blocked_cutoff_edges=row['blocked_edges'])
            for i,j in new[b].any(0).nonzero().tolist():
                active=new[b,:,i,j]; indices=active.nonzero().flatten().tolist();streak=0
                for value in active.tolist()+[False]:
                    if value:streak+=1
                    elif streak:runs.append(streak);streak=0
                motion=encoded['velocity'][b,:,j]
                change=(motion[1:]-motion[:-1]).norm(dim=-1)
                valid_change=encoded['continuity'][b,1:,j] & encoded['continuity'][b,:-1,j]
                relations.append(dict(split=split,sample_id=sample,target_id=int(batch['agent_ids'][b,i]),source_id=int(batch['agent_ids'][b,j]),
                    active_frames=len(indices),first_frame=indices[0],last_frame=indices[-1],
                    source_motion_change_mean=float(change[valid_change].mean()) if valid_change.any() else None))
    counts['one_history_full8_neighbor_fraction']=counts['one_history_full8_neighbor_edges']/max(counts['one_history_neighbor_edges'],1)
    counts['relation_pairs']=len(relations);counts['mean_pair_active_frames']=float(np.mean([r['active_frames'] for r in relations]))
    counts['mean_contiguous_edge_streak']=float(np.mean(runs));counts['edge_streak_counts']=dict(Counter(runs))
    return dict(counts),targets,relations


@torch.no_grad()
def probes(val):
    model=build_order_model('TOHI-H1');path=Path('experiments/stage2p6/new/TOHI-H1/seed_17/checkpoints/best.pt')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model.eval()
    weights=defaultdict(list);changes=[]
    for batch in DataLoader(val,batch_size=8,collate_fn=collate_trajectories):
        x,m,t=batch['observed'],batch['observed_valid'],batch['timestamps']
        p,d=model(x,m,t,return_debug=True);B,T,N,_=x.shape
        for step,f in enumerate(d['graph_frames']):
            scene=f.target//N;i=f.target%N;j=f.source%N
            v=d['velocity'][:,step];a=v[scene,i];z=v[scene,j]
            cosine=(a*z).sum(-1)/(a.norm(dim=-1)*z.norm(dim=-1)).clamp_min(1e-6)
            moving=(a.norm(dim=-1)>.2)&(z.norm(dim=-1)>.2)
            bins=dict(co_moving=moving & (cosine>.7),opposing=moving & (cosine<-.7),crossing=moving & (cosine.abs()<=.7),stationary_or_slow=~moving)
            for name,mask in bins.items():weights[name].extend(f.weight[mask].tolist())
        # Neighbor input probe: only fully observed source people (every third
        # history-ID slot) are made stationary around their current observed
        # location. Score prediction shifts on OTHER unchanged targets only.
        chosen=(m.sum(1)==8) & (torch.arange(N)[None]%3==0)
        z=torch.where(chosen[:,None,:,None],x[:,-1,None],x)
        q=model(z,m,t);eligible=m[:,-1] & ~chosen
        changes.extend((q-p).norm(dim=-1)[eligible[:,None].expand(-1,12,-1)].tolist())
    return dict(checkpoint=str(path),checkpoint_sha256=sha256(path),alpha_by_historical_motion={k:dict(edges=len(v),mean=float(np.mean(v)),p95=float(np.quantile(v,.95))) for k,v in weights.items()},
        neighbor_stationary_probe_prediction_change_L2_m=float(np.mean(changes)),future_gt_used=False,
        interpretation='History-only frozen inference perturbation; movement bins confounded by geometry/degree; input intervention OOD and chosen sources can also alter candidate ranking. Not causal efficacy or independent training evidence.')


def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage3/protocol.json').read_text());train,val=datasets(p)
    root=Path('results/tables/stage3/problem_analysis');root.mkdir(parents=True,exist_ok=True);summary={}
    for name,data in [('train',train),('development',val)]:
        stats,targets,relations=analyze(data,name);summary[name]=stats
        csv_new(root/(name+'_targets.csv'),targets);csv_new(root/(name+'_relations.csv'),relations)
    summary['frozen_history_probes']=probes(val)
    summary.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        data_manifest_sha256=sha256(p['data_manifest']),subset_manifest_sha256=sha256(p['subset_manifest']),test_evaluated=False,history_only=True)
    write_json_new(root/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
