"""Freeze two recording-isolated datasets, matched operators and decision gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.public_training.data import RECORDINGS,select_window
from src.graph_attribution.data import real_relation,controlled_relation,history_groups
from src.graph_attribution.model import GraphAttributionModel
from src.training.social_diagnostics import csv_new

CONF=Path('configs/stage4p2');ROOT=Path('results/tables/stage4p2');HELD=('crowds_zara02','biwi_hotel')


def graph_statistics(samples,fold,split):
    totals=defaultdict(float);degrees1=[];degrees2=[];distances1=[];distances2=[];rows=[]
    for s in samples:
        r,c=s['relation'],s['control'];a,b=r>0,c>0
        d=(s['observed'][:,:,None]-s['observed'][:,None,:]).norm(dim=-1)
        totals['directed_edges']+=int(a.sum());totals['replaced_directed_edges']+=int((b&~a).sum())
        totals['weighted_half_l1']+=float((c-r).abs().sum()/2);totals['real_mass']+=float(r.sum())
        totals['history_graphs']+=8;totals['nonempty_graphs']+=int(a.any(-1).any(-1).sum())
        totals['binary_changed_graphs']+=int((a!=b).any(-1).any(-1).sum());totals['control_edges_over4m']+=int((b&(d>4)).sum())
        degrees1.extend(a.sum(-2).flatten().tolist());degrees2.extend(b.sum(-2).flatten().tolist())
        distances1.extend(d[a].tolist());distances2.extend(d[b].tolist())
        assert torch.equal(a.sum(-1),b.sum(-1)) and not c.diagonal(dim1=-2,dim2=-1).any()
    row=dict(fold=fold,split=split,windows=len(samples),**totals,
        edge_replacement_fraction=totals['replaced_directed_edges']/max(totals['directed_edges'],1),
        weighted_half_l1_fraction=totals['weighted_half_l1']/max(totals['real_mass'],1),
        changed_graph_fraction=totals['binary_changed_graphs']/max(totals['nonempty_graphs'],1),
        control_over4m_fraction=totals['control_edges_over4m']/max(totals['directed_edges'],1),
        receiver_degrees_exact=True,self_coefficient=1.,neighbor_row_mass_when_active=.25,
        real_incoming_degree_mean=float(np.mean(degrees1)),real_incoming_degree_p95=float(np.quantile(degrees1,.95)),
        control_incoming_degree_mean=float(np.mean(degrees2)),control_incoming_degree_p95=float(np.quantile(degrees2,.95)),
        real_distance_mean_m=float(np.mean(distances1)) if distances1 else None,real_distance_p95_m=float(np.quantile(distances1,.95)) if distances1 else None,
        control_distance_mean_m=float(np.mean(distances2)) if distances2 else None,control_distance_p95_m=float(np.quantile(distances2,.95)) if distances2 else None)
    return row


def main():
    torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert base=='ce77c49c68367b169bc6d8f7a29bb4cba4400575'
    paths=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))-{''}
    previous=json.loads(Path('configs/stage4p1/protected_artifacts.json').read_text());paths|={r['path'] for r in previous['files']}
    inventory=list(__import__('csv').DictReader(Path('results/tables/stage4p1/checkpoint_inventory.csv').open()))
    paths|={r['path'] for r in inventory};paths.add('data/processed/stage4p1/A_windows.pt')
    paths|={str(Path('data/raw/eth_ucy')/(r+'.txt')) for r in RECORDINGS}
    write_json_new(CONF/'protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
    old=json.loads(Path('configs/stage3p5/folds/val_crowds_zara02.json').read_text());sources={};records={};t=time.perf_counter()
    for rec in RECORDINGS:
        path=Path('data/raw/eth_ucy')/(rec+'.txt');raw=np.loadtxt(path);sources[rec]=dict(path=str(path),sha256=sha256(path))
        assert sources[rec]['sha256']==old['source_files'][rec]['sha256']
        by=defaultdict(dict)
        for frame,pid,x,y in raw:by[int(frame)][int(pid)]=[x,y]
        frames=np.unique(raw[:,0]).astype(int);samples=[];cutoff=old['cutoff_by_recording'][rec]['train_end_exclusive']
        for k in range(len(frames)-19):
            span=frames[k:k+20]
            if not np.all(np.diff(span)==10) or (rec not in HELD and span[-1]>=cutoff):continue
            s=select_window(by,span,'A_offline_complete20')
            if s is None:continue
            s.update(recording=rec,window_id=f'{rec}:{span[0]}',train_eligible=bool(span[-1]<cutoff))
            r=real_relation(s['observed']);s['relation']=r;s['control']=controlled_relation(r,s['agent_ids'],s['window_id'])
            history_ids=set.intersection(*(set(by[int(f)]) for f in span[:8]));visible=[];complete=[]
            for j,frame in enumerate(span[:8]):
                ids=list(by[int(frame)]);pos=torch.tensor(np.round([by[int(frame)][i] for i in ids],4),dtype=torch.float32)
                d=(s['observed'][j,:,None]-pos[None]).square().sum(-1)
                not_self=torch.tensor([[pid!=other for other in ids] for pid in s['agent_ids']])
                visible.append(((d<=16)&not_self).sum(-1));legal=torch.tensor([i in history_ids for i in ids])
                complete.append(((d<=16)&not_self&legal).sum(-1))
            s['features']=history_groups(s['observed'],r,torch.stack(visible),torch.stack(complete))
            if rec in HELD:
                seed=int.from_bytes(hashlib.sha256(f'314159:{s["window_id"]}'.encode()).digest()[:8],'little')%(2**63-1)
                s['noise']=torch.randn(20,12,len(s['agent_ids']),2,generator=torch.Generator().manual_seed(seed))
            samples.append(s)
        records[rec]=samples
    folds={};counts=[];graph_rows=[];manifest_infos={}
    for held in HELD:
        folds[held]={split:[(rec,i) for rec,samples in records.items() for i,s in enumerate(samples)
            if (split=='validation' and rec==held) or (split=='train' and rec!=held and s['train_eligible'])] for split in ('train','validation')}
        manifest=dict(protocol='A_offline_complete20',held_recording=held,source_files=sources,dt_seconds=.4,frame_step=10,
            old_validation_promoted_to_train=False,independent_confirmation=False,ETH_evaluated=False,
            splits={split:[dict(recording=rec,window_id=records[rec][i]['window_id'],frames=records[rec][i]['frames'],agent_ids=records[rec][i]['agent_ids']) for rec,i in pairs] for split,pairs in folds[held].items()})
        path=CONF/'manifests'/(held+'.json');write_json_new(path,manifest);manifest_infos[held]=dict(path=str(path),sha256=sha256(path))
        for split,pairs in folds[held].items():
            samples=[records[rec][i] for rec,i in pairs]
            counts.append(dict(fold=held,split=split,windows=len(samples),target_windows=sum(len(s['agent_ids']) for s in samples),recordings='|'.join(sorted({s['recording'] for s in samples}))))
            graph_rows.append(graph_statistics(samples,held,split))
    cache=Path('data/processed/stage4p2/complete_windows.pt');cache.parent.mkdir(parents=True,exist_ok=True);assert not cache.exists();torch.save(dict(records=records,folds=folds),cache)
    csv_new(ROOT/'split_counts.csv',counts);csv_new(ROOT/'graph_control_statistics.csv',graph_rows)
    parameters={}
    for kind in ('G0','G1','G2'):
        torch.manual_seed(17);parameters[kind]=sum(p.numel() for p in GraphAttributionModel(kind).parameters())
    protocol=dict(stage='4.2',base_commit=base,upstream_frozen='333d3a57b4d2705e129b21aefefa09c79b2b9ae1',
        family='new small controlled temporal-CNN family inspired by public graph prediction; not the frozen Social-STGCNN architecture, not novel method',
        folds=list(HELD),models=['G0','G1'],conditional_model='G2',seed=17,parameters=parameters,width=16,
        data_protocol='A_offline_complete20',obs_len=8,pred_len=12,dt=.4,coordinate_unit='m',ETH_evaluated=False,
        train_policy='other four nonETH original train regions only; held recording excluded entirely; original other val+embargo unused',
        selection='minimum validation target-weighted K1 Gaussian mean ADE, earliest exact tie; paired same-checkpoint FDE',
        previous_selection='Stage4.1 validation NLL; never claim same-selection comparison',
        loss='unchanged official PDF-clamped Gaussian NLL; vectorized exact formula with per-window means',
        optimizer='SGD',lr=.01,momentum=0.,weight_decay=0.,scheduler=None,clip_grad=1.,batch_windows=16,accumulation_windows=64,
        max_epochs=30,smoke_epochs=5,smoke_gate='all finite metrics/loss/gradients, nonzero gradient, checkpoint exact restore; continue same run, no validation improvement requirement',
        normalization='shared PersonTemporalNorm: per node/channel observed/predicted time-axis only, eps1e-5, train=eval, no running moments or cross-node stats',
        graph=dict(self_coefficient=1.,gamma=.25,radius_m=4.,sigma_m=2.,weights='exp(-distance2/(2*sigma2)), nonself within4m, offdiag row-normalized only; empty row0',
            control='same-window/co-observed/nonself sources; same receiver degree/weight multiset, stable ID-keyed random source order;4m prior relaxed intentionally',control_seed=271828,
            confound='G2 distance distribution/source-degree changes; not distance-matched proof; G1 vsG0 no diagonal coefficient or normalization architecture change'),
        evaluation='all three K modes every epoch; independent K20 minima; common CPU standard-normal20 bank per window SHA(seed314159:window_id), batch padding independent',
        sampling_seed=314159,cpu_threads=1,device='cuda',cache=str(cache),cache_sha256=sha256(cache),manifests=manifest_infos,
        audit='old Full29/30 trainable weights x BN-running buffers factorial, all921 old Zara2 validation windows, fixed checkpoint inference only',
        worthwhile_ADE='G1 reduction>=max(.005m,.01*G0_ADE)',FDE_tolerance='G1 increase<=max(.01m,.02*G0_FDE)',
        extra_seed_action='recommend42/2026 only if both folds pass worthwhile ADE and FDE gate; do not run them in Stage4.2',
        G2_training_gate='only if both folds worthwhile+FDE-safe, and each validation fold edge replacement>=.25 and changed-nonempty-graph fraction>=.5; then train G2 on both folds, same budget; otherwise structural-only/incomplete performance',
        no_more_hyperparameter_or_structure_retries=True,auto_push=False)
    write_json_new(CONF/'protocol.json',protocol)
    write_json_new(ROOT/'preparation.json',dict(seconds=time.perf_counter()-t,script_sha256=sha256(__file__),source_commit=base,sources=sources,
        python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),environment=sys.prefix))
    print(json.dumps(dict(counts=counts,parameters=parameters,seconds=time.perf_counter()-t,control=graph_rows)))


if __name__=='__main__':main()
