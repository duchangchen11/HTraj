"""All998 CPU-only representative selection, with no GT/error table imports."""
from .common import *
from .representative import select,project_record
import numpy as np
import pandas as pd
import torch

def main():
    cpu_only();start=time.perf_counter();frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(CONF/'experiment_protocol.json')==frozen['protocol_SHA']
    manifest=pd.read_csv(OLD/'tensor_manifest.csv');pop=pd.read_csv(ROOT/'results/tables/stage11b2d/cluster_diagnostics.csv',usecols=['sample_id','center_index','Population']).set_index(['sample_id','center_index'])
    goals=[];ids=[];metadata=[];rows=[];summaries=[]
    for i,r in enumerate(manifest.to_dict('records')):
        assert sha(ROOT/r['path'])==r['sha256'];pack=torch.load(ROOT/r['path'],map_location='cpu',weights_only=True)
        sid,P,C,soft=project_record(pack);g,a,heads,cost=select(P,C,soft)
        assert sid==r['sample_id'] and len(a)==10000 and sum(h['population'] for h in heads)==10000
        for h in heads:
            assert h['population']==int(pop.loc[(sid,h['center_index']),'Population'])
            if h['proposal_index']>=0:
                assert a[h['proposal_index']]==h['center_index'] and np.array_equal(g[h['head']],P[h['proposal_index']])
            h.update(sample_id=sid,scene=r['scene'],assignment='FINAL_CENTER_VORONOI_ASSIGNMENT')
        tg=torch.from_numpy(g.copy());old=pack['tensors']['goals'][:,0]
        assert torch.equal(tg[0],old[0]);change=~np.all(g==old.numpy(),axis=-1)
        info=dict(sample_id=sid,scene=r['scene'],original_goal_SHA=r['goal_SHA'],alternative_goal_SHA=digest(tg[:,None]),
            proposal_SHA=r['proposal_SHA'],centers_SHA=r['centers_SHA'],indices_SHA=r['indices_SHA'])
        sums=dict(**info,changed_heads=int(change.sum()),all20_goals_unchanged=not bool(change.any()),
            unique_goals=len(np.unique(g,axis=0)),duplicate_goals=20-len(np.unique(g,axis=0)),head0_bitwise=True,
            mean_changed_distance_all20=float(np.linalg.norm(g.astype(float)-old.numpy().astype(float),axis=-1).mean()),
            mean_center_displacement19=float(np.mean([h['displacement'] for h in heads])),**cost)
        rows.extend(heads);summaries.append(sums);metadata.append(info);goals.append(tg);ids.append(sid)
        if (i+1)%200==0:print('CPU_SELECT',i+1,'/998',flush=True)
    algorithm_seconds=time.perf_counter()-start;reverse_start=time.perf_counter()
    # Full reverse repetition validates input-order independence without changing RNG.
    for i in range(len(ids)-1,-1,-1):
        r=manifest.iloc[i];pack=torch.load(ROOT/r.path,map_location='cpu',weights_only=True);sid,P,C,s=project_record(pack)
        alt,_,_,_=select(P,C,s);assert np.array_equal(alt,goals[i].numpy())
    counter=[]
    for i in [0,499,997]:
        r=manifest.iloc[i];p=torch.load(ROOT/r.path,map_location='cpu',weights_only=True)
        view0=project_record(p)
        p['future_GT']=torch.full((12,2),1e6);p['oracle_head']=-999;p['A25_failure_label']=not True
        view1=project_record(p)
        assert view0[0]==view1[0] and all(np.array_equal(x,y) for x,y in zip(view0[1:],view1[1:]))
        actual,_,_,_=select(*view1[1:]);assert np.array_equal(actual,goals[i].numpy())
        counter.append(dict(sample_id=r.sample_id,raw_record_future_tags_changed=True,selection_inputs_bitwise=True,representatives_bitwise=True))
    reverse_seconds=time.perf_counter()-reverse_start;paths=CACHE/'alternative_goals.pt';assert not paths.exists();t=time.perf_counter()
    tensor=torch.stack(goals,dim=1);assert tensor.shape==(20,998,2) and torch.isfinite(tensor).all()
    torch.save(dict(sample_ids=ids,goals=tensor,checkpoint_SHA=CK,metadata=metadata,
      partition_protocol_SHA=frozen['protocol_SHA'],algorithm_config_SHA=sha(Path(__file__).with_name('representative.py'))),paths)
    save_seconds=time.perf_counter()-t
    csvnew(OUT/'representative_selection.csv',rows);csvnew(OUT/'goal_change_summary.csv',summaries)
    jnew(OUT/'selection_audit.json',dict(passed=True,targets=998,all_provenance_and_population_verified=True,
      new_goals_path=str(paths.relative_to(ROOT)),new_goals_file_SHA=sha(paths),goals_shape=list(tensor.shape),
      all998_reverse_bitwise=True,GT_counterfactual=counter,actual_GT_loaded=False,no_sampler_or_kmeans_called=True,
      all_head0_bitwise=True,all_coordinates_from_original_cell=True,algorithm_seconds=algorithm_seconds,
      reverse_and_counterfactual_seconds=reverse_seconds,save_seconds=save_seconds,cache_bytes=paths.stat().st_size,
      assignment_seconds=sum(r['assignment_seconds'] for r in summaries),representative_seconds=sum(r['representative_seconds'] for r in summaries),
      total_process_seconds=time.perf_counter()-start,CUDA_initialized=False,source_commit=commit()))
    print('CPU_SELECTION_AND_ISOLATION_PASS',998,flush=True)

if __name__=='__main__':main()
