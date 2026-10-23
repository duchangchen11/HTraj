"""GT-free paired runtime study. Original numerical functions stay untouched."""
from .common import *
from .observer import KMeansObserver
from scripts.stage11b0.runtime import build_model,state_digest,original_modules
from scripts.stage11b0.training_adapter import make_templates
from scripts.stage11b1r.sampler import cluster_proposals,isolated_kmeans_rng
from scripts.stage11b1r.predictor import Predictor
from scripts.stage11b2c.adapter import AuditPredictor,digest
from scripts.stage11c1.preflight import core_sha
import torch,numpy as np,traceback,gc

IDENTITY_KEYS=['indices_SHA','proposal_SHA','centers_SHA','goal_SHA','path_SHA']

def save_pack(state,sid,tensors):
    directory=CACHE/'profile_tensors'/state;directory.mkdir(parents=True,exist_ok=True)
    path=directory/(hashlib.sha256(sid.encode()).hexdigest()+'.pt');assert not path.exists()
    torch.save(tensors,path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))

def main():
    frozen=verify_frozen();configure_torch();budget=Budget(float(os.environ['HTRAJ_PRIOR_GPU_SECONDS']),float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    budget.guard(1500);net,cfg=build_model(17,'cuda:0');assert core_sha(net)==INITIAL
    net.eval();templates=make_templates(cfg,'cuda:0');km=original_modules().model.evaluate.__globals__['kmeans']
    records=selected_records();table=shapes();source=json.loads((CONF/'runtime_protocol.json').read_text())
    semantics={}
    for scene,row in source['semantics'].items():
        assert sha(ROOT/row['path'])==row['file_SHA']
        semantics[scene]=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True)
        assert digest(semantics[scene])==row['tensor_SHA']
    diag=json.loads((OUT/'diagnostic_checkpoint_manifest.json').read_text());assert diag['status']=='COMPLETE'
    states=[dict(name='NoSemantic-initial',role='NoSemantic',checkpoint=diag['initial']),
            dict(name='NoSemantic-epoch001',role='NoSemantic',checkpoint=diag['epoch1'])]+source['semantic_states']
    predictor=AuditPredictor(net,cfg,templates);plain=Predictor(net,cfg,templates)
    runtime=[];iterations=[];equivalence=[];raw_times=[];cpu_rows=[];artifacts=[];failures=[];completed=[];state=None;current=None
    try:
        for state in states:
            # Early distributions are unknown; admit a whole fixed15 state, never drop a slow ID.
            estimate=600 if state['role']=='NoSemantic' else 300
            budget.guard(estimate)
            ck=state['checkpoint'];assert sha(ROOT/ck['path'])==ck['sha256']
            saved=torch.load(ROOT/ck['path'],map_location='cpu',weights_only=True)
            net.load_state_dict(saved['model'],strict=True);net.eval();del saved
            if state['name']=='NoSemantic-initial':assert core_sha(net)==INITIAL
            state_start=time.perf_counter();base={};cache={};monitored={}
            for mode in ['direct','observed','direct_repeat','observed_reverse']:
                order=list(reversed(records)) if mode=='observed_reverse' else records
                for record in order:
                    current=record;budget.guard(30);sid=record['sample_id'];scene=record['scene']
                    if scene not in cache:
                        cache[scene]=zero_semantic(scene,'cuda:0',table) if state['role']=='NoSemantic' else semantics[scene].cuda()
                    z=cache[scene]
                    before=rng_digest();torch.cuda.synchronize();allocated=torch.cuda.memory_allocated()
                    cpu_start=time.process_time();start=time.perf_counter();observer=None
                    if mode.startswith('observed'):
                        with KMeansObserver(km) as observer:p,g,t,pack=predictor.predict(sid,record['history'].unsqueeze(0),semantic=z)
                    else:p,g,t,pack=predictor.predict(sid,record['history'].unsqueeze(0),semantic=z)
                    before_sync=time.perf_counter();torch.cuda.synchronize();after_sync=time.perf_counter()
                    outer=after_sync-start;process_cpu=time.process_time()-cpu_start
                    after=rng_digest();assert before==after,'Predictor changed caller RNG'
                    assert t['segmentation_seconds']==0 and p.shape==(20,1,12,2) and g.shape==(20,1,2)
                    if mode=='direct':
                        base[sid]={k:t[k] for k in IDENTITY_KEYS};base[sid]['p']=p;base[sid]['g']=g
                        base[sid]['pack']=pack
                        artifacts.append(dict(state=state['name'],sample_id=sid,**save_pack(state['name'],sid,pack)))
                    else:
                        assert all(t[k]==base[sid][k] for k in IDENTITY_KEYS),'Observer/repeat changed indices/proposals/centers/goals/paths'
                        assert torch.equal(p,base[sid]['p']) and torch.equal(g,base[sid]['g'])
                        equivalence.append(dict(state=state['name'],sample_id=sid,mode=mode,bitwise=True,RNG_unchanged=True,
                            **{k:t[k] for k in IDENTITY_KEYS}))
                    row=dict(state=state['name'],role=state['role'],sample_id=sid,scene=scene,mode=mode,
                        checkpoint_SHA=ck['sha256'],outer_prediction_seconds=outer,cpu_process_seconds=process_cpu,
                        wall_before_completion_sync=before_sync-start,completion_sync_seconds=after_sync-before_sync,
                        allocated_before_bytes=allocated,allocated_after_bytes=torch.cuda.memory_allocated(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**t)
                    runtime.append(row)
                    if observer is not None:
                        observed=observer.result();iterations.append(dict(state=state['name'],sample_id=sid,scene=scene,mode=mode,
                            checkpoint_SHA=ck['sha256'],**observed))
                        if sid in monitored:assert monitored[sid]['iterations']==observed['iterations'] and monitored[sid]['stop_reason']==observed['stop_reason']
                        monitored[sid]=observed
                    print('PROFILE',state['name'],mode,sid,'KMeans',round(t['kmeans_seconds'],4),'elapsed',round(budget.elapsed(),2),flush=True)
            # Demonstrate the passive-export adapter matches the frozen predictor, not just itself.
            first=records[0];sid=first['sample_id'];p,g,t=plain.predict(sid,first['history'].unsqueeze(0),semantic=cache[first['scene']])
            assert torch.equal(p,base[sid]['p']) and torch.equal(g,base[sid]['g'])
            equivalence.append(dict(state=state['name'],sample_id=sid,mode='immutable_original_predictor',bitwise=True,RNG_unchanged=True,
                **{k:base[sid][k] for k in IDENTITY_KEYS}))
            for record in records:
                current=record;sid=record['sample_id'];budget.guard(30);pack=base[sid]['pack'];points=pack['proposals_network'][0,0].cuda()
                assert digest(points)==digest(pack['proposals_network'][0,0])
                # No Python profile is active for this exact original KMeans timing.
                before=rng_digest();torch.cuda.synchronize();cstart=time.process_time();start=time.perf_counter()
                centers=cluster_proposals(points,sid);returned=time.perf_counter();torch.cuda.synchronize();end=time.perf_counter();cpu_elapsed=time.process_time()-cstart
                assert rng_digest()==before and torch.equal(centers.cpu(),pack['centers_network'])
                n=monitored[sid]['iterations']
                raw_times.append(dict(state=state['name'],sample_id=sid,scene=record['scene'],checkpoint_SHA=ck['sha256'],
                    method='ORIGINAL_MIXED_DEVICE_NO_PROFILE',iterations=n,wall_seconds=end-start,
                    cpu_process_seconds=cpu_elapsed,return_before_final_sync_seconds=returned-start,
                    final_sync_seconds=end-returned,seconds_per_iteration=(end-start)/n,
                    explicit_D2H_payload_bytes_per_iteration=(10000*2+19*2)*4,
                    explicit_D2H_payload_bytes=(10000*2+19*2)*4*n,centers_bitwise_original=True))
                # Isolated diagnostic counterfactual; never fed back into the model.
                cpu_points=pack['proposals_network'][0,0];before=rng_digest();cstart=time.process_time();start=time.perf_counter()
                with isolated_kmeans_rng(sid,torch.device('cpu')):
                    choices,cpu_centers=km(X=cpu_points,num_clusters=19,distance='euclidean',device=torch.device('cpu'),
                        tqdm_flag=False,tol=.001,iter_limit=1000)
                cpu_seconds=time.perf_counter()-start;process_seconds=time.process_time()-cstart
                assert rng_digest()==before;profile_start=time.perf_counter()
                with KMeansObserver(km) as cpu_observer:
                    with isolated_kmeans_rng(sid,torch.device('cpu')):
                        again_choices,again_centers=km(X=cpu_points,num_clusters=19,distance='euclidean',device=torch.device('cpu'),
                            tqdm_flag=False,tol=.001,iter_limit=1000)
                profiled_cpu_seconds=time.perf_counter()-profile_start
                assert rng_digest()==before and torch.equal(cpu_centers,again_centers) and torch.equal(choices,again_choices)
                ci=cpu_observer.result();difference=(cpu_centers-pack['centers_network']).abs()
                cpu_rows.append(dict(state=state['name'],sample_id=sid,scene=record['scene'],method='CPU_RESIDENT_DIAGNOSTIC_NO_PROFILE',
                    wall_seconds=cpu_seconds,cpu_process_seconds=process_seconds,iterations=ci['iterations'],
                    profile_wall_seconds=profiled_cpu_seconds,CPU_monitor_equivalence_bitwise=True,
                    seconds_per_iteration=cpu_seconds/ci['iterations'],centers_bitwise=torch.equal(cpu_centers,pack['centers_network']),
                    center_max_abs_difference=float(difference.max()),center_mean_abs_difference=float(difference.mean()),
                    CPU_centers_SHA=digest(cpu_centers),original_centers_SHA=digest(pack['centers_network']),
                    original_proposals_SHA=digest(cpu_points),not_formal_predictor=True,not_used_for_paths=True,
                    comparison_caveat='CPU center reductions differ from CUDA; not an isolated transfer-time experiment'))
            completed.append(dict(state=state['name'],targets=15,passes=4,seconds=time.perf_counter()-state_start))
            print('STATE_COMPLETE',completed[-1],flush=True);del base,cache;gc.collect()
    except Exception as e:
        traceback.print_exc();failures.append(dict(state=state['name'] if state else None,sample_id=current['sample_id'] if current else None,
            error_type=type(e).__name__,error=str(e)));raise
    finally:
        for name,rows in [('kmeans_iteration_counts.csv',iterations),('kmeans_runtime.csv',raw_times),
            ('all_prediction_runtime.csv',runtime),('cpu_resident_diagnostic.csv',cpu_rows),('tensor_artifact_manifest.csv',artifacts)]:
            if rows:cnew(OUT/name,rows)
        for name,key in [('semantic_checkpoint_runtime.csv','Semantic'),('nosemantic_initial_runtime.csv','NoSemantic-initial'),('nosemantic_epoch1_runtime.csv','NoSemantic-epoch001')]:
            rows=[r for r in runtime if (r['role']=='Semantic' if key=='Semantic' else r['state']==key)]
            if rows:cnew(OUT/name,rows)
        if equivalence:cnew(OUT/'equivalence_details.csv',equivalence)
        jnew(OUT/'instrumentation_equivalence.json',dict(passed=not failures and len(completed)==6,
            complete_states=completed,expected_states=6,expected_targets_per_state=15,comparisons=len(equivalence),
            all_observed_checks_bitwise=all(r['bitwise'] for r in equivalence),random_state_unchanged=True,
            failures=failures,formal_predictor_modified=False,test_loaded=False,GT_loaded=False,metrics_computed=False))
        jnew(OUT/'profile_execution.json',dict(status='COMPLETE' if len(completed)==6 and not failures else 'INCOMPLETE',
            completed_states=completed,failures=failures,GPU_cumulative_seconds=budget.elapsed(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
if __name__=='__main__':main()
