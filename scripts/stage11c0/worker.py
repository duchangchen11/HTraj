"""GT-free frozen model worker. Passive audits do not alter network operators."""
from .common import *
from .interventions import intervene
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics,trajectory_maps
from scripts.stage11b1.common import configure_torch
from scripts.stage11b2c.adapter import AuditPredictor
from scripts.stage11b2e.decoder import decode
from contextlib import contextmanager
import torch,numpy as np,traceback

@contextmanager
def input_guard(net,history,semantic,templates):
    h,w=semantic.shape[-2:];expected=trajectory_maps(history.cuda()*.25,h,w,templates[0]);calls=[]
    def hook(module,args):
        x=args[0];assert x.shape[1]==14 and torch.equal(x[:,:6],semantic) and torch.equal(x[:,6:],expected)
        calls.append(True)
    handle=net.encoder.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()
    assert calls and torch.isfinite(expected).all()

@torch.no_grad()
def probed_decode(net,cfg,templates,history,semantic,goals):
    features=[];logits=[]
    def eh(module,args,out):
        features.extend([dict(SHA=digest(v),shape=list(v.shape),norm=float(v.norm()),mean=float(v.mean())) for v in out])
    def ph(module,args,out):logits.append(dict(SHA=digest(out),norm=float(out.norm()),mean=float(out.mean())))
    a=net.encoder.register_forward_hook(eh);b=net.traj_decoder.register_forward_hook(ph)
    try:
        with input_guard(net,history,semantic,templates):p,t=decode(net,cfg,templates,history,semantic,goals)
    finally:a.remove();b.remove()
    return p,t,dict(encoder_features=features,path_logits=logits,semantic_SHA=digest(semantic))

@torch.no_grad()
def main():
    mode=sys.argv[1];assert mode in {'precheck','regression','full','fixed'}
    log=dict(passed=False,mode=mode,generated=0,training_steps=0,GT_loaded=False,test_loaded=False)
    budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_GPU_PROCESS_START']))
    try:
        data=json.loads((CONF/'generation_config.json').read_text());assert not data['GT_fields']
        assert json.loads((OUT/'cache_audit.json').read_text())['passed'] and sha(CP)==CK
        assert json.loads((OUT/'history_before.json').read_text())['passed']
        if mode!='precheck':
            f=json.loads((OUT/'protocol_frozen.json').read_text())
            for r in f['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
            assert sha(CONF/'experiment_protocol.json')==f['protocol_SHA'] and sha(CONF/'generation_config.json')==f['generation_config_SHA']
            assert json.loads((OUT/'precheck_results.json').read_text())['passed']
            assert commit()!=f['pre_freeze_commit']
        if mode in {'full','fixed'}:assert json.loads((OUT/'baseline_regression.json').read_text())['passed']
        if mode=='fixed':assert json.loads((OUT/'full_generation.json').read_text())['passed']
        configure_torch();t=time.perf_counter();budget.guard(60)
        net,cfg=build_model(17,'cuda:0');saved=torch.load(CP,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model'],strict=True);net.eval();net.semantic_segmentation.eval()
        initial=state_digest(net.state_dict());assert initial==data['model_state_SHA'];del saved
        assert sum(p.numel() for p in net.parameters())==53155339
        templates=make_templates(cfg,'cuda:0');pred=AuditPredictor(net,cfg,templates);torch.cuda.synchronize();load_seconds=time.perf_counter()-t
        hp=checked(data['train_history'] if mode=='precheck' else data['history'])
        assert set(hp)=={'sample_ids','scenes','history'};semcache={};sem_seconds=0.
        def semantic(condition,scene):
            nonlocal sem_seconds
            key=(condition,scene)
            if key not in semcache:
                t=time.perf_counter();s=checked(data['semantics'][condition][scene]);assert digest(s)==data['semantics'][condition][scene]['tensor_SHA']
                semcache[key]=s.cuda();torch.cuda.synchronize();sem_seconds+=time.perf_counter()-t
            return semcache[key]
        rows=[];probes=[]
        if mode=='precheck':
            assert len(hp['sample_ids'])==2 and set(hp['scenes'])=={'bookstore_0'}
            # No historical validation cache is recomputed; only an uncached TRAIN reference.
            scene='bookstore_0';sem=fixed_semantics(net,preprocess_scene(scene,cfg,'cuda:0'));torch.cuda.synchronize()
            source=json.loads((ROOT/'results/tables/stage11b2c/input_export.json').read_text())['scenes'][scene]
            h,w=source['resized_height'],source['resized_width'];stored={};baselines={}
            for i,sid in enumerate(hp['sample_ids']):
                history=hp['history'][i:i+1]
                for condition in ['S0','S1','S2','S3']:
                    budget.guard(30);v=intervene(sem,condition,h,w);t=time.perf_counter()
                    with input_guard(net,history,v,templates):p,g,tr,x=pred.predict(sid,history,semantic=v)
                    elapsed=time.perf_counter()-t;rows.append(dict(sample_id=sid,condition=condition,experiment='FULL',actual_seconds=elapsed,**tr))
                    stored[i,condition]=x
                    if condition=='S0':
                        reference=ROOT/f'experiments/stage11b2c/cache/precheck/rep0_target{i}.pt'
                        orig=torch.load(reference,map_location='cpu',weights_only=True)
                        assert set(orig)==set(x) and all(torch.equal(orig[k],x[k]) for k in x)
                        baselines[i]=g[:,0]
                    else:
                        t=time.perf_counter()
                        with input_guard(net,history,v,templates):path,timing=decode(net,cfg,templates,history,v,baselines[i])
                        rows.append(dict(sample_id=sid,condition=condition,experiment='FIXED',actual_seconds=time.perf_counter()-t,**timing))
            # All four interventions repeat in reversed target execution order.
            for i in [1,0]:
                sid=hp['sample_ids'][i];history=hp['history'][i:i+1]
                for condition in ['S0','S1','S2','S3']:
                    v=intervene(sem,condition,h,w);_,_,_,x=pred.predict(sid,history,semantic=v)
                    assert all(torch.equal(stored[i,condition][k],x[k]) for k in x)
                from scripts.stage11b1.evaluation import history_record
                raw=dict(sample_id=sid,xy=torch.cat([history[0],torch.zeros(12,2)]).numpy())
                sid0,h0=history_record(raw);raw['xy'][8:]=1e6;sid1,h1=history_record(raw)
                assert sid0==sid1 and torch.equal(h0,h1)
                _,_,_,x=pred.predict(sid1,h1,semantic=intervene(sem,'S1',h,w))
                assert all(torch.equal(stored[i,'S1'][k],x[k]) for k in x)
            maxfull=max(r['actual_seconds'] for r in rows if r['experiment']=='FULL')
            maxfixed=max(r['actual_seconds'] for r in rows if r['experiment']=='FIXED')
            # Historical measured upper floor avoids extrapolating only the small train image.
            estimate=budget.elapsed()+998*3*(max(maxfull,.6387296889024584)+max(maxfixed,.15))*1.75+300
            assert estimate<6600,('estimated cost not admitted',estimate)
            csvnew(OUT/'precheck_timing.csv',rows)
            log.update(passed=True,train_targets=2,train_only_segmentation_recomputed=True,original_train_all_tensors_bitwise=True,
                four_conditions_reverse_repeat_bitwise=True,raw_future_counterfactual_bitwise=True,history_heatmap_unchanged=True,
                full_max_seconds=maxfull,fixed_max_seconds=maxfixed,estimated_all_groups_seconds_high=estimate,
                estimate_factor=1.75,estimate_overhead_seconds=300,historical_full_floor=.6387296889024584,fixed_floor=.15,budget_fit=True,GT_scoring=False)
        elif mode=='regression':
            refs=data['original_tensor_records']
            for repeat in range(2):
                for i in (data['regression_indices'] if repeat==0 else list(reversed(data['regression_indices']))):
                    budget.guard(30);sid=hp['sample_ids'][i];v=semantic('S0',hp['scenes'][i]);history=hp['history'][i:i+1]
                    with input_guard(net,history,v,templates):p,g,tr,x=pred.predict(sid,history,semantic=v)
                    rec=checked(refs[i]);orig=rec['tensors'];bad=[k for k in orig if not torch.equal(orig[k],x[k])]
                    rows.append(dict(sample_id=sid,repeat=repeat,all_original_tensors_bitwise=not bad,mismatches=json.dumps(bad),**tr))
                    assert not bad,(sid,bad)
                    if repeat==0:
                        _,_,probe=probed_decode(net,cfg,templates,history,v,g[:,0]);probes.append(dict(sample_id=sid,condition='S0',**probe))
            csvnew(OUT/'baseline_regression_rows.csv',rows);jnew(OUT/'baseline_internal_probes.json',probes)
            log.update(passed=True,full_original_prediction_bitwise=True,logits_sigmoid_weights_proposals_centers_softargmax_goals_paths_bitwise=True,
                repeats=2,real_targets=3,history_heatmap_unchanged=True)
        elif mode=='full':
            savedgroups=[]
            for condition in ['S1','S2','S3']:
                paths=[];goals=[];manifest=[];(CACHE/condition/'tensors').mkdir(parents=True,exist_ok=False)
                for i,sid in enumerate(hp['sample_ids']):
                    budget.guard(30);scene=hp['scenes'][i];v=semantic(condition,scene);h=hp['history'][i:i+1]
                    with input_guard(net,h,v,templates):p,g,tr,x=pred.predict(sid,h,semantic=v)
                    file=CACHE/condition/'tensors'/cache_name(sid);t=time.perf_counter()
                    torch.save(dict(sample_id=sid,scene=scene,condition=condition,tensors=x,checkpoint_SHA=CK,
                        history_SHA=digest(h),semantic_SHA=digest(v)),file)
                    row=dict(sample_id=sid,scene=scene,condition=condition,path=str(file.relative_to(ROOT)),sha256=sha(file),
                        cache_seconds=time.perf_counter()-t,history_SHA=digest(h),semantic_SHA=digest(v),**tr)
                    manifest.append(row);paths.append(p[:,0]);goals.append(g[:,0]);log['generated']+=1
                    if (i+1)%100==0:
                        csvnew(OUT/f'partial_{condition}_{i+1:04}.csv',manifest)
                        print('FULL',condition,i+1,'/998','GPU_total',round(budget.elapsed(),2),flush=True)
                file=CACHE/f'{condition}_full.pt';assert not file.exists()
                torch.save(dict(sample_ids=hp['sample_ids'],paths=torch.stack(paths,dim=1),goals=torch.stack(goals,dim=1),
                    checkpoint_SHA=CK,history_file_SHA=data['history']['sha256'],condition=condition),file)
                csvnew(OUT/f'{condition}_tensor_manifest.csv',manifest)
                savedgroups.append(dict(condition=condition,path=str(file.relative_to(ROOT)),sha256=sha(file),targets=998,
                    goals_shape=[20,998,2],paths_shape=[20,998,12,2],manifest_SHA=sha(OUT/f'{condition}_tensor_manifest.csv')))
            log.update(passed=True,groups=savedgroups,history_heatmap_unchanged=True,all_actual_full_predictions=True)
        else:
            old=checked(data['original_predictions']);assert old['sample_ids']==hp['sample_ids'];savedgroups=[]
            for condition in ['S1','S2','S3']:
                paths=[];timings=[]
                for i,sid in enumerate(hp['sample_ids']):
                    budget.guard(30);scene=hp['scenes'][i];v=semantic(condition,scene);h=hp['history'][i:i+1]
                    if i in data['regression_indices']:
                        path,timing,probe=probed_decode(net,cfg,templates,h,v,old['goals'][:,i]);probes.append(dict(sample_id=sid,condition=condition,**probe))
                    else:
                        with input_guard(net,h,v,templates):path,timing=decode(net,cfg,templates,h,v,old['goals'][:,i])
                    paths.append(path[:,0]);timings.append(dict(sample_id=sid,scene=scene,condition=condition,goal_SHA=digest(old['goals'][:,i:i+1]),path_SHA=digest(path),**timing))
                    log['generated']+=1
                    if (i+1)%200==0:print('FIXED',condition,i+1,'/998','GPU_total',round(budget.elapsed(),2),flush=True)
                file=CACHE/f'{condition}_fixed.pt';assert not file.exists()
                torch.save(dict(sample_ids=hp['sample_ids'],paths=torch.stack(paths,dim=1),goals=old['goals'].clone(),
                    checkpoint_SHA=CK,history_file_SHA=data['history']['sha256'],condition=condition+'_FIXED_GOAL'),file)
                csvnew(OUT/f'{condition}_fixed_timing.csv',timings)
                savedgroups.append(dict(condition=condition+'_FIXED_GOAL',path=str(file.relative_to(ROOT)),sha256=sha(file),targets=998,
                    goals_shape=[20,998,2],paths_shape=[20,998,12,2],all_paths_really_decoded=True,all_goals_S0_bitwise=True))
            jnew(OUT/'fixed_internal_probes.json',probes);log.update(passed=True,groups=savedgroups,history_heatmap_unchanged=True)
        assert state_digest(net.state_dict())==initial
        log.update(parameters_unchanged=True,model_state_SHA=initial,model_parameters=sum(p.numel() for p in net.parameters()),
            segmentation_eval=not net.semantic_segmentation.training,model_load_seconds=load_seconds,semantic_read_H2D_seconds=sem_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),source_commit=commit())
    except Exception as e:
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_at_log']=budget.elapsed()
        name={'precheck':'precheck_results','regression':'baseline_regression','full':'full_generation','fixed':'fixed_generation'}[mode]
        jnew(OUT/(name+'.json'),log);print('WORKER_END',mode,log['passed'],flush=True)
if __name__=='__main__':main()
