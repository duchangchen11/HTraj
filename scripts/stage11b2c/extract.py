"""All998 original-order exports, then all998 reverse-order bitwise regression.

Inputs: physically GT-free histories, RGB, weights, original prediction-only
cache and hash-only traces. No future-coordinate or error table loader.
"""
from .worker_common import *
import pandas as pd,traceback,gc

HASHES=['probability_SHA','weights_SHA','proposal_SHA','goal_SHA','path_SHA']

def main():
    log=dict(status='RUNNING',passed=False,generated=0,reverse_regressed=0,training_steps=0,GT_read_for_scoring=False,test_loaded=False)
    rows=[];repeats=[];sid=None
    try:
        assert json.loads((OUT/'precheck_results.json').read_text())['passed']
        net,cfg,templates,predictor,budget,initial=initialize();pack=histories('validation_history_only');ids=pack['sample_ids'];assert len(ids)==len(set(ids))==998
        assert sha(PP)==PSHA;old=torch.load(PP,map_location='cpu',weights_only=True);assert ids==old['sample_ids']
        traces=pd.read_csv(OLD/'prediction_traces_epoch030.csv.gz').set_index('sample_id');source=json.loads((OUT/'input_export.json').read_text());cache={}
        budget.guard(json.loads((OUT/'precheck_results.json').read_text())['expected_two_pass_GPU_seconds_high']+120)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for i,sid in enumerate(ids):
            budget.guard(30);scene=pack['scenes'][i];sem=semantic(net,cfg,scene,cache,True)
            t=time.perf_counter();p,g,tr,x=predictor.predict(sid,pack['history'][i:i+1],semantic=sem);extractseconds=time.perf_counter()-t
            mismatched=[k for k in HASHES if tr[k]!=traces.loc[sid,k]]
            path=CACHE/'tensors'/cache_name(sid);assert not path.exists()
            payload=dict(sample_id=sid,scene=scene,checkpoint_SHA=CK,trace=tr,tensors=x,reference=source['scenes'][scene],resize=.25,unit='original reference pixels; *_network tensors in resized coordinates')
            t=time.perf_counter();torch.save(payload,path);cache_seconds=time.perf_counter()-t
            assert not mismatched,dict(sample_id=sid,mismatched=mismatched)
            assert torch.equal(p,old['paths'][:,i:i+1]) and torch.equal(g,old['goals'][:,i:i+1]),sid
            row=dict(sample_id=sid,scene=scene,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,
                checkpoint_SHA=CK,bitwise_frozen_goals_paths=True,all_five_original_SHA=True,extract_seconds=extractseconds,
                cache_seconds=cache_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**tr)
            rows.append(row);log['generated']=i+1
            if (i+1)%50==0:print('EXPORT',i+1,'/998','seconds',round(time.perf_counter()-started),'GPU_total',round(budget.elapsed()),flush=True)
            del p,g,tr,x,sem,payload;gc.collect() if (i+1)%100==0 else None
        cnew(OUT/'tensor_manifest.csv',rows)
        # Second full pass deliberately reverses every target, not only a subset.
        for j,i in enumerate(reversed(range(998))):
            budget.guard(30);sid=ids[i];scene=pack['scenes'][i];saved=torch.load(CACHE/'tensors'/cache_name(sid),map_location='cpu',weights_only=True)
            p,g,tr,x=predictor.predict(sid,pack['history'][i:i+1],semantic=semantic(net,cfg,scene,cache))
            same(x,saved['tensors']);assert all(tr[k]==saved['trace'][k] for k in HASHES+['indices_SHA','centers_SHA','logits_SHA','threshold_weights_SHA','history_tensor_SHA','semantic_SHA'])
            assert torch.equal(p,old['paths'][:,i:i+1]) and torch.equal(g,old['goals'][:,i:i+1])
            repeats.append(dict(sample_id=sid,scene=scene,reverse_rank=j,all_tensors_bitwise=True,all_original_SHA=True,indices_SHA=tr['indices_SHA'],centers_SHA=tr['centers_SHA'],goal_SHA=tr['goal_SHA'],path_SHA=tr['path_SHA']))
            log['reverse_regressed']=j+1
            if (j+1)%50==0:print('REVERSE_REGRESSION',j+1,'/998','GPU_total',round(budget.elapsed()),flush=True)
        cnew(OUT/'all998_reverse_regression.csv',repeats)
        # Independent semantic recomputation on fixed identities after full pass.
        for i in [997,499,0]:
            sid=ids[i];scene=pack['scenes'][i];p,g,tr,x=predictor.predict(sid,pack['history'][i:i+1],image=preprocess_scene(scene,cfg,'cuda:0'))
            saved=torch.load(CACHE/'tensors'/cache_name(sid),map_location='cpu',weights_only=True);same(x,saved['tensors'])
        assert state_digest(net.state_dict())==initial
        log.update(passed=True,status='GENERATION_AND_FULL_REGRESSION_COMPLETE',model_state_SHA=initial,parameters_unchanged=True,
            all998_original_five_SHA=True,all998_goals_paths_bitwise=True,all998_reverse_all_tensors_bitwise=True,semantic_recompute_bitwise=True,
            tensor_manifest_SHA=sha(OUT/'tensor_manifest.csv'),all_generation_and_regression_before_any_GT_scoring=True,
            semantic_caches={s:dict(path=str((CACHE/('semantic_'+s+'.pt')).relative_to(ROOT)),file_SHA=sha(CACHE/('semantic_'+s+'.pt')),tensor_SHA=digest(v)) for s,v in cache.items()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:
        traceback.print_exc();log.update(status='INCOMPLETE',error_type=type(e).__name__,error=str(e),sample_id=sid)
        if rows and not (OUT/'tensor_manifest.csv').exists():cnew(OUT/'partial_tensor_manifest.csv',rows)
        raise
    finally:
        log['GPU_cumulative_seconds']=parent_seconds()+time.perf_counter()-float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']);jnew(OUT/'generation_execution.json',log)
        print('GENERATION_FINISHED',log['status'],'generated',log['generated'],'reverse',log['reverse_regressed'],flush=True)
if __name__=='__main__':main()
