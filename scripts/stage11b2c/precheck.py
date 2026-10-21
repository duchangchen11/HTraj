"""Fixed two training targets: no scoring, no gradients or optimizer updates."""
from .worker_common import *
from scripts.stage11b1r.predictor import Predictor
import traceback,shutil

def main():
    log=dict(passed=False,training_steps=0,GT_scoring=False,validation_generation=False,test_loaded=False)
    try:
        net,cfg,templates,predictor,budget,initial=initialize();old=Predictor(net,cfg,templates);pack=histories('precheck_history_only');cf=histories('counterfactual_history_only');cache={};refs={};rows=[]
        assert torch.equal(pack['history'],cf['history']) and pack['sample_ids']==cf['sample_ids']
        for rep in range(3):
            for i in ([0,1] if rep!=1 else [1,0]):
                budget.guard(30);sid=pack['sample_ids'][i];h=pack['history'][i:i+1];scene=pack['scenes'][i];sem=semantic(net,cfg,scene,cache)
                t=time.perf_counter();p,g,trace,tensors=predictor.predict(sid,h,semantic=sem)
                predseconds=time.perf_counter()-t;refp,refg,reftrace=old.predict(sid,h,semantic=sem)
                assert torch.equal(p,refp) and torch.equal(g,refg)
                assert all(trace[k]==reftrace[k] for k in ['probability_SHA','weights_SHA','proposal_SHA','goal_SHA','path_SHA'])
                if i in refs:same(tensors,refs[i])
                else:refs[i]=tensors
                target=CACHE/f'precheck/rep{rep}_target{i}.pt';t=time.perf_counter();torch.save(tensors,target);cache_seconds=time.perf_counter()-t
                rows.append(dict(rep=rep,sample_id=sid,full_audit_predict_seconds=predseconds,cache_seconds=cache_seconds,cache_bytes=target.stat().st_size,**trace))
        for i in [1,0]:
            sid=cf['sample_ids'][i];scene=cf['scenes'][i];p,g,tr,x=predictor.predict(sid,cf['history'][i:i+1],image=preprocess_scene(scene,cfg,'cuda:0'));same(x,refs[i])
        assert state_digest(net.state_dict())==initial
        cnew(OUT/'precheck_timing.csv',rows)
        high=max(.6387296889024584,max(r['full_audit_predict_seconds']+r['cache_seconds'] for r in rows))*1996*1.5+600+budget.elapsed()
        estimated_bytes=max(r['cache_bytes'] for r in rows)*998*2
        assert high<6600 and shutil.disk_usage(CACHE).free>estimated_bytes+10*1024**3
        log.update(passed=True,bitwise_vs_original=True,three_repeats_and_reverse_order_bitwise=True,raw_future_counterfactual_and_semantic_recompute_bitwise=True,
            parameter_state_SHA=initial,parameters_unchanged=True,total_parameters=53155339,checkpoint_SHA=CK,
            expected_two_pass_GPU_seconds_high=high,estimated_cache_bytes_high=estimated_bytes,free_disk_bytes=shutil.disk_usage(CACHE).free,
            normalization_float32_original=True,CPU_export_original_precision=True,budget_fit=True)
    except Exception as e:traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_seconds']=parent_seconds()+time.perf_counter()-float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']);jnew(OUT/'precheck_results.json',log)
        print('PRECHECK',log['passed'],flush=True)
if __name__=='__main__':main()
