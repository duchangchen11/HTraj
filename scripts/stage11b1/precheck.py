"""Train-only, no-error, full TTST K20 timing and real-data isolation gates."""
from .common import *
from .data import load,manifest
from .evaluation import Predictor,history_record
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics
import numpy as np,torch,resource,traceback

def main():
    configure_torch(); assert torch.cuda.is_available()
    assert json.loads((OUT/'data_audit.json').read_text())['passed']
    start=time.perf_counter();cpu0=time.process_time()
    log=dict(status='RUNNING',code_commit=commit(),no_validation_forward=True,no_error_metrics=True,
             no_optimizer_updates=True,random_predictor=True,old_predictor_weights_loaded=False,cases=[])
    try:
        df=load('train');rows=manifest('train')
        # One sample per first20 lexically sorted train recordings, stable ID within recording.
        selected=[sorted([r for r in rows if r['scene']==s],key=lambda r:r['sample_id'])[0]
                  for s in sorted(df.sceneId.unique())[:20]]
        jnew(OUT/'precheck_selected_targets.json',dict(selection_rule='First lexical sample_id per first20 lexical train recordings; GT errors never consulted',targets=selected))
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0')
        log['initial_predictor_state_SHA']=state_digest({k:v for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')})
        log['total_parameters']=sum(p.numel() for p in net.parameters())
        log['trainable_parameters']=sum(p.numel() for p in net.parameters() if p.requires_grad)
        frozen=state_digest(net.semantic_segmentation.state_dict())
        predictor=Predictor(net,cfg,templates);torch.cuda.reset_peak_memory_stats()
        cache={};records={};saved={};rows_time=[]
        for i,r in enumerate(selected):
            scene=r['scene'];g=df[df.metaId==int(r['meta_id'])]
            raw=dict(sample_id=r['sample_id'],xy=g[['x','y']].to_numpy(dtype='float32').reshape(20,2))
            sid,h=history_record(raw);records[sid]=raw
            t=time.perf_counter();image=preprocess_scene(scene,cfg,'cuda:0');semantic=fixed_semantics(net,image);torch.cuda.synchronize()
            cache_seconds=time.perf_counter()-t
            # CPU semantic cache avoids retaining all scenes in10GB VRAM.
            cache[scene]=semantic.cpu()
            p,goals,timing=predictor.predict(sid,h,semantic=semantic)
            saved[sid]=(p,goals)
            row=dict(sample_id=sid,scene=scene,**timing,cache_load_segment_seconds=cache_seconds,
                     padded_height=semantic.shape[-2],padded_width=semantic.shape[-1])
            rows_time.append(row)
            print('Train-only K20',i+1,scene,round(timing['total_seconds'],3),'TTST',round(timing['ttst_kmeans_seconds'],3),flush=True)
            if i==0:
                # Same target/seed with recomputed vs cached semantics, and arbitrary future labels.
                q,qg,rt=predictor.predict(sid,h,image=image)
                assert torch.equal(p,q) and torch.equal(goals,qg)
                changed=dict(sample_id=sid,xy=raw['xy'].copy());changed['xy'][8:]=1e6
                cs,ch=history_record(changed)
                z,zg,ct=predictor.predict(cs,ch,semantic=semantic)
                assert torch.equal(p,z) and torch.equal(goals,zg)
                log.update(real_shape_pass=True,pixel_restore_divisor=cfg['resize'],cache_equivalence_bitwise=True,
                           same_checkpoint_repeat_bitwise=True,real_future_GT_counterfactual_bitwise=True,
                           first_single_target_seconds=timing['total_seconds'],extra_gate_seconds=rt['total_seconds']+ct['total_seconds'])
            del image,semantic
            if time.perf_counter()-start>3000:raise TimeoutError('Precheck conservative50min stop; no training')
        # Reverse two distinct IDs to test identity RNG independently of call order.
        for r in selected[1::-1]:
            sid,h=history_record(records[r['sample_id']])
            p,g,tm=predictor.predict(sid,h,semantic=cache[r['scene']].cuda())
            assert torch.equal(p,saved[sid][0]) and torch.equal(g,saved[sid][1])
        assert frozen==state_digest(net.semantic_segmentation.state_dict())
        log.update(status='TRAIN_ONLY_K20_PREFLIGHT_PASS',order_invariance_bitwise=True,
                   model_unchanged=True,first20_full_paths_shape=[20,20,12,2],
                   twenty_target_total_seconds=sum(r['total_seconds']+r['cache_load_segment_seconds'] for r in rows_time),
                   mean_target_seconds=float(np.mean([r['total_seconds'] for r in rows_time])),
                   p90_target_seconds=float(np.quantile([r['total_seconds'] for r in rows_time],.9)),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   GPU_card_state=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip())
        cnew(OUT/'precheck_target_timing.csv',rows_time)
    except Exception as e:
        traceback.print_exc();log.update(status='K20_PREFLIGHT_FAILED',error_type=type(e).__name__,error=str(e))
    finally:
        log.update(wall_seconds=time.perf_counter()-start,cpu_process_seconds=time.process_time()-cpu0,
                   cpu_occupancy_process_cores=(time.process_time()-cpu0)/(time.perf_counter()-start),
                   host_peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        jnew(OUT/'precheck_execution.json',log)
        print(log['status'],'seconds',round(log['wall_seconds'],3),flush=True)

if __name__=='__main__':main()
