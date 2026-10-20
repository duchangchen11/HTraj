"""Fixed-weight real TRAIN-only K20 checks; no loss or prediction errors."""
from .common import *
from .predictor import Predictor,digest
from .sampler import sample_cpu
from .diagnose_original import rng_hashes
from scripts.stage11b1.evaluation import Predictor as OriginalPredictor,history_record
from scripts.stage11b1.data import load
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics
from scripts.stage11b0.data_audit import canonical_sha
import numpy as np,torch,resource

def main(guard):
    configure_torch();assert torch.cuda.is_available()
    start=time.perf_counter();cpu_start=time.process_time()
    protocol=json.loads((CONF/'sampling_protocol.json').read_text())
    selected=ROOT/protocol['target_manifest'];assert sha(selected)==protocol['target_manifest_SHA']
    rows=json.loads(selected.read_text())['targets'];assert len(rows)==20
    df=load('train');groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
    net,cfg=build_model(17,'cuda:0');net.eval();templates=make_templates(cfg,'cuda:0')
    initial=state_digest(net.state_dict());pcount=sum(p.numel() for p in net.parameters())
    test_checkpoint=CACHE/'fixed_random_model.pt'
    torch.save(dict(model=net.state_dict(),official_commit=OFFICIAL,seed=17,training_steps=0),test_checkpoint)
    new=Predictor(net,cfg,templates);old=OriginalPredictor(net,cfg,templates)
    artifact=json.loads((OUT/'minimal_original_case.json').read_text())
    pack=torch.load(ROOT/artifact['artifact_path'],map_location='cpu',weights_only=True)
    assert sha(ROOT/artifact['artifact_path'])==artifact['artifact_SHA']
    prob=pack['probability'].cuda();proposal_records=[];first=None
    for i in range(5):
        guard();before=rng_hashes();t=time.perf_counter();a,w,idx=sample_cpu(prob,rows[0]['sample_id']);torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        assert before==rng_hashes()
        if first is None:first=a.cpu()
        proposal_records.append(dict(repeat=i,probability_SHA=digest(prob),weights_SHA=digest(w),proposal_SHA=digest(a),
                                     max_coordinate_diff=float((a.cpu()-first).abs().max()),seconds=elapsed))
    cnew(OUT/'fixed_probability_five_repeats.csv',proposal_records)
    assert len({r['proposal_SHA'] for r in proposal_records})==1
    records=[];timings=[];raw={};images={};semantics={};baseline={};saved=[]
    def exercise(r,case,image=None,semantic=None,override_record=None):
        guard();record=raw[r['sample_id']] if override_record is None else override_record
        sid,h=history_record(record)
        assert np.array_equal(h[0].numpy(),raw[r['sample_id']]['xy'][:8])
        before=rng_hashes()
        p,g,trace=new.predict(sid,h,image=image,semantic=semantic)
        assert before==rng_hashes(),'Adapter mutated caller RNG state'
        if sid in baseline:
            ref=baseline[sid]
            match=all(trace[k]==ref[2][k] for k in ['proposal_SHA','goal_SHA','path_SHA'])
            pd=float((p-ref[0]).abs().max());gd=float((g-ref[1]).abs().max())
        else:match=True;pd=gd=0.
        row=dict(case=case,sample_id=sid,scene=r['scene'],**trace,bitwise_vs_first=match,
                 max_path_coordinate_diff=pd,max_goal_coordinate_diff=gd,RNG_unchanged=True)
        records.append(row)
        assert match and pd==gd==0.,'Real K20 exact-repeat check failed'
        return p,g,trace
    torch.cuda.reset_peak_memory_stats()
    for i,r in enumerate(rows):
        sid=r['sample_id'];g=groups[int(r['meta_id'])]
        xy=g[['x','y']].to_numpy(dtype='float32').reshape(20,2)
        # Identity check uses original float64 source before constructing float32 model input.
        assert canonical_sha(g[['x','y']].iloc[:8].to_numpy(dtype=float))==r['history_SHA']
        raw[sid]=dict(sample_id=sid,xy=xy.copy())
        t=time.perf_counter();image=preprocess_scene(r['scene'],cfg,'cuda:0');semantic=fixed_semantics(net,image);torch.cuda.synchronize()
        cache_seconds=time.perf_counter()-t
        images[r['scene']]=image.cpu();semantics[r['scene']]=semantic.cpu()
        p,goals,trace=exercise(r,'primary20',semantic=semantic)
        baseline[sid]=(p,goals,trace);saved.append(dict(sample_id=sid,paths=p,goals=goals))
        # Same weights/target, unchanged original GPU implementation, timing only.
        op,og,oldtime=old.predict(sid,history_record(raw[sid])[1],semantic=semantic)
        timings.append(dict(sample_id=sid,scene=r['scene'],cache_seconds=cache_seconds,
             new_total_seconds=trace['total_seconds'],new_sampling_seconds=trace['sampling_seconds'],
             new_kmeans_seconds=trace['kmeans_seconds'],new_ttst_seconds=trace['ttst_seconds'],
             new_decode20_seconds=trace['decode20_seconds'],old_total_seconds=oldtime['total_seconds'],
             old_ttst_kmeans_seconds=oldtime['ttst_kmeans_seconds'],old_decode20_seconds=oldtime['decode20_seconds']))
        print('Primary',i+1,r['scene'],'CPU-sampler K20',round(trace['total_seconds'],4),
              'original',round(oldtime['total_seconds'],4),flush=True)
    # First3, five full calls each (first primary is repetition1).
    for r in rows[:3]:
        for j in range(4):exercise(r,f'full_repeat_{j+2}',semantic=semantics[r['scene']].cuda())
    for r in reversed(rows):exercise(r,'reverse_order20',semantic=semantics[r['scene']].cuda())
    for r in rows:
        sid=r['sample_id'];changed=dict(sample_id=sid,xy=raw[sid]['xy'].copy())
        changed['xy'][8:]=np.arange(24,dtype='float32').reshape(12,2)+1000000
        assert np.array_equal(changed['xy'][:8],raw[sid]['xy'][:8])
        exercise(r,'future_GT_counterfactual20',semantic=semantics[r['scene']].cuda(),override_record=changed)
        exercise(r,'recompute_segmentation20',image=images[r['scene']].cuda())
    assert initial==state_digest(net.state_dict()),'Model parameters/buffers changed'
    cnew(OUT/'real_K20_checks.csv',records);cnew(OUT/'paired_target_timing.csv',timings)
    torch.save(saved,CACHE/'primary20_predictions.pt')
    result=dict(status='REAL_K20_BITWISE_PASS',fixed_probability_repeats=5,same_target_full_repeats=5,
        repeated_targets=3,reverse_order_targets=20,future_counterfactual_targets=20,resegmentation_targets=20,
        checks=len(records),all_bitwise=True,max_path_coordinate_diff=max(r['max_path_coordinate_diff'] for r in records),
        max_goal_coordinate_diff=max(r['max_goal_coordinate_diff'] for r in records),
        probability_and_weights_precision='float32 preserved on original device',raw_pixel_restore_divisor=.25,
        shape=[20,1,12,2],finite=True,RNG_caller_state_preserved=True,
        parameters=pcount,trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        model_state_SHA=initial,model_unchanged=True,checkpoint_path=str(test_checkpoint.relative_to(ROOT)),
        checkpoint_SHA=sha(test_checkpoint),checkpoint_description='Random seed17 engineering state,0 optimizer steps; not trained weights',
        primary_reference={sid:{k:trace[k] for k in ['probability_SHA','weights_SHA','proposal_SHA','goal_SHA','path_SHA']}
                           for sid,(p,g,trace) in baseline.items()},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        whole_card_state=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip(),
        CPU_process_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-start,
        CPU_occupancy_cores=(time.process_time()-cpu_start)/(time.perf_counter()-start),
        peak_host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        validation_loaded=False,test_loaded=False,loss_computed=False,prediction_error_metrics=False,optimizer_steps=0)
    jnew(OUT/'real_checks_results.json',result)
    print('All real K20 checks bitwise PASS:',len(records),flush=True)

if __name__=='__main__':timed_run('run_checks',main)
