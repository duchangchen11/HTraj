"""Exactly one authorized diagnostic epoch; never a formal Stage11C-1 run."""
from .common import *
from scripts.stage11b0.runtime import build_model,state_digest,save_checkpoint,load_checkpoint,rng_state
from scripts.stage11b0.training_adapter import make_templates
from scripts.stage11b1c.engine import train_batch
from scripts.stage11b1c.data import load,manifest,scene_arrays
from scripts.stage11c1.preflight import core_sha,first_weight
from scripts.stage11b0.run_admission import optimizer_digest
import torch,numpy as np,gc,traceback

def main():
    frozen=verify_frozen();configure_torch()
    budget=Budget(float(os.environ['HTRAJ_PRIOR_GPU_SECONDS']),float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    budget.guard(750)
    log=dict(status='RUNNING',role='DIAGNOSTIC_ONLY',formal_training=False,formal_Stage11C1_epochs=0,
        seed=17,epoch=0,global_step=0,optimizer_updates=0,test_loaded=False,validation_GT_loaded=False,
        trained_predictor_loaded=False,source_commit=commit())
    net=opt=None;visited=[];scene=None;offset=None
    fields=['global_step','scene','offset','targets','sample_ids','loss','goal_loss','path_loss','gradient_norm',
        'nonzero_gradient_elements','gradient_elements','seconds','semantic_nonzero_count','semantic_weight_nonzero_gradient',
        'peak_allocated_bytes','peak_reserved_bytes']
    sf,sw=open_csv(OUT/'diagnostic_training_steps.csv.gz',fields)
    try:
        net,cfg=build_model(17,'cuda:0');assert core_sha(net)==INITIAL
        templates=make_templates(cfg,'cuda:0');seg=state_digest(net.semantic_segmentation.state_dict())
        def forbidden(*a,**k):raise AssertionError('Diagnostic NoSemantic must never call segmentation')
        net.semantic_segmentation.register_forward_pre_hook(forbidden)
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001);assert not opt.state
        initial_rng=rng_digest()
        settings=dict(role='DIAGNOSTIC_ONLY',formal_training=False,config_SHA=sha(CONF/'runtime_protocol.json'),
            protocol_SHA=sha(OUT/'protocol_frozen.json'),source_files=frozen['source_files'],
            data_manifest_SHA=frozen['data_manifest_SHA'],initial_predictor_SHA=INITIAL,
            segmentation_frozen=True,semantic='Six all-zero channels including padding',
            reuse_requires_separate_approval=True)
        initial=save_checkpoint(CACHE/'diagnostic_initial.pt',net,opt,
            dict(epoch=0,global_step=0,epoch_boundary=True,role='DIAGNOSTIC_ONLY',rng_SHA=initial_rng),settings)
        jnew(OUT/'diagnostic_initialization.json',dict(passed=True,initial_predictor_SHA=INITIAL,new_Adam_state_empty=True,
            initial_RNG_SHA=initial_rng,initial_checkpoint=initial,total_parameters=53155339,predictor_parameters=1641381,
            segmentation_state_SHA=seg,checkpoint_loaded_for_initialization=False))
        df=load('train');scenes=sorted(df.sceneId.unique());ids_by_meta={int(r['meta_id']):r for r in manifest('train')}
        expected={r['sample_id'] for r in ids_by_meta.values()};table=shapes()
        order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(17)).tolist()]
        order_SHA=hashlib.sha256('\n'.join(order).encode()).hexdigest();old=next(csv.DictReader((ROOT/'results/tables/stage11c1/expected_orders.csv').open()))
        assert order_SHA==old['scene_order_SHA'];sums=np.zeros(3);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for scene in order:
            budget.guard(30);coords,ids=scene_arrays(df,scene);zero=zero_semantic(scene,'cuda:0',table)
            for offset in range(0,len(coords),4):
                budget.guard(5);batch=coords[offset:offset+4].cuda();t=time.perf_counter()
                result=train_batch(net,opt,batch,zero,cfg,templates);torch.cuda.synchronize()
                _,w=first_weight(net);sg=int(torch.count_nonzero(w.grad[:,:6]));assert sg==0 and not torch.count_nonzero(zero)
                sids=[ids_by_meta[mid]['sample_id'] for mid in ids[offset:offset+4]];visited.extend(sids)
                log['global_step']+=1;log['optimizer_updates']+=1;sums+=np.array([result[k] for k in ['loss','goal_loss','path_loss']])*len(batch)
                sw.writerow(dict(global_step=log['global_step'],scene=scene,offset=offset,targets=len(batch),sample_ids='|'.join(sids),
                    **result,seconds=time.perf_counter()-t,semantic_nonzero_count=0,semantic_weight_nonzero_gradient=sg,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()));sf.flush()
                if log['global_step']%100==0:print('DIAGNOSTIC_STEP',log['global_step'],scene,'loss',result['loss'],'elapsed',budget.elapsed(),flush=True)
            del coords,ids,zero,batch;gc.collect()
        train_seconds=time.perf_counter()-start
        assert len(visited)==len(set(visited))==7496 and set(visited)==expected
        visit_SHA=hashlib.sha256('\n'.join(visited).encode()).hexdigest();assert visit_SHA==old['visit_order_SHA']
        assert log['global_step']==1882 and state_digest(net.semantic_segmentation.state_dict())==seg
        progress=dict(epoch=1,global_step=1882,optimizer_updates=1882,epoch_boundary=True,role='DIAGNOSTIC_ONLY',
            scene_order_SHA=order_SHA,visit_order_SHA=visit_SHA,completed_scene_order=order)
        # Complete model/Adam/Python/NumPy/Torch CPU+CUDA RNG saved before any sampling performance measurement.
        ck=save_checkpoint(CACHE/'diagnostic_epoch001.pt',net,opt,progress,settings)
        want=core_sha(net);wantopt=optimizer_digest(opt);wantrng=rng_digest()
        loaded=load_checkpoint(CACHE/'diagnostic_epoch001.pt',net,opt)
        assert core_sha(net)==want and optimizer_digest(opt)==wantopt and rng_digest()==wantrng
        means=sums/7496
        log.update(status='COMPLETED_DIAGNOSTIC_EPOCH1',epoch=1,train_seconds=train_seconds,
            target_visits=7496,unique_targets=7496,optimizer_updates=1882,scene_order=order,
            scene_order_SHA=order_SHA,visit_order_SHA=visit_SHA,
            training_loss=float(means[0]),goal_loss=float(means[1]),path_loss=float(means[2]),
            initial_predictor_SHA=INITIAL,final_predictor_SHA=want,segmentation_state_SHA=seg,segmentation_unchanged=True,
            semantic_zero_nonzero_count=0,semantic_forward_calls=0,
            strict_checkpoint_restore_model_optimizer_RNG=True,sampling_before_checkpoint=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint=ck)
        jnew(OUT/'diagnostic_checkpoint_manifest.json',dict(status='COMPLETE',role='DIAGNOSTIC_ONLY',initial=initial,epoch1=ck,
            formal_Stage11C1_checkpoint=False,reuse_requires_separate_approval=True))
    except Exception as e:
        traceback.print_exc();log.update(status='STOPPED_BUDGET' if isinstance(e,TimeoutError) else 'FAILED',error_type=type(e).__name__,error=str(e),scene=scene,offset=offset)
        if net is not None and opt is not None:
            log['stopped_checkpoint']=save_checkpoint(CACHE/'diagnostic_stopped.pt',net,opt,
                dict(epoch_boundary=False,completed_epochs=0,global_step=log['global_step'],scene=scene,offset=offset),settings)
        raise
    finally:
        sf.close();log['GPU_cumulative_worker_seconds']=budget.elapsed();jnew(OUT/'diagnostic_epoch1_execution.json',log)
        print('DIAGNOSTIC_FINISH',log,flush=True)
if __name__=='__main__':main()
