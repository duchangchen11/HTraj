"""One from-scratch seed17 run; frozen original math, order, and K20 evaluator."""
from .common import *
from .preflight import core_sha,first_weight
from .validate import validate
from scripts.stage11b1c.data import load,manifest,scene_arrays
from scripts.stage11b1c.engine import train_batch,choose_best
from scripts.stage11b0.runtime import build_model,state_digest,save_checkpoint
from scripts.stage11b0.training_adapter import make_templates
import torch,numpy as np,traceback,gc

def main():
    cfgtrain=json.loads((CONF/'train_control.json').read_text());frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert frozen['numerical_execution_authorized'], 'Budget admission rejected; no formal run authorized'
    configure_torch();assert torch.cuda.is_available()
    assert sha(CONF/'train_control.json')==frozen['train_config_SHA'] and sha(CONF/'evaluation.json')==frozen['evaluation_config_SHA']
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert json.loads((OUT/'preflight_results.json').read_text())['passed']
    assert json.loads((OUT/'budget_preestimate.json').read_text())['fit_30_epochs_and7_validations']
    budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    log=dict(status='RUNNING',started_commit=commit(),completed_epochs=0,global_step=0,seed=17,optimizer_updates=0,
        training_started_from_random=True,old_predictor_checkpoint_loaded=False,test_loaded=False,failures=[])
    indices=[];results=[];epoch_rows=[];net=opt=None;ep=0;scene=None;offset=None;settings={}
    stepfields=['epoch','global_step','scene','offset','targets','sample_ids','loss','goal_loss','path_loss','gradient_norm',
        'nonzero_gradient_elements','gradient_elements','learning_rate','seconds','peak_allocated_bytes','peak_reserved_bytes',
        'semantic_zero_nonzero_count','semantic_weight_nonzero_gradient','history_weight_nonzero_gradient']
    sf,sw=open_csv(OUT/'training_steps.csv.gz',stepfields)
    efields=['epoch','global_step','train_loss','goal_loss','path_loss','learning_rate','train_seconds','GPU_seconds',
        'peak_allocated_bytes','peak_reserved_bytes','optimizer_steps','train_target_visits','unique_targets','recordings',
        'checkpoint_SHA','segmentation_state_SHA','scene_order','scene_order_SHA','visit_order_SHA','semantic_zero_nonzero_count']
    ef,ew=open_csv(OUT/'epochs.csv',efields)
    vf,vw=open_csv(OUT/'validation_metrics.csv',['epoch','status','checkpoint_SHA','targets','seconds',
        'ADE1','FDE1','minADE20','minFDE20','GoalCover10','GoalCover25','GoalCover50','macro_minADE20','macro_minFDE20'])
    def update_manifest():
        best=choose_best(results)
        replace_json(OUT/'checkpoint_manifest.json',dict(checkpoints=indices,best=best['checkpoint'] if best else None,
            best_epoch=best['epoch'] if best else None,last=indices[-1] if indices else None,immutable=True))
    try:
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0')
        assert sum(p.numel() for p in net.parameters())==53155339
        assert sum(p.numel() for p in net.parameters() if p.requires_grad)==1641381
        core=core_sha(net);assert core==INITIAL
        frozen_seg=state_digest(net.semantic_segmentation.state_dict())
        def forbidden(*a,**k):raise AssertionError('Segmentation forward forbidden')
        net.semantic_segmentation.register_forward_pre_hook(forbidden)
        log.update(initial_predictor_state_SHA=core,initial_segmentation_state_SHA=frozen_seg,
            total_parameters=53155339,predictor_parameters=1641381,semantic_zero_nonzero_count=0,
            semantic_forward_calls=0,initial_Adam_state_empty=True)
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001);assert not opt.state
        settings=dict(train=cfgtrain,train_config_SHA=sha(CONF/'train_control.json'),evaluation_config_SHA=sha(CONF/'evaluation.json'),
            data_manifest_SHA=frozen['data_manifest_SHA'],initial_predictor_SHA=INITIAL,
            zero_input_protocol_SHA=sha(CONF/'scene_shapes.json'),semantic_baseline=frozen['semantic_baseline'],freeze_commit=commit())
        def checkpoint(epoch,step,order,visit_SHA):
            info=save_checkpoint(CACHE/('checkpoints/initial.pt' if epoch==0 else f'checkpoints/epoch_{epoch:03d}.pt'),net,opt,
                dict(epoch=epoch,global_step=step,epoch_boundary=True,completed_scene_order=order,next_epoch=epoch+1,
                    next_scene_seed=17+epoch,visit_order_SHA=visit_SHA,scene_order_SHA=hashlib.sha256('\n'.join(order).encode()).hexdigest()),settings)
            info.update(epoch=epoch,role='initial' if epoch==0 else 'epoch_boundary',model_state_SHA=state_digest(net.state_dict()))
            return info
        indices.append(checkpoint(0,0,[],None));update_manifest()
        df=load('train');scenes=sorted(df.sceneId.unique());table=shapes()
        identity={int(r['meta_id']):r for r in manifest('train')};expected={r['sample_id'] for r in identity.values()}
        orders={int(r['epoch']):r for r in csv.DictReader((OUT/'expected_orders.csv').open())}
        pre=json.loads((OUT/'budget_preestimate.json').read_text())
        for ep in range(1,31):
            due=ep in cfgtrain['validation_epochs']
            estimate_epoch=max(pre['guard_epoch_seconds'],max([r['train_seconds'] for r in epoch_rows],default=0)*1.03)
            estimate_val=max(pre['guard_validation_seconds'],max([r['total_seconds'] for r in results],default=0)*1.03)
            budget.guard(estimate_epoch+(estimate_val if due else 0))
            epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(17+ep-1)).tolist()]
            order_SHA=hashlib.sha256('\n'.join(order).encode()).hexdigest();assert order_SHA==orders[ep]['scene_order_SHA']
            sums=np.zeros(3);visited=[];startstep=log['global_step']
            for scene in order:
                budget.guard(30);coords,ids=scene_arrays(df,scene);semantic=zero_semantic(scene,'cuda:0',table)
                for offset in range(0,len(coords),4):
                    budget.guard(5);batch=coords[offset:offset+4].cuda();t=time.perf_counter()
                    out=train_batch(net,opt,batch,semantic,cfg,templates);torch.cuda.synchronize()
                    _,weight=first_weight(net);sg=int(torch.count_nonzero(weight.grad[:,:6]));assert sg==0
                    hg=int(torch.count_nonzero(weight.grad[:,6:]));assert not torch.count_nonzero(semantic)
                    log['global_step']+=1;log['optimizer_updates']+=1
                    sampleids=[identity[mid]['sample_id'] for mid in ids[offset:offset+4]];visited.extend(sampleids)
                    sums+=np.array([out['loss'],out['goal_loss'],out['path_loss']])*len(batch)
                    sw.writerow(dict(epoch=ep,global_step=log['global_step'],scene=scene,offset=offset,targets=len(batch),
                        sample_ids='|'.join(sampleids),**out,learning_rate=.0001,seconds=time.perf_counter()-t,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        semantic_zero_nonzero_count=0,semantic_weight_nonzero_gradient=sg,history_weight_nonzero_gradient=hg));sf.flush()
                    if log['global_step']%100==0:print('TRAIN',ep,log['global_step'],scene,'loss',out['loss'],'GPU_elapsed',budget.elapsed(),flush=True)
                del coords,ids,semantic,batch;gc.collect()
            visit_SHA=hashlib.sha256('\n'.join(visited).encode()).hexdigest()
            assert len(visited)==len(set(visited))==7496 and set(visited)==expected
            assert log['global_step']-startstep==1882 and visit_SHA==orders[ep]['visit_order_SHA']
            assert state_digest(net.semantic_segmentation.state_dict())==frozen_seg
            ck=checkpoint(ep,log['global_step'],order,visit_SHA);indices.append(ck);means=sums/7496
            er=dict(epoch=ep,global_step=log['global_step'],train_loss=float(means[0]),goal_loss=float(means[1]),path_loss=float(means[2]),
                learning_rate=.0001,train_seconds=time.perf_counter()-epoch_start,GPU_seconds=budget.elapsed(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                optimizer_steps=1882,train_target_visits=7496,unique_targets=7496,recordings=25,checkpoint_SHA=ck['sha256'],
                segmentation_state_SHA=frozen_seg,scene_order='|'.join(order),scene_order_SHA=order_SHA,visit_order_SHA=visit_SHA,semantic_zero_nonzero_count=0)
            ew.writerow(er);ef.flush();epoch_rows.append(er);log['completed_epochs']=ep;update_manifest()
            replace_json(OUT/'training_progress.json',dict(**log,GPU_process_seconds=budget.elapsed()))
            print('EPOCH_COMPLETE',ep,'loss',er['train_loss'],'seconds',er['train_seconds'],flush=True)
            if due:
                budget.guard(estimate_val);result=validate(net,cfg,templates,ep,ck,budget);results.append(result)
                vw.writerow(dict(epoch=ep,status=result['status'],checkpoint_SHA=ck['sha256'],targets=998,seconds=result['total_seconds'],
                    **{k:result['window'][k] for k in ['ADE1','FDE1','minADE20','minFDE20']},
                    **{f'GoalCover{v}':result['goal_coverage'][f'within_{v}px'] for v in [10,25,50]},
                    macro_minADE20=result['recording_macro']['minADE20'],macro_minFDE20=result['recording_macro']['minFDE20']));vf.flush()
                assert state_digest(net.semantic_segmentation.state_dict())==frozen_seg;update_manifest()
                print('VALIDATION_COMPLETE',ep,result['window'],'seconds',result['total_seconds'],flush=True)
                replace_json(OUT/'training_progress.json',dict(**log,latest_validation=result['window'],GPU_process_seconds=budget.elapsed()))
        log['status']='COMPLETED_30_EPOCHS'
    except Exception as e:
        traceback.print_exc();log.update(status='STOPPED_BUDGET' if isinstance(e,TimeoutError) else 'STOPPED_FAILURE',error_type=type(e).__name__,error=str(e))
        log['failures'].append(dict(epoch=ep,scene=scene,offset=offset,global_step=log['global_step'],error=str(e)))
        if net is not None and opt is not None:
            try:log['stopped_checkpoint']=save_checkpoint(CACHE/'checkpoints/stopped_state.pt',net,opt,
                dict(completed_epochs=log['completed_epochs'],current_epoch=ep,global_step=log['global_step'],scene=scene,offset=offset,epoch_boundary=False),settings)
            except Exception as e2:log['failure_checkpoint_error']=str(e2)
    finally:
        sf.close();ef.close();vf.close();best=choose_best(results)
        log.update(best_epoch=best['epoch'] if best else None,best_metrics=best['window'] if best else None,
            completed_validation_epochs=[r['epoch'] for r in results],GPU_process_seconds=budget.elapsed(),checkpoints=indices,
            formal_initialization_checkpoint_loaded=False,validation_GT_used_in_prediction=False,
            final_segmentation_state_SHA=state_digest(net.semantic_segmentation.state_dict()) if net is not None else None)
        jnew(OUT/'training_execution.json',log);print('RUN_FINISHED',log['status'],log['completed_epochs'],budget.elapsed(),flush=True)
    if log['status']=='STOPPED_FAILURE':raise SystemExit(1)
if __name__=='__main__':main()
