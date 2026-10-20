"""One immutable30-epoch seed17 run, no restart or hyperparameter search."""
from .common import *
from .data import load,manifest,scene_arrays
from .engine import train_batch,checkpoint,choose_best
from .validate import validate
from scripts.stage11b0.runtime import build_model,state_digest,save_checkpoint
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics
import torch,numpy as np,os,traceback,gc

def main():
    configure_torch();assert torch.cuda.is_available()
    config=json.loads((CONF/'train.json').read_text());frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'train.json')==frozen['train_config_SHA'] and sha(CONF/'evaluation.json')==frozen['evaluation_config_SHA']
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert json.loads((OUT/'preflight_results.json').read_text())['passed']
    assert json.loads((OUT/'budget_preestimate.json').read_text())['fit_30_epochs_and7_validations']
    started=float(os.environ.get('HTRAJ_PROCESS_MONOTONIC_START',time.perf_counter()))
    budget=Budget(parent_seconds(),started=started)
    log=dict(status='RUNNING',started_commit=commit(),completed_epochs=0,global_step=0,seed=17,optimizer_updates=0,
        training_started_from_random=True,old_predictor_checkpoint_loaded=False,test_loaded=False,failures=[])
    indices=[];results=[];epoch_rows=[];net=opt=None;ep=0;scene=None;offset=None
    stepfields=['epoch','global_step','scene','offset','targets','sample_ids','loss','goal_loss','path_loss','gradient_norm',
                'nonzero_gradient_elements','gradient_elements','learning_rate','seconds','peak_allocated_bytes','peak_reserved_bytes']
    sf,sw=open_csv(OUT/'training_steps.csv.gz',stepfields)
    ef,ew=open_csv(OUT/'epochs.csv',['epoch','global_step','train_loss','goal_loss','path_loss','learning_rate','train_seconds',
        'peak_allocated_bytes','peak_reserved_bytes','optimizer_steps','train_target_visits','unique_targets','recordings',
        'checkpoint_SHA','segmentation_state_SHA','scene_order','visit_order_SHA'])
    def update_manifest():
        best=choose_best(results)
        replace_json(OUT/'checkpoint_manifest.json',dict(checkpoints=indices,best=best['checkpoint'] if best else None,
              best_epoch=best['epoch'] if best else None,last=indices[-1] if indices else None,
              aliases='Immutable checkpoint paths; no overwrite of checkpoint tensors'))
    try:
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0')
        assert sum(p.numel() for p in net.parameters())==config['expected_parameters']
        assert sum(p.numel() for p in net.parameters() if p.requires_grad)==config['expected_predictor_parameters']
        core=state_digest({k:v for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')})
        assert core==json.loads((OUT/'preflight_results.json').read_text())['random_predictor_state_SHA']
        frozen_seg=state_digest(net.semantic_segmentation.state_dict())
        log.update(initial_predictor_state_SHA=core,initial_segmentation_state_SHA=frozen_seg,
                   total_parameters=53155339,predictor_parameters=1641381)
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001)
        assert not opt.state
        settings=dict(train=config,train_config_SHA=sha(CONF/'train.json'),evaluation_config_SHA=sha(CONF/'evaluation.json'),
            sampler_protocol_SHA=json.loads((CONF/'evaluation.json').read_text())['sampler_protocol_SHA'],
            data_manifest_SHA=frozen['data_manifest_SHA'],segmentation_checkpoint_SHA=config['segmentation_SHA'],
            freeze_commit=frozen['freeze_commit'])
        initial=checkpoint(CACHE/'checkpoints/initial.pt',net,opt,0,0,[],settings);initial.update(epoch=0,role='initial');indices.append(initial)
        update_manifest()
        df=load('train');scenes=sorted(df.sceneId.unique());identity={int(r['meta_id']):r for r in manifest('train')}
        expected={r['sample_id'] for r in identity.values()}
        reference_epoch=453.825087123;reference_val=998*.5178994235004211*1.1
        for ep in range(1,31):
            due=ep in config['validation_epochs']
            estimate_epoch=max(reference_epoch,1.25*epoch_rows[-1]['train_seconds']) if epoch_rows else reference_epoch*1.25
            estimate_val=max(reference_val,1.25*results[-1]['total_seconds']) if results else reference_val*1.25
            budget.guard(estimate_epoch+(estimate_val if due else 0)+30)
            epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(17+ep-1)).tolist()]
            sums=np.zeros(3);visited=[];startstep=log['global_step']
            for scene in order:
                budget.guard(30);coords,ids=scene_arrays(df,scene)
                image=preprocess_scene(scene,cfg,'cuda:0');semantic=fixed_semantics(net,image)
                for offset in range(0,len(coords),4):
                    budget.guard(30);batch=coords[offset:offset+4].cuda();t=time.perf_counter()
                    out=train_batch(net,opt,batch,semantic,cfg,templates);torch.cuda.synchronize()
                    log['global_step']+=1;log['optimizer_updates']+=1
                    sampleids=[identity[mid]['sample_id'] for mid in ids[offset:offset+4]];visited.extend(sampleids)
                    n=len(batch);sums+=np.array([out['loss'],out['goal_loss'],out['path_loss']])*n
                    row=dict(epoch=ep,global_step=log['global_step'],scene=scene,offset=offset,targets=n,sample_ids='|'.join(sampleids),
                        **{k:out[k] for k in ['loss','goal_loss','path_loss','gradient_norm','nonzero_gradient_elements','gradient_elements']},
                        learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    sw.writerow(row);sf.flush()
                    if log['global_step']%100==0:
                        print('TRAIN epoch',ep,'step',log['global_step'],'scene',scene,'loss',round(out['loss'],6),
                              'GPU_process_elapsed',round(budget.elapsed()),flush=True)
                del coords,ids,image,semantic,batch;gc.collect()
            assert len(visited)==len(set(visited))==7496 and set(visited)==expected
            assert log['global_step']-startstep==1882
            assert state_digest(net.semantic_segmentation.state_dict())==frozen_seg
            assert all(not m.training for m in net.semantic_segmentation.modules() if isinstance(m,torch.nn.BatchNorm2d))
            ck=checkpoint(CACHE/f'checkpoints/epoch_{ep:03d}.pt',net,opt,ep,log['global_step'],order,settings)
            ck.update(epoch=ep,role='epoch_boundary');indices.append(ck)
            means=sums/7496
            er=dict(epoch=ep,global_step=log['global_step'],train_loss=float(means[0]),goal_loss=float(means[1]),path_loss=float(means[2]),
                learning_rate=.0001,train_seconds=time.perf_counter()-epoch_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),optimizer_steps=1882,train_target_visits=7496,unique_targets=7496,recordings=25,
                checkpoint_SHA=ck['sha256'],segmentation_state_SHA=frozen_seg,scene_order='|'.join(order),
                visit_order_SHA=hashlib.sha256('\n'.join(visited).encode()).hexdigest())
            ew.writerow(er);ef.flush();epoch_rows.append(er);log['completed_epochs']=ep
            update_manifest();replace_json(OUT/'training_progress.json',dict(**log,GPU_process_seconds=budget.elapsed()))
            print('EPOCH_COMPLETE',ep,'loss',round(er['train_loss'],6),'seconds',round(er['train_seconds'],2),'checkpoint',ck['sha256'],flush=True)
            if due:
                budget.guard(estimate_val+30)
                result=validate(net,cfg,templates,ep,ck,budget);results.append(result)
                assert state_digest(net.semantic_segmentation.state_dict())==frozen_seg
                update_manifest()
                print('VALIDATION_COMPLETE',ep,'ADE1',round(result['window']['ADE1'],6),'FDE1',round(result['window']['FDE1'],6),
                    'minADE20',round(result['window']['minADE20'],6),'minFDE20',round(result['window']['minFDE20'],6),
                    'seconds',round(result['total_seconds'],2),flush=True)
                replace_json(OUT/'training_progress.json',dict(**log,latest_validation=result['window'],GPU_process_seconds=budget.elapsed()))
        log['status']='COMPLETED_30_EPOCHS'
    except Exception as e:
        traceback.print_exc();log.update(status='STOPPED_BUDGET' if isinstance(e,TimeoutError) else 'STOPPED_FAILURE',error_type=type(e).__name__,error=str(e))
        log['failures'].append(dict(epoch=ep,scene=scene,offset=offset,global_step=log['global_step'],error=str(e)))
        if net is not None and opt is not None:
            try:
                ck=save_checkpoint(CACHE/'checkpoints/stopped_state.pt',net,opt,
                    dict(completed_epochs=log['completed_epochs'],current_epoch=ep,epoch_boundary=False,global_step=log['global_step'],scene=scene,offset=offset),settings)
                log['stopped_state_checkpoint']=ck
            except Exception as e2:log['failure_checkpoint_error']=str(e2)
    finally:
        sf.close();ef.close();best=choose_best(results)
        log.update(best_epoch=best['epoch'] if best else None,best_metrics=best['window'] if best else None,
            last_complete_validation_epoch=results[-1]['epoch'] if results else None,
            GPU_process_seconds=budget.elapsed(),completed_validation_epochs=[r['epoch'] for r in results],
            checkpoints=indices,initialization_checkpoint_loaded=False,validation_GT_used_in_prediction=False,
            final_segmentation_state_SHA=state_digest(net.semantic_segmentation.state_dict()) if net is not None else None)
        jnew(OUT/'training_execution.json',log)
        print('RUN_FINISHED',log['status'],'epochs',log['completed_epochs'],'best',log['best_epoch'],'GPU',round(budget.elapsed(),2),flush=True)
    if log['status']=='STOPPED_FAILURE':raise SystemExit(1)

if __name__=='__main__':main()
