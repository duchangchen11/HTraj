"""Bounded real-data training admission: 3 smoke steps + one unaugmented pass.

No validation/test forward, scores, checkpoint selection, or multi-epoch loop.
"""
from .common import *
from .runtime import (build_model,state_digest,save_checkpoint,load_checkpoint,seed_all)
from .training_adapter import (training_loss,fixed_semantics,preprocess_scene,make_templates,gradients,ego_inference_no_GT)
from .data_audit import load_train
import torch,numpy as np,traceback,collections,gc

def core_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')}
def optimizer_digest(opt):
    h=hashlib.sha256()
    for idx,st in sorted(opt.state_dict()['state'].items()):
        h.update(str(idx).encode())
        for k,v in sorted(st.items()):
            h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()

def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    settings=json.loads((CONF/'training_admission.json').read_text())
    assert settings['real_gpu_training_authorized'] and settings['target_batch_size']==4
    gate=json.loads((OUT/'cpu_test_gate.json').read_text());assert gate['passed']
    split=json.loads((OUT/'split_audit.json').read_text());assert split['status']=='SPLIT_IDENTITY_PASS'
    start=time.perf_counter();prior=gate['wall_seconds_conservative_GPU_budget_charge']
    safe_limit=3300-prior
    def guard(estimated_next=5):
        if time.perf_counter()-start+estimated_next>=safe_limit:raise TimeoutError('3300s conservative cumulative safety stop')
    log=dict(status='RUNNING',started_code_commit=commit(),official_commit=OFFICIAL,settings_SHA=sha(CONF/'training_admission.json'),
        source_train_manifest_SHA=sha(OUT/'train_target_manifest.csv'),scene_batch_size=1,target_batch_size=4,seed=17,
        no_validation_forward=True,no_test_pickle_load=True,no_prediction_metrics=True,predictor_pretrained_weights_loaded=False,
        augmentation_executed=False,original_scene_cap_removed_in_independent_runner=True,complete_pass=False,checkpoints=[],failures=[])
    fields=['phase','step','scene','target_start','targets','sample_ids','loss','goal_loss','path_loss','gradient_norm','nonzero_gradient_elements','gradient_elements','lr','wall_seconds','GPU_event_seconds','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes']
    timing_file=(OUT/'training_timing.csv').open('x',newline='');writer=csv.DictWriter(timing_file,fieldnames=fields,lineterminator='\n');writer.writeheader()
    memory=[];scene_records=[];allsteps=[];visited=[];totalsteps=0
    net=opt=None
    try:
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0')
        for n,p in net.named_parameters():
            if n.startswith('semantic_segmentation.'):assert not p.requires_grad
        initial=core_state(net);log['initial_predictor_state_SHA']=state_digest(initial)
        frozen_before=state_digest(net.semantic_segmentation.state_dict());log['initial_segmentation_state_SHA']=frozen_before
        log.update(total_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-4)
        df=load_train();scenes=sorted(df.sceneId.unique());assert len(scenes)==25
        manifest={int(r['meta_id']):r for r in csv.DictReader((OUT/'train_target_manifest.csv').open())}
        # Seeded scene order, within-scene original author row order. Scene means recording, not synchronized crowd.
        order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(17)).tolist()]
        log['scene_order']=order
        def get_scene(scene):
            t=time.perf_counter();g=df[df.sceneId==scene]
            ids=g.metaId.drop_duplicates().astype(int).tolist()
            coordinates=torch.from_numpy(g[['x','y']].to_numpy(dtype='float32').reshape(-1,20,2)*cfg['resize'])
            assert len(ids)==len(coordinates)
            image=preprocess_scene(scene,cfg,'cuda:0');torch.cuda.synchronize()
            loaded=time.perf_counter()-t;t=time.perf_counter()
            sem=fixed_semantics(net,image);torch.cuda.synchronize()
            return coordinates,ids,image,sem,loaded,time.perf_counter()-t

        def step(phase,scene,offset,coordinates,ids,semantic):
            nonlocal totalsteps
            guard()
            batch=coordinates.to('cuda:0');t=time.perf_counter()
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
            opt.zero_grad(set_to_none=True)
            out=training_loss(net,batch,semantic,cfg,templates)
            if not all(torch.isfinite(out[k]) for k in ['loss','goal_loss','path_loss']):raise FloatingPointError('Nonfinite loss')
            out['loss'].backward();norm,nz,ng=gradients(net)
            opt.step();last.record();torch.cuda.synchronize();totalsteps+=1
            row=dict(phase=phase,step=totalsteps,scene=scene,target_start=offset,targets=len(batch),sample_ids='|'.join(manifest[i]['sample_id'] for i in ids),
                loss=float(out['loss']),goal_loss=float(out['goal_loss']),path_loss=float(out['path_loss']),gradient_norm=norm,
                nonzero_gradient_elements=nz,gradient_elements=ng,lr=opt.param_groups[0]['lr'],wall_seconds=time.perf_counter()-t,
                GPU_event_seconds=first.elapsed_time(last)/1000,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            writer.writerow(row);timing_file.flush();allsteps.append(row)
            if totalsteps%50==0 or totalsteps<=3:
                card=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
                memory.append(dict(step=totalsteps,phase=phase,scene=scene,card_used_MiB=card,allocated_bytes=row['allocated_bytes'],reserved_bytes=row['reserved_bytes']))
                print(phase,'step',totalsteps,scene,'targets',len(batch),'loss',round(row['loss'],4),'seconds',round(row['wall_seconds'],3),'elapsed',round(time.perf_counter()-start),flush=True)
            return row

        torch.cuda.reset_peak_memory_stats()
        smoke_scene=scenes[0];coords,ids,image,semantic,load_s,seg_s=get_scene(smoke_scene)
        # Gates TRAIN_FORWARD/BACKWARD/STEP, all original computation paths.
        for i,b in enumerate([1,4,4]):step('smoke',smoke_scene,i*4,coords[i*4:i*4+b],ids[i*4:i*4+b],semantic)
        assert state_digest(core_state(net))!=log['initial_predictor_state_SHA']
        assert state_digest(net.semantic_segmentation.state_dict())==frozen_before
        log.update(TRAIN_FORWARD=True,TRAIN_BACKWARD=True,OPTIMIZER_STEP=True,smoke_steps=3,smoke_target_visits=9)
        log['checkpoints'].append(save_checkpoint(CACHE/'checkpoints/smoke_after3.pt',net,opt,dict(phase='smoke',steps=3),settings))
        # One real TRAIN input no-GT inference and counterfactual check, no metrics.
        history=coords[:1,:8].cuda()
        a=ego_inference_no_GT(net,image,history,cfg,templates)
        changed=coords[:1].clone();changed[:,8:]=1e6
        b=ego_inference_no_GT(net,image,changed[:,:8].cuda(),cfg,templates)
        assert torch.equal(a,b) and torch.isfinite(a).all()
        log['INFERENCE_WITHOUT_GT']=True
        del coords,ids,image,semantic,a,b,history,changed;gc.collect()
        # Restore exact random predictor start and empty optimizer for the measured full pass.
        combined=net.state_dict();combined.update(initial);net.load_state_dict(combined,strict=True)
        seed_all(17);opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-4)
        assert state_digest(core_state(net))==log['initial_predictor_state_SHA'] and not opt.state
        log['complete_pass_from_same_random_initial_state']=True
        pass_start=time.perf_counter();step_start=totalsteps
        for scene in order:
            guard(10)
            coords,ids,image,semantic,loading,segmentation=get_scene(scene)
            scene_start=time.perf_counter();scene_steps=0
            for i in range(0,len(coords),4):
                row=step('full_pass',scene,i,coords[i:i+4],ids[i:i+4],semantic);scene_steps+=1
                visited.extend(manifest[mid]['sample_id'] for mid in ids[i:i+4])
            scene_records.append(dict(scene=scene,targets=len(coords),steps=scene_steps,loading_seconds=loading,
                segmentation_seconds=segmentation,training_seconds=time.perf_counter()-scene_start,padded_height=semantic.shape[-2],padded_width=semantic.shape[-1]))
            del coords,ids,image,semantic;gc.collect()
        log.update(complete_pass=True,full_pass_wall_seconds=time.perf_counter()-pass_start,full_pass_steps=totalsteps-step_start,
                   full_pass_target_visits=len(visited),full_pass_unique_targets=len(set(visited)),full_pass_recordings=len(scene_records))
        expected={r['sample_id'] for r in manifest.values()}
        assert len(visited)==7496 and len(set(visited))==7496 and set(visited)==expected
        assert state_digest(net.semantic_segmentation.state_dict())==frozen_before
        assert all(not m.training for m in net.semantic_segmentation.modules() if isinstance(m,torch.nn.BatchNorm2d))
        log['final_predictor_state_SHA']=state_digest(core_state(net));assert log['final_predictor_state_SHA']!=log['initial_predictor_state_SHA']
        log['last_checkpoint']=save_checkpoint(CACHE/'checkpoints/full_pass_last.pt',net,opt,dict(phase='full_pass',completed=True,steps=log['full_pass_steps'],targets=7496),settings)
        log['checkpoints'].append(log['last_checkpoint'])
        # Exact restored Adam/RNG continuation: two identical transient updates; last checkpoint stays unchanged.
        coords,ids,image,semantic,_,_=get_scene(smoke_scene)
        step('resume_reference',smoke_scene,0,coords[:4],ids[:4],semantic)
        reference=state_digest(core_state(net));reference_opt=optimizer_digest(opt)
        load_checkpoint(CACHE/'checkpoints/full_pass_last.pt',net,opt)
        assert state_digest(core_state(net))==log['final_predictor_state_SHA']
        step('resume_restored',smoke_scene,0,coords[:4],ids[:4],semantic)
        assert state_digest(core_state(net))==reference and optimizer_digest(opt)==reference_opt
        log['checkpoint_restore_and_next_step_exact']=True
        load_checkpoint(CACHE/'checkpoints/full_pass_last.pt',net,opt)
        log.update(status='ENGINEERING_TRAINING_PASS',frozen_segmentation_weights_and_BN_unchanged=True,
            frozen_final_SHA=state_digest(net.semantic_segmentation.state_dict()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            all_predictor_parameters_finite=all(torch.isfinite(p).all() for p in net.parameters()))
    except Exception as e:
        traceback.print_exc();log.update(status='ENGINEERING_TRAINING_INCOMPLETE',error_type=type(e).__name__,error=str(e))
        log['failures'].append(dict(error_type=type(e).__name__,error=str(e),step=totalsteps,last_scene=allsteps[-1]['scene'] if allsteps else None))
        if net is not None and opt is not None:
            try:log['checkpoints'].append(save_checkpoint(CACHE/'checkpoints/failure_scene.pt',net,opt,dict(steps=totalsteps,visited=len(visited)),settings))
            except Exception as e2:log['failure_checkpoint_error']=str(e2)
    finally:
        timing_file.close();log.update(total_optimizer_steps=totalsteps,cumulative_process_wall_seconds=time.perf_counter()-start,
              prior_budget_charge_seconds=prior,conservative_total_GPU_charge_seconds=prior+time.perf_counter()-start,
              GPU_budget_limit_seconds=3600,full_pass_targets_actually_visited=len(visited))
        assert log['conservative_total_GPU_charge_seconds']<3600, 'Budget exceeded'
        if memory:cnew(OUT/'training_memory.csv',memory)
        if scene_records:cnew(OUT/'scene_timing.csv',scene_records)
        if visited:cnew(OUT/'actual_target_visits.csv',[dict(visit=i,sample_id=s) for i,s in enumerate(visited)])
        jnew(OUT/'training_execution.json',log)
        print('Finished:',log['status'],'steps',totalsteps,'GPU conservative seconds',round(log['conservative_total_GPU_charge_seconds'],3),flush=True)
if __name__=='__main__':main()
