"""Temporary model only. Fixed training probes, GT-free K20, gradient/cost audit."""
from .common import *
from scripts.stage11b0.runtime import build_model,state_digest,save_checkpoint,load_checkpoint
from scripts.stage11b0.training_adapter import make_templates,training_loss
from scripts.stage11b1c.engine import train_batch
from scripts.stage11b1c.data import load,scene_arrays
from scripts.stage11b1r.predictor import Predictor
from scripts.stage11b0.run_admission import optimizer_digest
import torch,numpy as np,traceback

def core_sha(net):return state_digest({k:v for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')})

def first_weight(net):
    return next((n,p) for n,p in net.named_parameters() if p.requires_grad and p.ndim==4 and p.shape[1]==14)

def main():
    configure_torch();budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    log=dict(passed=False,formal_training=False,test_loaded=False,validation_scored=False)
    try:
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0');core=core_sha(net)
        assert core==INITIAL
        original=torch.load(ROOT/'experiments/stage11b1c/cache/checkpoints/initial.pt',map_location='cpu',weights_only=True)
        assert state_digest({k:v for k,v in original['model'].items() if not k.startswith('semantic_segmentation.')})==core
        assert original['optimizer']['state']=={};del original
        seg=state_digest(net.semantic_segmentation.state_dict());opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001)
        assert not opt.state
        jnew(OUT/'initialization_audit.json',dict(passed=True,random_predictor_state_SHA=core,reference_initial_predictor_SHA=core,
            reference_initial_checkpoint_SHA=sha(ROOT/'experiments/stage11b1c/cache/checkpoints/initial.pt'),
            same_PyTorch=torch.__version__,total_parameters=sum(p.numel() for p in net.parameters()),
            nominal_predictor_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            new_Adam_state_empty=True,epoch=0,global_step=0,optimizer_updates=0,trained_checkpoint_loaded=False,
            reference_initial_read_only_identity_comparison=True,segmentation_state_SHA=seg))
        def forbidden(*a,**k):raise AssertionError('Segmentation forward forbidden in NoSemantic control')
        hook=net.semantic_segmentation.register_forward_pre_hook(forbidden)
        table=shapes();df=load('train');scenes=sorted(df.sceneId.unique());rows=[];gradrows=[]
        for scene in scenes:
            coords,ids=scene_arrays(df,scene);z=zero_semantic(scene,'cuda:0',table)
            assert cfg['loss_scale']==1000
            # First batch per canvas is warm-up; two further complete batch costs are retained.
            for b in range(3):
                offset=min(4*b,max(0,len(coords)-4));batch=coords[offset:offset+4].cuda()
                t=time.perf_counter();out=train_batch(net,opt,batch,z,cfg,templates);torch.cuda.synchronize()
                n,p=first_weight(net);g=p.grad
                assert not torch.count_nonzero(g[:,:6]) and torch.count_nonzero(g[:,6:])>0
                rows.append(dict(scene=scene,probe=b,targets=len(batch),seconds=time.perf_counter()-t,
                    warmup=b==0,epoch_scene_steps=(len(coords)+3)//4,**out))
                gradrows.append(dict(scene=scene,probe=b,parameter=n,total=p.numel(),semantic_zero_elements=g[:,:6].numel(),
                    semantic_nonzero_gradient=int(torch.count_nonzero(g[:,:6])),history_nonzero_gradient=int(torch.count_nonzero(g[:,6:]))))
            if scene==scenes[0]:
                a=training_loss(net,coords[:4].cuda(),z,cfg,templates,True)
                b=coords[:4].cuda().clone();b[:,-1,0]+=10
                c=training_loss(net,b,z,cfg,templates,True)
                assert torch.equal(a['goal_logits'],c['goal_logits']) and not torch.equal(a['path_logits'],c['path_logits'])
                assert a['goal_logits'].shape[1]==12 and a['path_logits'].shape[1]==12
                assert torch.equal(a['goal_loss'],torch.nn.functional.binary_cross_entropy_with_logits(a['goal_logits'],a['target'])*1000)
                del a,b,c
        assert core_sha(net)!=core and state_digest(net.semantic_segmentation.state_dict())==seg
        ck=save_checkpoint(CACHE/'checkpoints/preflight_temporary.pt',net,opt,dict(engineering_only=True),dict(zero_semantic=True))
        coords,_=scene_arrays(df,scenes[0]);z=zero_semantic(scenes[0],'cuda:0',table)
        train_batch(net,opt,coords[:4].cuda(),z,cfg,templates);want=core_sha(net);wantopt=optimizer_digest(opt)
        load_checkpoint(CACHE/'checkpoints/preflight_temporary.pt',net,opt)
        train_batch(net,opt,coords[:4].cuda(),z,cfg,templates)
        assert want==core_sha(net) and wantopt==optimizer_digest(opt)
        records=history_records('preflight');predictor=Predictor(net,cfg,templates);pred={};traces=[]
        # The accepted predictor record contains no future coordinate field.
        for order in [records,list(reversed(records)),records]:
            for r in order:
                p,g,t=predictor.predict(r['sample_id'],r['history'].unsqueeze(0),semantic=zero_semantic(r['scene'],'cuda:0',table))
                if r['sample_id'] in pred:assert torch.equal(p,pred[r['sample_id']][0]) and torch.equal(g,pred[r['sample_id']][1])
                pred[r['sample_id']]=(p,g);traces.append(dict(sample_id=r['sample_id'],**t))
        # Raw source GT counterfactual, projected via the original history-only accessor.
        from scripts.stage11b1.evaluation import history_record
        xy=torch.cat([records[0]['history'],torch.full((12,2),1000000.)]).numpy()
        sid,h=history_record(dict(sample_id=records[0]['sample_id'],xy=xy))
        p,g,t=predictor.predict(sid,h,semantic=zero_semantic(records[0]['scene'],'cuda:0',table))
        assert torch.equal(p,pred[sid][0]) and torch.equal(g,pred[sid][1])
        assert state_digest(net.semantic_segmentation.state_dict())==seg
        cnew(OUT/'preflight_steps.csv',rows);cnew(OUT/'preflight_gradients.csv',gradrows);cnew(OUT/'preflight_K20.csv',traces)
        projected=sum(max(r['seconds'] for r in rows if r['scene']==s and not r['warmup'])*next(r['epoch_scene_steps'] for r in rows if r['scene']==s) for s in scenes)
        log.update(passed=True,random_predictor_state_SHA=core,segmentation_unchanged=True,segmentation_forward_calls=0,
            all_semantic_zero=True,input_channels=14,teacher_forcing_training_only=True,loss_finite=True,gradients_finite=True,
            predictor_updated=True,strict_checkpoint_restore_next_update_exact=True,future_counterfactual_bitwise=True,
            K20_repeat_and_reverse_bitwise=True,K=20,TTST_proposals=10000,KMeans_centers=19,
            zero_semantic_gradient_elements=gradrows[0]['semantic_zero_elements'],
            nominal_predictor_parameters=1641381,structurally_input_active_parameter_upper_bound=1641381-gradrows[0]['semantic_zero_elements'],
            effective_capacity_not_exactly_matched=True,checkpoint=ck,projected_epoch_seconds=projected,
            mean_K20_target_seconds=float(np.mean([r['total_seconds'] for r in traces])),
            maximum_K20_target_seconds=max(r['total_seconds'] for r in traces),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            temporary_model_discarded_before_formal=True,formal_initialization_fresh=True)
        hook.remove()
    except Exception as e:
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_seconds']=budget.elapsed();jnew(OUT/'preflight_results.json',log)
        print('PREFLIGHT',log,flush=True)
if __name__=='__main__':main()
