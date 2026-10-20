"""New integrated GPU entry tests, then exit; formal training starts afresh."""
from .common import *
from .data import load,scene_arrays,input_records
from .engine import train_batch,checkpoint
from .validate import predict_all
from scripts.stage11b0.runtime import build_model,state_digest,load_checkpoint,seed_all
from scripts.stage11b0.run_admission import optimizer_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics,training_loss
import torch,numpy as np,os,traceback

def main():
    configure_torch();assert torch.cuda.is_available()
    budget=Budget(parent_seconds(),started=float(os.environ.get('HTRAJ_PROCESS_MONOTONIC_START',time.perf_counter())))
    log=dict(passed=False,formal_training=False,validation_or_test_metrics=False)
    try:
        net,cfg=build_model(17,'cuda:0');templates=make_templates(cfg,'cuda:0')
        assert sum(p.numel() for p in net.parameters())==53155339
        assert sum(p.numel() for p in net.parameters() if p.requires_grad)==1641381
        core=state_digest({k:v for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')})
        frozen=state_digest(net.semantic_segmentation.state_dict())
        log['random_predictor_state_SHA']=core
        df=load('train');scene=sorted(df.sceneId.unique())[0];coords,ids=scene_arrays(df,scene)
        image=preprocess_scene(scene,cfg,'cuda:0');semantic=fixed_semantics(net,image)
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001)
        original=training_loss(net,coords[:4].cuda(),semantic,cfg,templates,True)
        goal_equation=torch.nn.functional.binary_cross_entropy_with_logits(original['goal_logits'],original['target'])*1000
        assert torch.equal(goal_equation,original['goal_loss'])
        changed=coords[:4].cuda().clone();changed[:,-1,0]+=10
        other=training_loss(net,changed,semantic,cfg,templates,True)
        assert torch.equal(original['goal_logits'],other['goal_logits'])
        assert not torch.equal(original['path_logits'],other['path_logits']),'Teacher-forced goal input must affect path'
        del original,other,changed
        step=train_batch(net,opt,coords[:4].cuda(),semantic,cfg,templates)
        assert step['gradient_norm']>0
        assert core!=state_digest({k:v for k,v in net.state_dict().items() if not k.startswith('semantic_segmentation.')})
        assert frozen==state_digest(net.semantic_segmentation.state_dict())
        ck=checkpoint(CACHE/'checkpoints/preflight_after1.pt',net,opt,0,1,[scene],dict(preflight=True))
        step2=train_batch(net,opt,coords[:4].cuda(),semantic,cfg,templates)
        reference=state_digest(net.state_dict());reference_opt=optimizer_digest(opt)
        load_checkpoint(CACHE/'checkpoints/preflight_after1.pt',net,opt)
        again=train_batch(net,opt,coords[:4].cuda(),semantic,cfg,templates)
        assert reference==state_digest(net.state_dict()) and reference_opt==optimizer_digest(opt)
        load_checkpoint(CACHE/'checkpoints/preflight_after1.pt',net,opt)
        assert frozen==state_digest(net.semantic_segmentation.state_dict())
        records=input_records('train')[:2]
        # Actual future labels are deliberately corrupted outside the predictor.
        # Re-project through the same history-only interface; future arrays are
        # never fields in predict_all's accepted record schema.
        groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
        byid={r['sample_id']:r for r in __import__('scripts.stage11b1c.data',fromlist=['manifest']).manifest('train')}
        from scripts.stage11b1.evaluation import history_record
        cf=[]
        for r in records:
            xy=groups[int(byid[r['sample_id']]['meta_id'])][['x','y']].to_numpy(dtype='float32').reshape(20,2).copy()
            xy[8:]=1000000
            sid,h=history_record(dict(sample_id=r['sample_id'],xy=xy))
            cf.append(dict(sample_id=sid,scene=r['scene'],history=h[0].numpy()))
        p,g,pi=predict_all(net,cfg,templates,records,0,ck,budget,scope='preflight_original')
        q,h,qi=predict_all(net,cfg,templates,cf,0,ck,budget,scope='preflight_counterfactual')
        assert torch.equal(p,q) and torch.equal(g,h)
        assert frozen==state_digest(net.semantic_segmentation.state_dict())
        assert all(not m.training for m in net.semantic_segmentation.modules() if isinstance(m,torch.nn.BatchNorm2d))
        log.update(passed=True,parameters=53155339,predictor_parameters=1641381,segmentation_state_SHA=frozen,
            loss_finite=True,backward=True,optimizer_step=True,segmentation_unchanged=True,teacher_forcing_checked=True,
            strict_checkpoint_restore_and_next_update_exact=True,K20_shared_validation_entry=True,
            K20_repeat_order_cache_bitwise=True,future_GT_counterfactual_bitwise=True,shape=list(p.shape),
            original_pixel_restoration='Immutable Stage11B-1R /0.25; integrated synthetic formula test also required',
            preflight_checkpoint=ck,temporary_updates=3,formal_model_uses_none_of_these_weights=True)
    except Exception as e:
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_seconds']=budget.elapsed();jnew(OUT/'preflight_results.json',log)
        print('Integrated preflight PASS:',log['passed'],'seconds',round(budget.elapsed(),3),flush=True)

if __name__=='__main__':main()
