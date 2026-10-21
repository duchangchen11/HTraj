"""Full restore admission; disposable engineering updates never enter continuation."""
from .common import *
from scripts.stage11b1c.data import load,scene_arrays,input_records,manifest
from scripts.stage11b1c.engine import train_batch,checkpoint
from .validate import predict_all
from scripts.stage11b0.runtime import build_model,state_digest,load_checkpoint,rng_state,set_training
from scripts.stage11b0.run_admission import optimizer_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics,training_loss
import torch,numpy as np,os,traceback,io

def rng_digest(s):
    # Canonical state values, not pickle storage identifiers.
    h=hashlib.sha256();h.update(repr(s['python']).encode());h.update(repr(s['numpy']).encode())
    for t in [s['torch']]+s['cuda']:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()

def verify_restore(net,opt,saved):
    assert saved['progress']==json.loads((OUT/'checkpoint_static_audit.json').read_text())['progress']
    assert saved['progress']['epoch']==30 and saved['progress']['global_step']==56460
    assert saved['official_commit']==OFFICIAL
    assert state_digest(net.state_dict())==state_digest(saved['model'])
    states=opt.state_dict()['state'];assert len(states)==90
    assert all(float(s['step'])==56460 for s in states.values())
    assert opt.param_groups[0]['lr']==.0001 and len(opt.param_groups)==1
    assert len(opt.param_groups[0]['params'])==90
    assert optimizer_digest(opt)==json.loads((OUT/'checkpoint_static_audit.json').read_text())['optimizer_SHA']
    for k,v in saved['optimizer']['param_groups'][0].items():
        if k!='params':assert opt.state_dict()['param_groups'][0][k]==v
    named=[n for n,p in net.named_parameters() if p.requires_grad]
    assert named==json.loads((OUT/'checkpoint_static_audit.json').read_text())['optimizer_parameter_names']
    assert rng_digest(rng_state())==rng_digest(saved['rng'])
    return dict(model_state_SHA=state_digest(net.state_dict()),optimizer_SHA=optimizer_digest(opt),
        RNG_SHA=rng_digest(rng_state()),optimizer_states=len(states),Adam_steps=56460,
        optimizer_parameter_order_SHA=hashlib.sha256('\n'.join(named).encode()).hexdigest())

def main():
    configure_torch();assert torch.cuda.is_available()
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_PROCESS_MONOTONIC_START']))
    log=dict(passed=False,status='BLOCKED_RESTORE',formal_training=False,test_loaded=False)
    try:
        assert sha(CP)==CK
        net,cfg=build_model(17,'cuda:0');opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0001)
        saved=load_checkpoint(CP,net,opt);digest=verify_restore(net,opt,saved)
        seg=state_digest(net.semantic_segmentation.state_dict())
        assert seg=='79bd03ee82eea39723c27cba241f961669a53d2c47e7f26d4dce3b7bc949bd83'
        mode_layers=[dict(name=n,type=type(m).__name__) for n,m in net.named_modules() if not n.startswith('semantic_segmentation') and isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd,torch.nn.RNNBase))]
        assert not mode_layers,'Unexpected train/eval-dependent core layer; stop rather than repair'
        templates=make_templates(cfg,'cuda:0');df=load('train');scenes=sorted(df.sceneId.unique());scene=scenes[0]
        coords,ids=scene_arrays(df,scene);semantic=fixed_semantics(net,preprocess_scene(scene,cfg,'cuda:0'))
        # Core numerical mode test. training_loss itself resets predictor train, as in Stage11B-1C.
        x=torch.randn(1,14,64,64,device='cuda:0');net.eval()
        with torch.no_grad():ea=net.pred_goal(net.pred_features(x))
        set_training(net)
        with torch.no_grad():ta=net.pred_goal(net.pred_features(x))
        assert torch.equal(ea,ta)
        assert all(not m.training for m in net.semantic_segmentation.modules())
        original=training_loss(net,coords[:4].cuda(),semantic,cfg,templates,True)
        changed=coords[:4].cuda().clone();changed[:,8:]+=100000
        other=training_loss(net,changed,semantic,cfg,templates,True)
        assert torch.equal(original['goal_logits'],other['goal_logits'])
        assert not torch.equal(original['path_logits'],other['path_logits'])
        del original,other,changed,ea,ta,x
        # Restore again, checkpoint/resume then prove identical next optimizer update.
        load_checkpoint(CP,net,opt)
        ck=checkpoint(CACHE/'checkpoints/admission_restored.pt',net,opt,30,56460,[],dict(preflight=True))
        a=train_batch(net,opt,coords[:4].cuda(),semantic,cfg,templates)
        ref=state_digest(net.state_dict());refopt=optimizer_digest(opt)
        load_checkpoint(CACHE/'checkpoints/admission_restored.pt',net,opt)
        b=train_batch(net,opt,coords[:4].cuda(),semantic,cfg,templates)
        assert ref==state_digest(net.state_dict()) and refopt==optimizer_digest(opt) and a==b
        load_checkpoint(CP,net,opt);verify_restore(net,opt,saved)
        records=input_records('train')[:2];byid={r['sample_id']:r for r in manifest('train')};groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
        from scripts.stage11b1.evaluation import history_record
        cf=[]
        for r in records:
            xy=groups[int(byid[r['sample_id']]['meta_id'])][['x','y']].to_numpy(dtype='float32').reshape(20,2).copy();xy[8:]=1000000
            sid,h=history_record(dict(sample_id=r['sample_id'],xy=xy));cf.append(dict(sample_id=sid,scene=r['scene'],history=h[0].numpy()))
        p,g,_=predict_all(net,cfg,templates,records,30,dict(sha256=CK),budget,scope='preflight_original')
        q,h,_=predict_all(net,cfg,templates,cf,30,dict(sha256=CK),budget,scope='preflight_counterfactual')
        assert torch.equal(p,q) and torch.equal(g,h)
        order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(47)).tolist()]
        log.update(passed=True,status='RESTORE_ENGINEERING_GO',checkpoint_SHA=CK,**digest,epoch=30,global_step=56460,next_epoch=31,
            next_scene_order=order,next_scene_order_SHA=hashlib.sha256('|'.join(order).encode()).hexdigest(),
            total_parameters=53155339,predictor_parameters=1641381,segmentation_state_SHA=seg,
            training_config_SHA=sha(CONF/'train.json'),original_train_config_SHA=sha(ROOT/'configs/stage11b1c/train.json'),
            original_optimizer_paramgroups=saved['optimizer']['param_groups'],core_mode_sensitive_layers=mode_layers,
            core_train_eval_output_bitwise=True,old_training_loss_resets_train_each_batch=True,segmentation_always_eval=True,
            strict_checkpoint_restore_next_Adam_update_bitwise=True,future_GT_counterfactual_bitwise=True,
            K20_repeat_order_cache_bitwise=True,shape=list(p.shape),temporary_updates=2,
            formal_run_reloads_original_epoch30=True,checkpoint_optimizer_not_reset=True)
    except Exception as e:
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_seconds']=budget.elapsed();jnew(OUT/'resume_admission.json',log)
        print('RESTORE_ADMISSION',log['passed'],round(budget.elapsed(),3),flush=True)
if __name__=='__main__':main()
