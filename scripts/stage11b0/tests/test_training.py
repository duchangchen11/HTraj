"""CPU admission gates before any real GPU training work."""
import copy,inspect,json,random
from pathlib import Path
import numpy as np,pandas as pd,pytest,torch
from scripts.stage11b0.runtime import (build_model,original_modules,state_digest,save_checkpoint,load_checkpoint,seed_all)
from scripts.stage11b0.training_adapter import training_loss,fixed_semantics,make_templates,gradients,ego_inference_no_GT
from scripts.stage11b0.evaluation import metrics
from scripts.stage11ap.adapters import SceneDatasetAdapter,scene_collate_adapter,augment_data_adapter
from scripts.stage11b0.data_audit import load_train,canonical_sha

@pytest.fixture(scope='module')
def exercised_model(tmp_path_factory):
    torch.set_num_threads(1)
    net,cfg=build_model(17,'cpu')
    image=torch.randn(1,3,64,96)
    templates=make_templates(cfg,'cpu')
    trajectory=torch.stack([torch.arange(20.)*.5+15,torch.arange(20.)*.25+20],1).unsqueeze(0)
    before=state_digest(net.semantic_segmentation.state_dict())
    semantic=fixed_semantics(net,image)
    initial=state_digest(net.encoder.state_dict())
    out=training_loss(net,trajectory,semantic,cfg,templates,True)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-4)
    opt.zero_grad(set_to_none=True);out['loss'].backward();audit=gradients(net);opt.step()
    checkpoint=tmp_path_factory.mktemp('checkpoint')/'test.pt'
    ck=save_checkpoint(checkpoint,net,opt,dict(step=1),dict(test=True))
    return dict(net=net,cfg=cfg,image=image,templates=templates,trajectory=trajectory,semantic=semantic,loss=out,
                frozen_before=before,encoder_before=initial,opt=opt,gradient=audit,path=checkpoint,checkpoint=ck)

def test_original_loss_finite_and_equations(exercised_model):
    z=exercised_model;r=z['loss'];scale=z['cfg']['loss_scale']
    assert torch.isfinite(r['loss'])
    expected=torch.nn.BCEWithLogitsLoss()(r['goal_logits'],r['target'])*scale
    torch.testing.assert_close(expected,r['goal_loss'])
    torch.testing.assert_close(r['loss'],r['goal_loss']+r['path_loss'])

def test_gradients_and_actual_parameter_update(exercised_model):
    z=exercised_model
    assert z['gradient'][0]>0 and z['gradient'][1]>0
    assert state_digest(z['net'].encoder.state_dict())!=z['encoder_before']
    assert all(torch.isfinite(p).all() for p in z['net'].parameters())

def test_frozen_segmentation_parameters_and_BN(exercised_model):
    z=exercised_model
    assert z['frozen_before']==state_digest(z['net'].semantic_segmentation.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in z['net'].semantic_segmentation.parameters())
    assert all(not m.training for m in z['net'].semantic_segmentation.modules() if isinstance(m,torch.nn.BatchNorm2d))

def test_checkpoint_exact_state_optimizer_and_continuation(exercised_model):
    z=exercised_model
    before=state_digest(z['net'].state_dict());step0=[v['step'].clone() for v in z['opt'].state.values()]
    torch.rand(3);np.random.rand();random.random()
    saved=load_checkpoint(z['path'],z['net'],z['opt'])
    assert state_digest(z['net'].state_dict())==before and saved['progress']['step']==1
    assert all(torch.equal(a,b['step']) for a,b in zip(step0,z['opt'].state.values()))
    expected=torch.rand(3);load_checkpoint(z['path'],z['net'],z['opt']);torch.testing.assert_close(torch.rand(3),expected,rtol=0,atol=0)
    semantic=fixed_semantics(z['net'],z['image'])
    r=training_loss(z['net'],z['trajectory'],semantic,z['cfg'],z['templates'])
    z['opt'].zero_grad(set_to_none=True);r['loss'].backward();gradients(z['net']);z['opt'].step()
    assert state_digest(z['net'].state_dict())!=before
    load_checkpoint(z['path'],z['net'],z['opt'])

def test_same_seed_initialization_and_scene_order(exercised_model):
    net,_=build_model(17,'cpu');net2,_=build_model(17,'cpu')
    assert state_digest(net.encoder.state_dict())==state_digest(net2.encoder.state_dict())
    seed_all(17);a=torch.randperm(25);seed_all(17);assert torch.equal(a,torch.randperm(25))

def test_inference_has_no_future_dependency(exercised_model):
    z=exercised_model;args=list(inspect.signature(ego_inference_no_GT).parameters)
    assert 'future' not in args and 'truth' not in args
    a=z['trajectory'].clone();b=a.clone();b[:,8:]=100000
    p=ego_inference_no_GT(z['net'],z['image'],a[:,:8],z['cfg'],z['templates'])
    q=ego_inference_no_GT(z['net'],z['image'],b[:,:8],z['cfg'],z['templates'])
    torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_scene_and_GT_identity_no_mixing():
    df=load_train();ds=SceneDatasetAdapter(df,.25,20)
    assert len(ds)==25 and sum(len(t) for t in ds.trajectories)==7496
    for i in [0,len(ds)-1]:
        t,meta,s=scene_collate_adapter([ds[i]])
        assert meta[0].sceneId.nunique()==1 and meta[0].sceneId.iloc[0]==s
        expected=meta[0][['x','y']].to_numpy(dtype='float32').reshape(-1,20,2)*.25
        np.testing.assert_array_equal(t.numpy(),expected)
        for frame in meta[0].groupby('metaId',sort=False):assert len(frame[1])==20

def test_real_scene_image_and_segmentation_identity():
    from scripts.stage11b0.common import OUT,RES,ROOT,sha
    inventory={r['path']:r['sha256'] for r in __import__('csv').DictReader((ROOT/'results/tables/stage11ap/extracted_resource_SHA256.csv').open())}
    for scene in sorted(load_train().sceneId.unique()):
        r=f'ynet_additional_files/data/SDD/train/{scene}/reference.jpg'
        m=f'ynet_additional_files/data/SDD_semantic_maps/train_masks/{scene}_mask.png'
        assert sha(RES/f'data/SDD/train/{scene}/reference.jpg')==inventory[r]
        assert sha(RES/f'data/SDD_semantic_maps/train_masks/{scene}_mask.png')==inventory[m]

def test_augmentation_image_trajectory_affine_and_eightfold(tmp_path):
    import cv2
    image=np.arange(40*60*3,dtype=np.uint8).reshape(40,60,3)
    folder=tmp_path/'a';folder.mkdir();cv2.imwrite(str(folder/'reference.png'),image)
    df=pd.DataFrame(dict(sceneId=['a']*20,metaId=[0]*20,x=np.linspace(10,30,20),y=np.linspace(10,20,20)))
    aug,ims=augment_data_adapter(df,tmp_path,original_modules().preprocess,image_file='reference.png')
    assert len(aug)==160 and len(ims)==8
    for k,name in [(1,'_rot90'),(2,'_rot180'),(3,'_rot270')]:
        np.testing.assert_array_equal(ims['a'+name],np.rot90(image,k).copy())
        expected=df[['x','y']].to_numpy()-[30,20]
        theta=-k*np.pi/2;c,s=np.cos(theta),np.sin(theta)
        expected=expected@np.array([[c,s],[-s,c]])+np.array([ims['a'+name].shape[1]/2,ims['a'+name].shape[0]/2])
        np.testing.assert_allclose(aug[aug.sceneId=='a'+name][['x','y']],expected)

def test_retained_outside_points_have_valid_heatmaps(exercised_model):
    from scripts.stage11b0.training_adapter import trajectory_maps
    z=exercised_model
    points=torch.tensor([[[361.5,1950.5],[367.5,1956.5]]])*.25
    maps=trajectory_maps(points,512,352,z['templates'][0])
    assert maps.shape==(1,2,512,352) and torch.isfinite(maps).all()

def test_metric_independent_best_heads_and_goal_path_distinction():
    gt=torch.zeros(1,3,2)
    pred=torch.tensor([[[[0.,0.],[0.,0.],[9.,0.]]],[[[4.,0.],[4.,0.],[0.,0.]]]])
    goals=torch.ones(2,1,2)*20
    r=metrics(pred,gt,goals)
    assert r['best_ADE_head'].item()==1 and r['best_FDE_head'].item()==1
    # A head with better average but worse final point must be independently selectable.
    pred[0,0,-1,0]=5
    r=metrics(pred,gt,goals)
    assert r['best_ADE_head'].item()==0 and r['best_FDE_head'].item()==1
    assert r['minGoalFDE_K'].item()>r['minFDE_K'].item()

def test_metric_translation_and_budget_shapes():
    gt=torch.zeros(2,12,2);p=torch.ones(20,2,12,2)
    a=metrics(p,gt);b=metrics(p+42,gt+42)
    torch.testing.assert_close(a['minADE_K'],b['minADE_K'])
    with pytest.raises(ValueError):metrics(p,gt[:,:8])
