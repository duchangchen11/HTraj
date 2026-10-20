"""Few TRAIN-only real forwards: official weights, predicted goals, no GT in API.

Inference-only wrapper of official test.py's goal-only branch. It calls original
segmentation/encoder/goal/path/TTST functions; removes all GT and metric access.
No new model, no losses, no optimizer, no checkpoint selection.
"""
from common import *
from safe_data import load_data_pickle
from checkpoint_io import load_author_segmentation
import numpy as np, torch, torch.nn as nn, yaml, cv2

RES=CACHE/'author_resources/ynet_additional_files'

def predicted_paths(model,image,history,params,device):
    """history is exactly [N,8,2] in raw author pixel coordinates; no future arg."""
    assert history.ndim==3 and history.shape[1:]==(8,2)
    assert params['waypoints']==[11] and params['use_CWS'] is False
    from utils.image_utils import resize,pad,preprocess_image_for_segmentation,create_dist_mat,get_patch,sampling
    from utils.kmeans import kmeans
    images={'scene':image.copy()}
    resize(images,params['resize']);pad(images,2**len(params['encoder_channels']))
    preprocess_image_for_segmentation(images)
    template=torch.as_tensor(create_dist_mat(int(4200*params['resize'])),dtype=torch.float32,device=device)
    with torch.inference_mode():
        semantic=model.segmentation(images['scene'].unsqueeze(0).to(device))
        _,_,H,W=semantic.shape
        observed=np.asarray(history,dtype=np.float32)*params['resize']
        maps=torch.stack(get_patch(template,observed.reshape(-1,2),H,W)).reshape(-1,8,H,W)
        features=model.pred_features(torch.cat([semantic.expand(len(history),-1,-1,-1),maps],1))
        goals=model.pred_goal(features)[:,params['waypoints']]
        probabilities=model.sigmoid(goals/params['temperature'])
        assert torch.isfinite(probabilities).all()
        # Same TTST semantics as the official author evaluator: 10k then19means+softargmax.
        proposals=sampling(probabilities[:,-1:],10000,replacement=True,rel_threshold=params['rel_threshold']).permute(2,0,1,3)
        clusters=[]
        for person in range(len(history)):
            _,centers=kmeans(X=proposals[:,person,0],num_clusters=19,distance='euclidean',device=device,tqdm_flag=False,tol=.001,iter_limit=1000)
            clusters.append(centers)
        candidates=torch.stack(clusters).permute(1,0,2).unsqueeze(2)
        candidates=torch.cat([model.softargmax(goals[:,-1:]).unsqueeze(0),candidates],0)
        paths=[]
        for waypoint in candidates:
            wm=torch.stack(get_patch(template,waypoint.reshape(-1,2).cpu().numpy(),H,W)).reshape(-1,1,H,W)
            pyramid=[wm]+[nn.AvgPool2d(2**i,2**i)(wm) for i in range(1,len(features))]
            output=model.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)])
            paths.append(model.softargmax(output))
        paths=torch.stack(paths)
        assert paths.shape==(20,len(history),12,2) and torch.isfinite(paths).all()
    return paths/params['resize'],candidates/params['resize'],semantic

def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(17);np.random.seed(17)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sys.path.insert(0,str(CACHE/'official/ynet'))
    from model import YNetTorch
    params=yaml.safe_load((RES/'config/sdd_trajnet.yaml').read_text())
    log=dict(source_commit=MODEL_COMMIT,device=str(device),seed=17,train_only=True,optimizer_updates=0,loss_computed=False,
             metrics_computed=False,GT_provided_to_prediction=False,use_TTST=True,TTST_proposals=10000,num_goals=20,num_traj=1,CWS=False,
             official_core_unchanged=True,official_evaluate_called=False,cases=[],passed=False)
    try:
        seg,initial_error=load_author_segmentation(RES/params['segmentation_model_fp'])
        log['weights_only_legacy_error']=initial_error
        model=YNetTorch(8,12,None,params['use_features_only'],params['semantic_classes'],params['encoder_channels'],params['decoder_channels'],len(params['waypoints']))
        model.semantic_segmentation=seg  # same module as author constructor, safely loaded outside it
        state=torch.load(RES/'pretrained_models/sdd_trajnet_weights.pt',map_location='cpu',weights_only=True)
        checkpoint_shapes={k:list(v.shape) for k,v in state.items()}
        expected_shapes={k:list(v.shape) for k,v in model.state_dict().items()}
        assert checkpoint_shapes==expected_shapes, 'Parameter keys/shapes mismatch'
        loaded=model.load_state_dict(state,strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        log.update(strict_load=True,checkpoint_keys=len(state),checkpoint_SHA=sha(RES/'pretrained_models/sdd_trajnet_weights.pt'),
                   segmentation_SHA=sha(RES/params['segmentation_model_fp']),parameters=sum(p.numel() for p in model.parameters()),
                   core_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('semantic_segmentation.')),
                   segmentation_parameters=sum(p.numel() for p in model.semantic_segmentation.parameters()),parameter_shapes=checkpoint_shapes)
        del state
        model=model.eval().to(device)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        df,_=load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
        candidates=json.loads((OUT/'split_identity_audit.json').read_text())['candidate_val_scenes']
        scenes=[]
        for site in sorted({s.rsplit('_',1)[0] for s in df.sceneId.unique()}):
            scenes.append(sorted(s for s in df.sceneId.unique() if s.startswith(site+'_') and s not in candidates)[0])
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        for scene in scenes:
            g=next(iter(df[df.sceneId==scene].groupby('metaId',sort=True)))[1]
            h=g[['x','y']].to_numpy(dtype=np.float32)[:8].copy()[None]
            image=cv2.imread(str(RES/'data/SDD/train'/scene/'reference.jpg'))
            t=time.perf_counter()
            paths,goals,sem=predicted_paths(model,image,h,params,device)
            if device.type=='cuda':torch.cuda.synchronize()
            row=dict(scene=scene,meta_id=int(g.metaId.iloc[0]),person_id=int(g.trackId.iloc[0]),frame=int(g.frame.iloc[0]),
                     history_SHA=hashlib.sha256(h.tobytes()).hexdigest(),reference_SHA=sha(RES/'data/SDD/train'/scene/'reference.jpg'),
                     path_shape=list(paths.shape),goal_shape=list(goals.shape),semantic_shape=list(sem.shape),semantic_probability_sum_max_error=float((sem.sum(1)-1).abs().max()),
                     finite=bool(torch.isfinite(paths).all()),seconds=time.perf_counter()-t,unit='original reference pixels',
                     source_resize=params['resize'],GT_input=False)
            cnew(OUT/f'forward_{scene}_predicted_paths.csv',[dict(candidate=k,time_step=t+1,x=float(paths[k,0,t,0]),y=float(paths[k,0,t,1])) for k in range(20) for t in range(12)])
            log['cases'].append(row);print(scene,'paths',row['path_shape'],'seconds',round(row['seconds'],3),flush=True)
            if time.perf_counter()-start>3300:raise RuntimeError('Engineering wall-clock stop before1 GPU hour')
        assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
        log.update(passed=True,status='REAL_TRAIN_INPUT_PIPELINE_PASS; NOT official benchmark performance reproduction',weights_unchanged=True,
                   gpu_peak_allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                   gpu_peak_reserved=torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
                   GPU_card_state=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip() if device.type=='cuda' else None)
    except Exception as e:
        import traceback
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e))
    log['wall_seconds']=time.perf_counter()-start
    jnew(OUT/'full_pipeline_admission.json',log)
    print('Admission passed:',log['passed'],flush=True)
if __name__=='__main__':main()
