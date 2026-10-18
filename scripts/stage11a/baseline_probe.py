"""CPU synthetic interfaces only; no segmentation weights or real prediction."""
from common import *
import importlib
import importlib.metadata
import importlib.util
import time
import numpy as np
import torch

def main():
    official=CACHE/'official/ynet'
    expected=json.loads((OUT/'source_download.json').read_text())
    before={r['path']:r['sha256'] for r in expected['files'] if r['verified']}
    env={name:(importlib.metadata.version(name) if importlib.util.find_spec(module) else None)
        for name,module in [('torch','torch'),('numpy','numpy'),('pandas','pandas'),('opencv-python','cv2'),
            ('matplotlib','matplotlib'),('segmentation-models-pytorch','segmentation_models_pytorch')]}
    sys.path.insert(0,str(official))
    torch.set_num_threads(1);torch.manual_seed(17)
    log=dict(environment=env,python=sys.version,device='cpu',official_commit=expected['commit'],
        semantic_backbone_weights_obtained=False,prediction_weights_loaded=False,
        author_config_available=False,optimizer_updates=0,real_data_prediction=False,
        engineering_config=dict(obs_len=8,pred_len=12,semantic_classes=6,segmentation_model_fp=None,
            encoder_channels=[32,32,64,64,64],decoder_channels=[64,64,64,32,32],waypoints=1,
            meaning='Author classes with an explicitly engineering-only config; not claimed exact pretrained configuration'),
        failures=[])
    try:
        module=importlib.import_module('model')
        log['CPU_import_passed']=True
        model=module.YNetTorch(**{k:v for k,v in log['engineering_config'].items() if k!='meaning'})
        model.eval()
        state={k:v.clone() for k,v in model.state_dict().items()}
        records=[]
        for batch in (1,2):
            x=torch.rand(batch,14,64,96)
            t=time.perf_counter()
            with torch.no_grad():
                features=model.pred_features(x)
                goal=model.pred_goal(features)
                arbitrary_waypoint=torch.rand(batch,1,64,96)
                down=[arbitrary_waypoint]+[torch.nn.functional.avg_pool2d(arbitrary_waypoint,2**i,2**i) for i in range(1,len(features))]
                conditioned=[torch.cat([f,w],1) for f,w in zip(features,down)]
                traj=model.pred_traj(conditioned)
                coords=model.softargmax(traj)
            assert list(goal.shape)==[batch,12,64,96] and list(coords.shape)==[batch,12,2]
            assert torch.isfinite(goal).all() and torch.isfinite(traj).all() and torch.isfinite(coords).all()
            records.append(dict(batch=batch,input_shape=list(x.shape),feature_shapes=[list(f.shape) for f in features],
                goal_shape=list(goal.shape),trajectory_shape=list(traj.shape),coordinate_shape=list(coords.shape),
                wall_seconds=time.perf_counter()-t,finite=True))
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        log.update(synthetic_interface_passed=True,parameters=sum(p.numel() for p in model.parameters()),
            segmentation_parameters=0,records=records,weights_unchanged=True,
            parameter_float32_bytes=4*sum(p.numel() for p in model.parameters()))
        # Full pipeline dependency is tested explicitly, not silently stubbed.
        try:
            importlib.import_module('segmentation_models_pytorch')
            log['segmentation_dependency_import']=True
        except Exception as e:
            log['segmentation_dependency_import']=False
            log['failures'].append(dict(stage='full RGB segmentation pipeline import',type=type(e).__name__,message=str(e)))
        # Original SceneDataset ragged scene counts against current NumPy.
        import pandas as pd
        dl=importlib.import_module('utils.dataloader')
        dfs=[]
        for scene,people in [('synthetic_A',1),('synthetic_B',2)]:
            for person in range(people):
                for f in range(20):dfs.append(dict(sceneId=scene,x=float(f),y=float(person),metaId=person,frame=f))
        try:
            ds=dl.SceneDataset(pd.DataFrame(dfs),1.,20)
            log['original_ragged_scene_dataset_passed']=True
        except Exception as e:
            log['original_ragged_scene_dataset_passed']=False
            log['failures'].append(dict(stage='original ragged SceneDataset',type=type(e).__name__,message=str(e)))
    except Exception as e:
        log['synthetic_interface_passed']=False
        log['failures'].append(dict(stage='core import/forward',type=type(e).__name__,message=str(e)))
    assert all(sha(CACHE/'official'/p)==s for p,s in before.items())
    log['frozen_source_unchanged']=True
    log['GPU_memory_measured']=False
    log['memory_estimate_limit']='Synthetic 64x96 CPU test cannot determine full-scene RTX3080 training memory or epoch time; segmentation and image size unresolved'
    jnew(OUT/'baseline_probe.json',log)
    print(json.dumps(log,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
