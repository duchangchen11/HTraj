"""New interpreter, strict fixed-weight restore, three target identity matches."""
from .common import *
from .predictor import Predictor
from scripts.stage11b1.data import load
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics,trajectory_maps
import torch
import torch.nn.functional as F

def main(guard):
    configure_torch();previous=json.loads((OUT/'real_checks_results.json').read_text())
    checkpoint=ROOT/previous['checkpoint_path'];assert sha(checkpoint)==previous['checkpoint_SHA']
    net,cfg=build_model(17,'cuda:0')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['training_steps']==0 and state['official_commit']==OFFICIAL
    info=net.load_state_dict(state['model'],strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    assert state_digest(net.state_dict())==previous['model_state_SHA']
    predictor=Predictor(net,cfg,make_templates(cfg,'cuda:0'));df=load('train')
    rows=json.loads((ROOT/'results/tables/stage11b1/precheck_selected_targets.json').read_text())['targets'][:3]
    records=[]
    for r in reversed(rows):
        guard();h=torch.as_tensor(df[df.metaId==int(r['meta_id'])][['x','y']].iloc[:8].to_numpy(dtype='float32')).unsqueeze(0)
        image=preprocess_scene(r['scene'],cfg,'cuda:0')
        p,g,trace=predictor.predict(r['sample_id'],h,image=image)
        reference=previous['primary_reference'][r['sample_id']]
        for key in reference:assert reference[key]==trace[key],(r['sample_id'],key)
        # Independent raw-pixel restoration using unchanged author neural ops.
        # The first goal is softargmax, so this needs no random draw.
        with torch.no_grad():
            semantic=fixed_semantics(net,image);H,W=semantic.shape[-2:]
            observed=trajectory_maps(h.cuda()*.25,H,W,predictor.templates[0])
            features=net.pred_features(torch.cat([semantic,observed],1))
            scaled_goal=net.softargmax(net.pred_goal(features)[:,[11]])
            assert torch.equal(g[0],(scaled_goal/.25).cpu()[:,0])
            wm=trajectory_maps(scaled_goal,H,W,predictor.templates[0])
            pyramid=[wm]+[F.avg_pool2d(wm,2**i,2**i) for i in range(1,len(features))]
            scaled_path=net.softargmax(net.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)]))
            assert torch.equal(p[0],(scaled_path/.25).cpu())
        records.append(dict(sample_id=r['sample_id'],**trace,bitwise_match=True,independent_raw_pixel_restoration=True))
    cnew(OUT/'fresh_process_checks.csv',records)
    jnew(OUT/'fresh_process_results.json',dict(passed=True,strict_checkpoint_restore=True,targets=3,
         all_probability_proposal_goal_path_SHA_equal=True,checkpoint_SHA=sha(checkpoint),training_steps=0,
         raw_pixel_restoration_independently_verified=True,
         interpretation='Fresh interpreter on same hardware/library stack; not cross-version portability'))
    print('Fresh interpreter3 identities bitwise PASS; strict restore PASS',flush=True)

if __name__=='__main__':timed_run('fresh_process',main)
