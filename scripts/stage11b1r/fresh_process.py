"""New interpreter, strict fixed-weight restore, three target identity matches."""
from .common import *
from .predictor import Predictor
from scripts.stage11b1.data import load
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene
import torch

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
        p,g,trace=predictor.predict(r['sample_id'],h,image=preprocess_scene(r['scene'],cfg,'cuda:0'))
        reference=previous['primary_reference'][r['sample_id']]
        for key in reference:assert reference[key]==trace[key],(r['sample_id'],key)
        records.append(dict(sample_id=r['sample_id'],**trace,bitwise_match=True))
    cnew(OUT/'fresh_process_checks.csv',records)
    jnew(OUT/'fresh_process_results.json',dict(passed=True,strict_checkpoint_restore=True,targets=3,
         all_probability_proposal_goal_path_SHA_equal=True,checkpoint_SHA=sha(checkpoint),training_steps=0,
         interpretation='Fresh interpreter on same hardware/library stack; not cross-version portability'))
    print('Fresh interpreter3 identities bitwise PASS; strict restore PASS',flush=True)

if __name__=='__main__':timed_run('fresh_process',main)
