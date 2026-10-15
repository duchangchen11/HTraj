"""Bounded mismatch diagnosis and synthetic structure checks, not prediction results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,time
import torch
from src.evaluation.public_baseline import state_dict_audit
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from scripts.evaluate_stage4_eqmotion import geometry


def main():
    root=Path('results/tables/stage4');torch.set_num_threads(1)
    sys.path.insert(0,str(Path('third_party/stage4/eqmotion').resolve()))
    from eth_ucy.model_t import EqMotion
    records=[]
    for fold,channels in [('hotel',64),('zara1',128)]:
        t=time.perf_counter()
        model=EqMotion(8,2,64,8,channels,12,device='cpu',n_layers=4,recurrent=True,norm_diff=False,tanh=False)
        checkpoint=Path('data/external/stage4/assets/eqmotion/eth_ucy/saved_models')/(fold+'_ckpt.pth.tar')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
        audit=state_dict_audit(model,state)
        try:
            model.load_state_dict(state,strict=True)
        except RuntimeError:
            strict_pass=False
        else:strict_pass=True
        assert not strict_pass
        records.append(dict(fold=fold,channels=channels,checkpoint_sha256=sha256(checkpoint),strict_load_pass=strict_pass,
            **audit,diagnostic_seconds=time.perf_counter()-t,evaluated_target_windows=0,ADE_m=None,FDE_m=None,
            decision='stop checkpoint performance evaluation; do not accept random missing parameters or silently discard unverified modules'))
    # CPU source-only geometry test: synthetic causal history, no real data or checkpoint.
    torch.manual_seed(17)
    model=EqMotion(8,2,64,8,64,12,device='cpu',n_layers=4,recurrent=True,norm_diff=False,tanh=False).eval()
    loc=torch.randn(2,3,8,2).cumsum(2);loc[1,2]=0;num=torch.tensor([3,2],dtype=torch.int32)
    t=time.perf_counter()
    with torch.no_grad():checks=geometry(model,loc,num)
    seconds=time.perf_counter()-t
    rows=[dict(r,scope='synthetic random-weight source check only; no checkpoint metrics',device='cpu',seconds=seconds,
        pass_tolerance_1e_minus3=r['finite'] and r['valid_agent_prediction_max_error_m']<1e-3 and r['category_max_difference']<1e-3) for r in checks]
    csv_new(root/'eqmotion_synthetic_geometry_checks.csv',rows)
    write_json_new(root/'eqmotion_execution.json',dict(status='blocked_checkpoint_source_mismatch',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256(__file__),protocol_sha256=sha256('configs/stage4/protocol.json'),diagnostics=records,
        initial_log='experiments/stage4/eqmotion_execution.log',initial_log_sha256=sha256('experiments/stage4/eqmotion_execution.log'),
        planned_folds=['hotel','zara1'],planned_max_target_cases_per_fold=128,actual_evaluated_targets=0,training=False,ETH_evaluated=False,
        paper_reproduced=False,source_only_synthetic_checks=True,geometry_seconds=seconds,
        amendment='finite planned checkpoint validation stopped at strict key checks; only bounded key diagnosis and source-only synthetic geometry checks performed; no architecture changes, missing-weight initialization or relaxed-load retries'))
    print(json.dumps(dict(status='blocked',folds=[dict(fold=r['fold'],missing=len(r['missing_keys']),extra=len(r['unexpected_keys']),shapes=len(r['shape_mismatches'])) for r in records],geometry=rows)))


if __name__=='__main__':main()
