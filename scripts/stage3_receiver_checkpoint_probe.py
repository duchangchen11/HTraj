"""Same old H1 weights, receiver-policy inference only; not a new training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess
import torch
from src.training.order_verification import datasets
from src.training.social_diagnostics import csv_new
from src.data.preprocessing import sha256,write_json_new
from src.evaluation.relation_routing import cache_validation,evaluate
from src.models.relation_routing import build_routing_model
from src.models.order_verification import build_order_model


def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage3/protocol.json').read_text());_,val=datasets(p);batches=cache_validation(val)
    path=Path('experiments/stage2p6/new/TOHI-H1/seed_17/checkpoints/best.pt');weights=torch.load(path,map_location='cpu',weights_only=True)['model']
    old=build_order_model('TOHI-H1');old.load_state_dict(weights);new=build_routing_model('Receiver-Fix');new.load_state_dict(weights,strict=True)
    before=evaluate(old,batches,torch.device('cpu'));after=evaluate(new,batches,torch.device('cpu'),reference=before['predictions'])
    for a,b in zip(before['predictions'],after['predictions']):assert torch.isfinite(a).all() and torch.isfinite(b).all()
    full8_delta=0.
    for a,b,batch in zip(before['predictions'],after['predictions'],batches):
        mask=batch['groups']['history_full8'][:,None].expand(-1,12,-1)
        full8_delta=max(full8_delta,float((a-b).abs()[mask].max()) if mask.any() else 0.)
    print('full8 max abs numerical difference m:',full8_delta)
    # Different edge-row counts can select different float32 GEMM kernels.
    # Compare at sub-0.01mm tolerance, not bitwise identity for real batches.
    assert full8_delta<1e-5
    prior={r['group']:r for r in before['groups']}
    csv_new('results/tables/stage3/receiver_same_checkpoint_probe.csv',[dict(**r,delta_ADE_m=r['ADE_m']-prior[r['group']]['ADE_m'] if r['ADE_m'] is not None else None,
        delta_FDE_m=r['FDE_m']-prior[r['group']]['FDE_m'] if r['FDE_m'] is not None else None) for r in after['groups']])
    write_json_new('results/tables/stage3/receiver_same_checkpoint_provenance.json',dict(checkpoint=str(path),checkpoint_sha256=sha256(path),new_weights_trained=False,
        full8_prediction_max_abs_change_m=full8_delta,all_prediction_change_L2_m=after['diagnostics']['prediction_change_L2_m'],
        before_ADE_m=before['ADE_m'],after_ADE_m=after['ADE_m'],before_FDE_m=before['FDE_m'],after_FDE_m=after['FDE_m'],
        independent_training=False,full8_numerical_tolerance_m=1e-5,initial_bitwise_probe_assertion_passed=False,
        interpretation='Receiver-policy frozen old-model intervention; new-entry inputs OOD; full8 prediction unchanged within reported float32 tolerance, trained full8 degradation reflects shared optimization/checkpoint selection rather than material direct receiver inference effect',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),test_evaluated=False))


if __name__=='__main__':main()
