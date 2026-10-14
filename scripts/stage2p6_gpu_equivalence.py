"""GPU forward/backward equivalence audit; zero optimizer steps, not a training run."""
from pathlib import Path
import json
import subprocess
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import sha256,write_json_new
from src.training.order_verification import ROOT,PROTOCOL,datasets,cached_validation
from src.training.smoke import seed_all,move_batch
from src.evaluation.metrics import masked_l2_loss
from src.models.social_diagnostics_models import TOHIResidual
from src.models.order_verification import build_order_model


def main():
    torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text()); _,val=datasets(protocol)
    batch=move_batch(cached_validation(val,8)[0],torch.device('cuda'))
    path=Path('experiments/stage2p5/round2/TOHI-A-R/seed_17/checkpoints/best.pt')
    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
    seed_all(17); original=TOHIResidual(hidden_dim=64).cuda()
    seed_all(17); measured=build_order_model('TOHI-H3').cuda()
    cases=[]
    for mode in ['initial','stage2p5_best_weights_readonly']:
        if mode!='initial':
            original.load_state_dict(checkpoint['model']); measured.load_state_dict(checkpoint['model'])
        original.zero_grad(set_to_none=True); measured.zero_grad(set_to_none=True)
        a=original(batch['observed'],batch['observed_valid'],batch['timestamps'])
        b=measured(batch['observed'],batch['observed_valid'],batch['timestamps'])
        max_prediction=float((a-b).abs().max())
        masked_l2_loss(a,batch['future'],batch['future_valid']).backward()
        masked_l2_loss(b,batch['future'],batch['future_valid']).backward()
        maximum=0.; worst=''
        for (name,p),(other_name,q) in zip(original.named_parameters(),measured.named_parameters()):
            assert name==other_name and (p.grad is None)==(q.grad is None)
            if p.grad is not None:
                delta=float((p.grad-q.grad).abs().max())
                if delta>maximum:
                    maximum,worst=delta,name
        cases.append(dict(case=mode,max_prediction_absolute_difference_m=max_prediction,
                          max_parameter_gradient_absolute_difference=maximum,worst_gradient_parameter=worst))
        assert max_prediction<1e-5 and maximum<1e-5
    write_json_new(ROOT/'gpu_equivalence_audit.json',dict(status='passed',cases=cases,optimizer_steps=0,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runner_sha256=sha256(__file__),
        old_checkpoint_loaded_only_for_equivalence=True,old_checkpoint_sha256=sha256(path),
        actual_training_runs_untouched=True,data_manifest_sha256=sha256(protocol['data_manifest']),
        subset_manifest_sha256=sha256(protocol['subset_manifest']),test_evaluated=False))
    print(json.dumps(cases),flush=True)


if __name__=='__main__':
    main()
