"""Corrected, source-saved posthoc audit of the failed NoSocial normalization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy,json,torch
from src.public_training.model import make_model,forward,NodewiseBatchNorm2d,nll
from src.public_training.data import network_input
from src.public_training.engine import truth_velocity
from src.data.preprocessing import write_json_new,sha256


def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage4p1/protocol.json').read_text())
    checkpoint='experiments/stage4p1/NoSocial/epoch004.pt';before_sha=sha256(checkpoint)
    base=make_model('NoSocial');base.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    data=torch.load(p['cache'],weights_only=True)['validation'][:64];baseline=None;rows=[];delta=None
    for policy in ('stored_running_statistics','same_per_node_statistics_as_training'):
        model=copy.deepcopy(base).eval()
        if policy.startswith('same'):
            for module in model.modules():
                if isinstance(module,NodewiseBatchNorm2d):module.train()
        rho=0.;log_sigma=0.;losses=[];predictions=[]
        with torch.no_grad():
            for sample in data:
                v,a=network_input(sample);result=forward(model,v,a);predictions.append(result)
                rho=max(rho,float(result[...,4].abs().max()));log_sigma=max(log_sigma,float(result[...,2:4].abs().max()))
                losses.append(float(nll(result,truth_velocity(sample))))
        rows.append(dict(policy=policy,windows=64,max_absolute_correlation_logit=rho,max_absolute_log_sigma=log_sigma,
            finite_NLL_windows=int(torch.isfinite(torch.tensor(losses)).sum())))
        if baseline is None:baseline=predictions
        else:delta=max(float((a-b).abs().max()) for a,b in zip(baseline,predictions))
    assert before_sha==sha256(checkpoint)
    write_json_new('results/tables/stage4p1/nosocial_failure_diagnosis_v2.json',dict(checkpoint=checkpoint,checkpoint_sha256=before_sha,
        policies=rows,max_gaussian_parameter_shift=delta,weights_updated=False,checkpoint_unchanged=True,
        correction='supersedes max_gaussian_parameter_shift=0 in initial diagnosis: inline audit reset baseline inside loop; policy maxima were valid, zero shift field was invalid and must not be used',
        inference='supports train/eval normalization amplification in epoch4; epoch5 failed state not saved, exact offending value unavailable',
        scope='posthoc fixed weights, first64 windows; no performance model selection',script_sha256=sha256(__file__),ETH_evaluated=False))
    print(rows,delta)


if __name__=='__main__':main()
