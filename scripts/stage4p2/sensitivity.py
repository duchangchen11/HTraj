"""No-GT numerical sensitivity diagnostic; never selects or retrains a model."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import torch
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel,gaussian_predictions,PersonTemporalNorm
from src.graph_attribution.data import real_relation


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    root=Path('results/tables/stage4p2');config=json.loads(Path('configs/stage4p2/protocol.json').read_text());cache=torch.load(config['cache'],weights_only=False)
    rows=[]
    for fold in config['folds']:
        samples=[cache['records'][r][i] for r,i in cache['folds'][fold]['validation']]
        s=next(s for s in samples if s['relation'].any());receiver,source=(s['relation'].sum(0)>0).nonzero()[0].tolist()
        for kind in ('G0','G1'):
            summary=json.loads((root/fold/kind/'deterministic_scan/summary.json').read_text());model=GraphAttributionModel(kind).cuda().eval()
            model.load_state_dict(torch.load(summary['best']['checkpoint'],weights_only=False)['model'])
            moments={}
            def capture(name):
                def hook(module,args):
                    if name not in moments:moments[name]=args[0].detach().var(2,unbiased=False)
                return hook
            for name,module in model.named_modules():
                if isinstance(module,PersonTemporalNorm):module.register_forward_pre_hook(capture(name))
            def predict(observed):
                n=observed.shape[1];v=torch.cat([torch.zeros_like(observed[:1]),observed[1:]-observed[:-1]],0).permute(2,0,1)[None]
                relation=real_relation(observed)[None];valid=torch.ones(1,n,device='cuda',dtype=torch.bool)
                parameters=model(v,relation,valid)
                mean,_=gaussian_predictions(parameters,observed[-1][None],torch.zeros(1,20,12,n,2,device='cuda'))
                return mean,parameters,relation
            observed=s['observed'].cuda().requires_grad_();mean,parameters,relation=predict(observed)
            gradient=torch.autograd.grad(mean[0,-1,receiver,0],observed)[0][:,source]
            maximum=float(gradient.abs().max());index=int(gradient.abs().argmax());t,coordinate=divmod(index,2)
            sigma=parameters[...,2:4].exp();remaining=1-parameters[...,4].tanh().square()
            assert torch.isfinite(sigma).all() and sigma.min()>0 and remaining.min()>0
            for epsilon in (1e-5,1e-4,1e-3):
                changed=observed.detach().clone();changed[t,source,coordinate]+=epsilon
                with torch.no_grad():altered,_,edges=predict(changed)
                difference=(altered[0,:,receiver]-mean[0,:,receiver]).norm(dim=-1)
                row=dict(fold=fold,model=kind,window_id=s['window_id'],receiver=receiver,source=source,history_frame=t,coordinate=coordinate,
                    epsilon_m=epsilon,max_receiver_final_x_jacobian_wrt_source_position=maximum,
                    receiver_mean_prediction_shift_m=float(difference.mean()),receiver_final_prediction_shift_m=float(difference[-1]),
                    binary_edge_changes=int(((relation>0)!=(edges>0)).sum()),
                    sigma_min_step_m=float(sigma.min()),sigma_max_step_m=float(sigma.max()),one_minus_rho_square_min=float(remaining.min()),future_used=False)
                for name,variance in moments.items():
                    for role,node in (('sender',source),('receiver',receiver)):
                        key=name.replace('.','_')+'_'+role
                        row[key+'_variance_min']=float(variance[0,:,node].min())
                        row[key+'_fraction_variance_below_eps']=float((variance[0,:,node]<1e-5).float().mean())
                if kind=='G0':assert maximum==0 and float(difference.max())==0
                rows.append(row)
    csv_new(root/'history_position_sensitivity_with_norm.csv',rows)
    print(rows)


if __name__=='__main__':main()
