"""Numerical structural evidence before training, including all requested sizes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import subprocess
import torch
from src.data.preprocessing import write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel
from src.graph_attribution.data import real_relation


def main():
    torch.set_num_threads(1);torch.manual_seed(17);rows=[];dependencies=[]
    for kind in ('G0','G1','G2'):
        for n in (2,3,5,8,12):
            model=GraphAttributionModel(kind);x=torch.randn(1,2,8,n,requires_grad=True);pos=torch.randn(8,n,2)
            a=real_relation(pos)[None];mask=torch.ones(1,n,dtype=torch.bool);p=torch.randperm(n);inv=p.argsort()
            y=model(x,a,mask);gx,*gp=torch.autograd.grad(y.square().sum(),[x,*model.parameters()])
            xx=x.detach()[...,p].requires_grad_();yy=model(xx,a[:,:,p][:,:,:,p],mask[:,p])
            gxx,*gpp=torch.autograd.grad(yy.square().sum(),[xx,*model.parameters()])
            row=dict(model=kind,people=n,parameter_count=sum(q.numel() for q in model.parameters()),
                max_distribution_difference=float((y-yy[:,:,inv]).abs().max()),
                max_position_difference_m=float((y[...,:2].cumsum(1)-yy[:,:,inv,:2].cumsum(1)).abs().max()),
                max_input_gradient_difference=float((gx-gxx[...,inv]).abs().max()),
                max_parameter_gradient_difference=max(float((first-second).abs().max()) for first,second in zip(gp,gpp)))
            assert row['max_distribution_difference']<1e-5 and row['max_position_difference_m']<5e-5
            assert row['max_input_gradient_difference']<1e-4 and row['max_parameter_gradient_difference']<1e-3
            rows.append(row)
        model=GraphAttributionModel(kind);x=torch.randn(1,2,8,3,requires_grad=True);a=torch.zeros(1,8,3,3);a[:,:,0,1]=1;mask=torch.ones(1,3,dtype=torch.bool)
        y=model(x,a,mask);g=torch.autograd.grad(y[:,:,0].sum(),x)[0];without=model(x,torch.zeros_like(a),mask)
        row=dict(model=kind,cross_source_jacobian_max=float(g[...,1].abs().max()),unconnected_source_jacobian_max=float(g[...,2].abs().max()),
            changed_graph_prediction_max=float((y-without).abs().max()))
        assert row['unconnected_source_jacobian_max']==0
        assert (row['cross_source_jacobian_max']==0 and row['changed_graph_prediction_max']==0) if kind=='G0' else (row['cross_source_jacobian_max']>1e-6 and row['changed_graph_prediction_max']>1e-6)
        dependencies.append(row)
    root=Path('results/tables/stage4p2');csv_new(root/'pretrain_permutation.csv',rows);csv_new(root/'pretrain_jacobian.csv',dependencies)
    write_json_new(root/'structure_execution.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),passed=True,
        sizes=[2,3,5,8,12],device='cpu',torch=str(torch.__version__),ETH_evaluated=False))


if __name__=='__main__':main()
