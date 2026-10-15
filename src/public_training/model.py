"""Unmodified official Full and an explicit, capacity-matched NoSocial control."""
from pathlib import Path
import torch
from torch import nn
from src.evaluation.public_baseline import load_module

UPSTREAM=Path('third_party/stage4/social_stgcnn')
official=load_module('stage4p1_official_model',UPSTREAM/'model.py')
metrics=load_module('stage4p1_official_metrics',UPSTREAM/'metrics.py')


class NodewiseBatchNorm2d(nn.BatchNorm2d):
    """Shared affine/running stats, training moments separate for each (B,node).

    Test-time running moments depend on training data only. Unlike upstream BN,
    a person's training output never uses another person's current activations.
    """
    def forward(self,x):
        if not self.training:return super().forward(x)
        mean=x.mean(2,keepdim=True);var=x.var(2,unbiased=False,keepdim=True)
        if self.track_running_stats:
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                momentum=self.momentum if self.momentum is not None else 1./float(self.num_batches_tracked)
                self.running_mean.lerp_(mean.mean((0,2,3)),momentum)
                self.running_var.lerp_((var*x.shape[2]/(x.shape[2]-1)).mean((0,2,3)),momentum)
        y=(x-mean)*torch.rsqrt(var+self.eps)
        if self.affine:y=y*self.weight[None,:,None,None]+self.bias[None,:,None,None]
        return y


def make_model(kind):
    assert kind in ('Full','NoSocial')
    model=official.social_stgcnn(n_stgcnn=1,n_txpcnn=5,input_feat=2,output_feat=5,seq_len=8,pred_seq_len=12,kernel_size=3)
    if kind=='NoSocial':
        # Same nine coefficients per input/output channel, all confined to one node.
        # TXP-CNN height is its existing feature axis, NOT a new motion decoder.
        for parent in (model.tpcnns,):
            for i,conv in enumerate(parent):parent[i]=node_only_conv(conv)
        model.tpcnn_ouput=node_only_conv(model.tpcnn_ouput)
        def replace_bn(parent):
            for name,module in list(parent.named_children()):
                if isinstance(module,nn.BatchNorm2d):
                    replacement=NodewiseBatchNorm2d(module.num_features,eps=module.eps,momentum=module.momentum,affine=module.affine)
                    replacement.load_state_dict(module.state_dict());setattr(parent,name,replacement)
                else:replace_bn(module)
        replace_bn(model)
    model.stage4p1_kind=kind
    return model


def node_only_conv(conv):
    assert conv.kernel_size==(3,3) and conv.groups==1
    replacement=nn.Conv2d(conv.in_channels,conv.out_channels,(9,1),padding=(4,0),bias=conv.bias is not None)
    with torch.no_grad():
        replacement.weight.copy_(conv.weight.reshape_as(replacement.weight))
        if conv.bias is not None:replacement.bias.copy_(conv.bias)
    return replacement


def forward(model,v,a):
    if model.stage4p1_kind=='NoSocial':a=torch.eye(v.shape[-1],device=v.device,dtype=v.dtype)[None].expand(8,-1,-1)
    raw,_=model(v,a)
    return raw.permute(0,2,3,1)[0]  # [12,N,5], no squeeze ambiguity for N=1.


def nll(prediction,target,mask=None):
    if mask is not None:prediction,target=prediction[:,mask],target[:,mask]
    return metrics.bivariate_loss(prediction,target)


def gaussian_trajectories(parameters,last,noise):
    mean=parameters[...,:2];sx=parameters[...,2].exp();sy=parameters[...,3].exp();rho=parameters[...,4].tanh()
    if not torch.isfinite(parameters).all() or not (1-rho.square()>0).all():raise FloatingPointError('Nonfinite/non-positive Gaussian')
    step=torch.stack([sx*noise[...,0],sy*(rho*noise[...,0]+torch.sqrt(1-rho.square())*noise[...,1])],-1)+mean
    samples=last[None,None]+step.cumsum(1)
    return last[None]+mean.cumsum(0),samples
