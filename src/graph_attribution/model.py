"""One explicit graph step and shared per-person temporal decoder."""
import torch
from torch import nn


class PersonTemporalNorm(nn.Module):
    def __init__(self,channels,eps=1e-5):
        super().__init__();self.eps=eps
        self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))

    def forward(self,x):
        mean=x.mean(2,keepdim=True);variance=x.var(2,unbiased=False,keepdim=True)
        return (x-mean)*torch.rsqrt(variance+self.eps)*self.weight[None,:,None,None]+self.bias[None,:,None,None]


class SharedTemporalDecoder(nn.Module):
    def __init__(self,width=16):
        super().__init__();self.time_map=nn.Linear(8,12)
        self.temporal=nn.Conv2d(width,width,(3,1),padding=(1,0))
        self.norm=PersonTemporalNorm(width);self.activation=nn.PReLU(width)
        self.output=nn.Conv2d(width,5,1)

    def forward(self,h):
        z=self.time_map(h.permute(0,1,3,2)).permute(0,1,3,2)
        z=z+self.activation(self.norm(self.temporal(z)))
        return self.output(z).permute(0,2,3,1) # B,12,N,5


class GraphAttributionModel(nn.Module):
    def __init__(self,kind,width=16,gamma=.25):
        super().__init__();assert kind in ('G0','G1','G2')
        self.kind=kind;self.gamma=gamma
        self.ego=nn.Conv2d(2,width,1)
        self.motion=nn.Conv2d(2,width,(3,1),padding=(1,0))
        self.motion_norm=PersonTemporalNorm(width);self.motion_activation=nn.PReLU(width)
        self.shared_projection=nn.Conv2d(width,width,1)
        self.post_norm=PersonTemporalNorm(width);self.post_activation=nn.PReLU(width)
        self.decoder=SharedTemporalDecoder(width)

    def forward(self,v,relation,valid,return_state=False):
        mask=valid[:,None,None,:].to(v.dtype)
        h=(self.ego(v)+self.motion_activation(self.motion_norm(self.motion(v))))*mask
        r=relation if self.kind!='G0' else torch.zeros_like(relation)
        neighbor=torch.einsum('btij,bctj->bcti',r,h)
        correction=self.gamma*neighbor
        mixed=h+correction # self coefficient exactly1, no joint I+A degree normalization
        z=(mixed+self.post_activation(self.post_norm(self.shared_projection(mixed))))*mask
        prediction=self.decoder(z)*valid[:,None,:,None]
        if return_state:
            return prediction,dict(ego_norm=h.norm(dim=1),neighbor_norm=neighbor.norm(dim=1),
                correction_norm=correction.norm(dim=1),mixed_norm=mixed.norm(dim=1),
                relative_correction=correction.norm(dim=1)/(h.norm(dim=1)+1e-8))
        return prediction


def window_nll(parameters,target,valid):
    """Vectorized unchanged official PDF-clamped Gaussian NLL, mean per window."""
    dx=target[...,0]-parameters[...,0];dy=target[...,1]-parameters[...,1]
    sx=parameters[...,2].exp();sy=parameters[...,3].exp();rho=parameters[...,4].tanh()
    product=sx*sy;z=(dx/sx)**2+(dy/sy)**2-2*rho*dx*dy/product;remaining=1-rho**2
    density=torch.exp(-z/(2*remaining))/(2*torch.pi*product*torch.sqrt(remaining))
    point=-torch.log(torch.clamp(density,min=1e-20))
    return (point*valid[:,None,:]).sum((1,2))/(12*valid.sum(1))


def gaussian_predictions(parameters,last,noise):
    mean=parameters[...,:2];sx=parameters[...,2].exp();sy=parameters[...,3].exp();rho=parameters[...,4].tanh()
    if not torch.isfinite(parameters).all() or not (1-rho.square()>0).all():raise FloatingPointError('Invalid Gaussian')
    e1,e2=noise[...,0],noise[...,1]
    delta=torch.stack([sx[:,None]*e1,sy[:,None]*(rho[:,None]*e1+(1-rho.square()).sqrt()[:,None]*e2)],-1)
    samples=last[:,None,None]+(mean[:,None]+delta).cumsum(2)
    return last[:,None]+mean.cumsum(1),samples
