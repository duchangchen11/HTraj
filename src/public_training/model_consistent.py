"""One bounded correction: NoSocial normalization agrees in train and eval.

Original Stage4.1 NoSocial and its failed checkpoints remain available unchanged.
No network/loss/optimizer changes to Full. Not a new prediction method.
"""
import torch
from torch import nn
from src.public_training.model import make_model as original_model,NodewiseBatchNorm2d


class ConsistentTemporalNorm(nn.Module):
    def __init__(self,original):
        super().__init__();self.eps=original.eps
        self.weight=nn.Parameter(original.weight.detach().clone());self.bias=nn.Parameter(original.bias.detach().clone())

    def forward(self,x):
        # Exactly the legacy NoSocial train-mode transform, in both modes.
        mean=x.mean(2,keepdim=True);var=x.var(2,unbiased=False,keepdim=True)
        return (x-mean)*torch.rsqrt(var+self.eps)*self.weight[None,:,None,None]+self.bias[None,:,None,None]


def make_model(kind):
    assert kind=='NoSocial-ConsistentNorm'
    model=original_model('NoSocial')
    def replace(parent):
        for name,module in list(parent.named_children()):
            if isinstance(module,NodewiseBatchNorm2d):setattr(parent,name,ConsistentTemporalNorm(module))
            else:replace(module)
    replace(model)
    return model
