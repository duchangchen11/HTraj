"""Frozen Full network with one preregistered target-only synthetic context.

No learned modules added/removed. Real neighbor packet is never consulted.
Offsets form four cardinal points; bounded target-prefix displacement adds
opposite deterministic drifts. No validation-error or recording dependence.
"""
import torch
from src.socialvae_attribution.model import OfficialSocialVAE,structural_parameter_masks as official_masks


def synthetic_context(x,training=False):
    assert x.shape[0]==8 and x.shape[-1]==6
    with torch.no_grad():
        position=x[...,:2]
        offsets=x.new_tensor([[.8,0.],[-.8,0.],[0.,.8],[0.,-.8]])
        signs=x.new_tensor([1.,-1.,1.,-1.])
        drift=.2*torch.tanh((position-position[:1])/1.)
        p=position.unsqueeze(2)+offsets+drift.unsqueeze(2)*signs[None,None,:,None]
        result=x.new_full((20 if training else 8,x.shape[1],4,6),1e9)
        result[:8,:,:,:2]=p
        # Boundary velocity/acceleration are explicit unknown-boundary zero
        # placeholders; official initialization reads positions only at t=0.
        result[:8,:,:,2:]=0.
        velocity=p[1:]-p[:-1]
        result[1:8,:,:,2:4]=velocity
        result[2:8,:,:,4:6]=velocity[1:]-velocity[:-1]
        return result


class SocialVAEPlacebo(OfficialSocialVAE):
    def enc(self,x,neighbor,*,y=None):
        return super().enc(x,synthetic_context(x,training=y is not None),y=y)


def new_model(device='cuda',hidden_dim=256):
    return SocialVAEPlacebo(12,2,hidden_dim).to(device)


def structural_parameter_masks(model):
    # Same causal future-social zero columns as Full, not weight masking.
    return official_masks(model,'Full')
