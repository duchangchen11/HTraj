"""Strict target-only encoder; inherit frozen prior/posterior/decoder/loss.

Retain all registered weights for initialization/state-schema comparability.
Unused weights and zero-input GRU columns are NOT counted as effective capacity.
No neighbor value, mask, count, identity, order or entry time is read by enc().
"""
import torch
from src.socialvae_causal.official import load_official

OfficialSocialVAE=load_official().SocialVAE


class SocialVAENoSocial(OfficialSocialVAE):
    def enc(self,x,neighbor,*,y=None):
        with torch.no_grad():
            L1=x.size(0)-1;N=x.size(1);state=x
            positions=state[...,:2]
            if y is not None:
                L2=y.size(0)
                positions=torch.cat((positions,y),0)
            v=positions[1:]-positions[:-1]
            a=v[1:]-v[:-1]
            a=torch.cat((state[1:2,...,4:6],a))
        s=self.embed_s(torch.cat((v,a),-1))
        h=x.new_zeros(self.rnn_fx.num_layers,N,self.rnn_fx.hidden_size)
        # Original concatenation order is [social128, ego128]. Never multiply
        # neighbor tensors by zero: even NaN/Inf neighbor inputs are irrelevant.
        zero=x.new_zeros(N,self.rnn_fx.input_size-s.size(-1))
        for t in range(L1):
            x_t=torch.cat((zero,s[t]),-1).unsqueeze(0)
            _,h=self.rnn_fx(x_t,h)
        encoded=h[-1]
        if y is None:return encoded
        s_t=s[L1:L2+L2]  # exactly the frozen 8/12 encoder slice
        zero_future=x.new_zeros(L2,N,self.rnn_by.input_size-s_t.size(-1))
        x_t=torch.cat((zero_future,s_t),-1)
        b,_=self.rnn_by(torch.flip(x_t,(0,)))
        if self.rnn_by.num_layers>1:b=b[...,-b.size(-1)//self.rnn_by.num_layers:]
        return encoded,torch.flip(b,(0,))


def new_model(hidden_dim=256,device='cuda'):
    return SocialVAENoSocial(12,2,hidden_dim).to(device)


DISCONNECTED_PREFIXES=('rnn_fx_init.','embed_n.','embed_k.','embed_q.')


def structural_parameter_masks(model,variant):
    """Structural upper bound on effective scalar parameters, not a rank claim.

    Full on causal protocol: posterior future social inputs are always zero.
    NoSocial: both history and posterior first128 GRU columns are always zero.
    """
    masks={name:torch.ones_like(p,dtype=torch.bool,device='cpu') for name,p in model.named_parameters()}
    if variant=='NoSocial':
        for name,mask in masks.items():
            if name.startswith(DISCONNECTED_PREFIXES):mask.zero_()
        masks['rnn_fx.weight_ih_l0'][:,:128]=False
    elif variant!='Full':raise ValueError(variant)
    masks['rnn_by.weight_ih_l0'][:,:128]=False
    return masks
