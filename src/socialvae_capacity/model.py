"""Target-only residual capacity; no arbitrary sparsity masks or new loss."""
import torch
from src.socialvae_attribution.model import SocialVAENoSocial,structural_parameter_masks as no_social_masks


class SocialVAEEgoCapacity(SocialVAENoSocial):
    def __init__(self,horizon=12,ob_radius=2,hidden_dim=256):
        super().__init__(horizon,ob_radius,hidden_dim)
        self.ego_capacity=torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,4*hidden_dim),torch.nn.ReLU6(),
            torch.nn.Linear(4*hidden_dim,hidden_dim))
        self.control_kind='EgoCapacity'

    def enc(self,x,neighbor,*,y=None):
        result=super().enc(x,neighbor,y=y)
        h=result if y is None else result[0]
        corrected=h+self.ego_capacity(h)
        return corrected if y is None else (corrected,result[1])


def new_model(kind='EgoCapacity',device='cuda',hidden_dim=256):
    if kind!='EgoCapacity':raise ValueError('Only preregistered phase-one model is implemented')
    return SocialVAEEgoCapacity(12,2,hidden_dim).to(device)


def structural_parameter_masks(model):
    # Analytical audit masks only. They are never applied to model weights.
    return no_social_masks(model,'NoSocial')
