"""Factories only: reuse frozen Full and frozen Stage5.4 Placebo unchanged."""
from src.socialvae_attribution.model import OfficialSocialVAE
from src.socialvae_capacity.placebo import SocialVAEPlacebo

def new_model(kind,device='cuda'):
    if kind not in ['Full','PlaceboSocial']:raise ValueError(kind)
    cls=OfficialSocialVAE if kind=='Full' else SocialVAEPlacebo
    return cls(12,2,256).to(device)
