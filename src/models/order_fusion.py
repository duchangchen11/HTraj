import torch
from torch import nn


class OrderFusion(nn.Module):
    def __init__(self, hidden_dim=64, max_order=3):
        super().__init__()
        self.scores = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh(),
                                                 nn.Linear(hidden_dim, 1)) for _ in range(max_order)])
        self.mix = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, ego, states, support):
        scores = torch.cat([scorer(torch.cat([ego, states[:, :, q]], -1))
                            for q, scorer in enumerate(self.scores)], -1)
        available = support.any(-1)
        scores = scores.masked_fill(~support, -torch.inf)
        scores = torch.where(available[..., None], scores, torch.zeros_like(scores))
        beta = torch.softmax(scores, -1)*support.to(scores.dtype)
        pooled = (states*beta[..., None]).sum(2)
        context = self.mix(torch.cat([ego, pooled], -1))
        # Exact ego-only fallback; a bias cannot manufacture absent social order.
        return torch.where(available[..., None], context, ego), beta
