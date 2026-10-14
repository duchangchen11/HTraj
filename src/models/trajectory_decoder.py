import torch
from torch import nn


class TrajectoryDecoder(nn.Module):
    """One shared deterministic direct-displacement head for all models."""
    def __init__(self, hidden_dim=64, prediction_steps=12):
        super().__init__()
        self.prediction_steps = prediction_steps
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, prediction_steps*2))

    def forward(self, context, anchor, target_valid):
        b, n, _ = context.shape
        offset = self.head(context).reshape(b, n, self.prediction_steps, 2).permute(0, 2, 1, 3)
        prediction = anchor[:, None]+offset
        return torch.where(target_valid[:, None, :, None], prediction, torch.zeros_like(prediction))
