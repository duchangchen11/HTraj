"""Exact A1 recurrence, double-buffered social orders and Boolean support."""
import torch
from torch import nn
from .temporal_graph import aggregate


class OrderState(nn.Module):
    def __init__(self, hidden_dim=64, max_order=3):
        super().__init__()
        if max_order < 1:
            raise ValueError('Need at least one social order')
        self.max_order, self.hidden_dim = max_order, hidden_dim
        self.projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(max_order)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim+7, hidden_dim), nn.ReLU(),
                                                nn.Linear(hidden_dim, hidden_dim)) for _ in range(max_order)])

    def step(self, old, old_support, previous_ego, current_ego, frame, continuity):
        b, n, _, d = old.shape
        # Reset carry on absence/reappearance/gap. All orders read this immutable
        # previous snapshot; never read a just-written lower-order state.
        previous = old*continuity[:, :, None, None]
        support = old_support & continuity[:, :, None]
        gate_input = torch.cat([current_ego, frame.relation_mean], -1)
        new_states, new_support = [], []
        for q in range(self.max_order):
            source_state = previous_ego if q == 0 else previous[:, :, q-1]
            source_valid = continuity if q == 0 else support[:, :, q-1]
            message = aggregate(frame, self.projections[q](source_state))
            # Support ignores learned magnitude: an edge can exist with a zero
            # projected message. This is distinct from "message != 0".
            received = torch.zeros(b*n, device=old.device, dtype=torch.long).index_add(
                0, frame.target, source_valid.reshape(-1)[frame.source].long()).reshape(b, n) > 0
            valid_q = (support[:, :, q] | received) & continuity
            gamma = torch.sigmoid(self.gates[q](gate_input))
            state = gamma*previous[:, :, q]+message
            new_states.append(torch.where(valid_q[..., None], state, torch.zeros_like(state)))
            new_support.append(valid_q)
        return torch.stack(new_states, 2), torch.stack(new_support, 2)

    def forward(self, hidden, frames, continuity):
        b, t_steps, n, d = hidden.shape
        if len(frames) != t_steps:
            raise ValueError('Graph sequence length mismatch')
        state = hidden.new_zeros(b, n, self.max_order, d)
        support = torch.zeros(b, n, self.max_order, dtype=torch.bool, device=hidden.device)
        states, supports = [], []
        for t, frame in enumerate(frames):
            previous_ego = hidden[:, t-1] if t else torch.zeros_like(hidden[:, 0])
            state, support = self.step(state, support, previous_ego, hidden[:, t], frame, continuity[:, t])
            states.append(state)
            supports.append(support)
        return torch.stack(states, 1), torch.stack(supports, 1)
