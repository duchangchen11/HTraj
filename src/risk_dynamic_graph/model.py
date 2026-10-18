"""Shared causal GRU + one-hop mean messages + deterministic decoder.

No recursive social hidden state, no attention, normalization, or learned gate.
All classes keep identical nominal parameters. E has a disclosed inactive path.
"""
import torch
from torch import nn
from .graph_builder import build_graph, GraphConfig
from .interaction_features import historical_motion


class RiskGraphModel(nn.Module):
    def __init__(self, mode='E', hidden_dim=64, config=None):
        super().__init__()
        if mode not in ('E', 'S', 'D', 'R'):
            raise ValueError('Unknown mode')
        self.mode, self.config, self.hidden_dim = mode, config or GraphConfig(), hidden_dim
        self.encoder = nn.GRUCell(3, hidden_dim)
        self.message = nn.Sequential(nn.Linear(hidden_dim + 9, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.decoder = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 24))

    def forward(self, position, valid, ids, diagnostics=False):
        p, velocity, vv = historical_motion(position, valid, self.config.dt)
        graph = build_graph(position, valid, ids, self.mode, self.config)
        b, t, n, _ = p.shape
        h = p.new_zeros(b, n, self.hidden_dim)
        total = torch.zeros_like(h)
        count = p.new_zeros(b, n, 1)
        state_trace = []
        for step in range(t):
            continuous = vv[:, step, :, None]
            # New arrivals never inherit stale carry. Unknown velocity is 0
            # placeholder with an explicit validity bit, not measured zero.
            inp = torch.cat([velocity[:, step], vv[:, step, :, None].to(p.dtype)], -1)
            proposed = self.encoder(inp.reshape(b * n, 3), torch.where(continuous, h, 0.).reshape(b * n, -1))
            h = torch.where(valid[:, step, :, None], proposed.reshape(b, n, -1), 0.)
            state_trace.append(h)
            if self.mode != 'E':
                bi, target, source = graph['edge_mask'][:, step].nonzero(as_tuple=True)
                data = torch.cat([h[bi, source], graph['features'][bi, step, target, source]], -1)
                msg = self.message(data)
                sums = p.new_zeros(b * n, self.hidden_dim).index_add(0, bi * n + target, msg).reshape(b, n, -1)
                degree = p.new_zeros(b * n).index_add(0, bi * n + target, torch.ones_like(bi, dtype=p.dtype)).reshape(b, n, 1)
                # Reset accumulated context when receiver disappears/re-enters.
                total = torch.where(continuous, total, 0.) + sums / degree.clamp_min(1.)
                count = torch.where(continuous, count, 0.) + (degree > 0).to(p.dtype)
        social = total / count.clamp_min(1.)
        offset = self.decoder(torch.cat([h, social], -1)).reshape(b, n, 12, 2)
        prediction = p[:, -1, :, None] + offset.cumsum(-2)
        prediction = torch.where(valid[:, -1, :, None, None], prediction, 0.)
        if diagnostics:
            return dict(prediction=prediction, graph=graph, social=social,
                        ego_trace=torch.stack(state_trace, 1))
        return prediction
