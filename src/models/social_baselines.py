import torch
from torch import nn
from .tohi_a import TOHIA
from .ped_base import PedBase
from .temporal_graph import TemporalGraph, aggregate


class FirstOrderGNN(TOHIA):
    """A single social channel receives only ego features, never social states."""
    def __init__(self, hidden_dim=64, prediction_steps=12, max_neighbors=8, radius=4.0, **kwargs):
        super().__init__(hidden_dim, prediction_steps, max_order=1, max_neighbors=max_neighbors, radius=radius)


class DynamicGNN(PedBase):
    """Ungraded recurrent dynamic GNN. No separation by social edge count."""
    def __init__(self, hidden_dim=64, prediction_steps=12, max_neighbors=8, radius=4.0, **kwargs):
        super().__init__(hidden_dim, prediction_steps)
        self.graph = TemporalGraph(hidden_dim, radius, max_neighbors)
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.social_gru = nn.GRUCell(2*hidden_dim, hidden_dim)
        self.mix = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, observed, valid=None, timestamps=None, return_debug=False):
        encoded = self.encoder(observed, valid, timestamps)
        hidden = encoded['hidden']
        b, t_steps, n, d = hidden.shape
        frames = self.graph.sequence(encoded)
        state = hidden.new_zeros(b, n, d)
        has_social = torch.zeros(b, n, device=hidden.device, dtype=torch.bool)
        states = []
        for t, frame in enumerate(frames):
            previous = state*encoded['continuity'][:, t, :, None]
            message = aggregate(frame, self.projection(previous))
            state = self.social_gru(torch.cat([hidden[:, t], message], -1).reshape(b*n, -1),
                                    previous.reshape(b*n, -1)).reshape(b, n, d)
            state = state*encoded['valid'][:, t, :, None]
            degree = torch.zeros(b*n, dtype=torch.long, device=hidden.device).index_add(
                0, frame.target, torch.ones_like(frame.target)).reshape(b, n)
            has_social = (has_social & encoded['continuity'][:, t]) | (degree > 0)
            states.append(state)
        context = self.mix(torch.cat([hidden[:, -1], state], -1))
        context = torch.where(has_social[..., None], context, hidden[:, -1])
        prediction = self.decoder(context, encoded['positions'][:, -1], encoded['valid'][:, -1])
        return (prediction, dict(**encoded, social_states=torch.stack(states, 1), graph_frames=frames)) if return_debug else prediction
