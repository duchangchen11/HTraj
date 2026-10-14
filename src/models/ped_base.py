from torch import nn
from .ego_encoder import EgoEncoder
from .trajectory_decoder import TrajectoryDecoder


class PedBase(nn.Module):
    def __init__(self, hidden_dim=64, prediction_steps=12, **unused_graph_options):
        super().__init__()
        self.encoder = EgoEncoder(hidden_dim)
        self.decoder = TrajectoryDecoder(hidden_dim, prediction_steps)

    def forward(self, observed, valid=None, timestamps=None, return_debug=False):
        encoded = self.encoder(observed, valid, timestamps)
        prediction = self.decoder(encoded['hidden'][:, -1], encoded['positions'][:, -1], encoded['valid'][:, -1])
        return (prediction, encoded) if return_debug else prediction
