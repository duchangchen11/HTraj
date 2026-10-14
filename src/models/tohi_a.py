from .ped_base import PedBase
from .temporal_graph import TemporalGraph
from .order_state import OrderState
from .order_fusion import OrderFusion


class TOHIA(PedBase):
    def __init__(self, hidden_dim=64, prediction_steps=12, max_order=3, max_neighbors=8, radius=4.0, **kwargs):
        super().__init__(hidden_dim, prediction_steps)
        self.graph = TemporalGraph(hidden_dim, radius, max_neighbors)
        self.order_state = OrderState(hidden_dim, max_order)
        self.fusion = OrderFusion(hidden_dim, max_order)

    def forward(self, observed, valid=None, timestamps=None, return_debug=False):
        encoded = self.encoder(observed, valid, timestamps)
        frames = self.graph.sequence(encoded)
        states, support = self.order_state(encoded['hidden'], frames, encoded['continuity'])
        context, beta = self.fusion(encoded['hidden'][:, -1], states[:, -1], support[:, -1])
        prediction = self.decoder(context, encoded['positions'][:, -1], encoded['valid'][:, -1])
        if return_debug:
            return prediction, dict(**encoded, graph_frames=frames, order_states=states, order_support=support, order_weights=beta)
        return prediction
