"""A0-preserving variants and explicit evaluation interventions."""
import torch
from .tohi_a import TOHIA
from .social_baselines import DynamicGNN, FirstOrderGNN
from .ped_base import PedBase
from .order_fusion import OrderFusion
from .social_fusion_variants import ControlledSocialFusion
from .temporal_graph import GraphFrame, aggregate, relation_geometry, target_softmax


def intervene_graphs(model, encoded, frames, intervention):
    if not intervention:
        return frames
    if intervention.get('remove_messages'):
        b, _, n, _ = encoded['hidden'].shape
        return [GraphFrame.artificial(b, n, [], encoded['hidden'].device, encoded['hidden'].dtype) for _ in frames]
    if not intervention.get('shuffle_graph'):
        return frames
    b, _, n, _ = encoded['hidden'].shape
    out = []
    for t, original in enumerate(frames):
        r, w, distance, tau, closest = relation_geometry(encoded['positions'][:, t], encoded['velocity'][:, t])
        sources = original.source.clone()
        for scene in range(b):
            eligible = encoded['continuity'][scene, t].nonzero().flatten().cpu()
            for target in range(n):
                edge_ids = (original.target == scene*n+target).nonzero().flatten()
                degree = len(edge_ids)
                if degree:
                    choices = eligible[(eligible != target) & (distance[scene, target, eligible.to(distance.device)].cpu() <= model.graph.radius)]
                    generator = torch.Generator().manual_seed(int(intervention.get('shuffle_seed', 17025))+t*10007+scene*7919+target)
                    order = torch.randperm(len(choices), generator=generator)[:degree]
                    sources[edge_ids] = (scene*n+choices[order]).to(sources.device)
        # Incoming degree, valid endpoints, self exclusion and scene budget are
        # unchanged. Recompute geometry/alpha for the sampled negative edges.
        scene = original.target//n
        target, source = original.target % n, sources % n
        features = torch.cat([r[scene, target, source], w[scene, target, source],
                              tau[scene, target, source, None], closest[scene, target, source, None],
                              torch.ones(len(source), 1, device=sources.device)], -1)
        previous = encoded['hidden'][:, t-1] if t else torch.zeros_like(encoded['hidden'][:, 0])
        scores = model.graph.relation_score(torch.cat([features, encoded['hidden'][scene, t, target], previous[scene, source]], -1)).squeeze(-1)
        weight = target_softmax(scores, original.target, b*n)
        totals = features.new_zeros(b*n, 7).index_add(0, original.target, features)
        degree = weight.new_zeros(b*n).index_add(0, original.target, torch.ones_like(weight))
        out.append(GraphFrame(sources, original.target, weight, features, (totals/degree[:, None].clamp_min(1)).reshape(b, n, 7)))
    return out


class TOHIDiagnostic(TOHIA):
    def __init__(self, hidden_dim=64, prediction_steps=12, max_order=3, max_neighbors=8, radius=4.,
                 residual=False, normalization=False, **kwargs):
        super().__init__(hidden_dim, prediction_steps, max_order, max_neighbors, radius)
        self.fusion = ControlledSocialFusion(self.fusion, hidden_dim, max_order, residual, normalization)

    def forward(self, observed, valid=None, timestamps=None, return_debug=False, intervention=None):
        encoded = self.encoder(observed, valid, timestamps)
        original = self.graph.sequence(encoded)
        frames = intervene_graphs(self, encoded, original, intervention)
        states, support = self.order_state(encoded['hidden'], frames, encoded['continuity'])
        context, beta, extra = self.fusion(encoded['hidden'][:, -1], states[:, -1], support[:, -1], intervention)
        prediction = self.decoder(context, encoded['positions'][:, -1], encoded['valid'][:, -1])
        if return_debug:
            return prediction, dict(**encoded, graph_frames=frames, original_graph_frames=original,
                                    order_states=states, order_support=support, order_weights=beta, **extra)
        return prediction


class TOHIResidual(TOHIDiagnostic):
    def __init__(self, **kwargs):
        super().__init__(residual=True, **kwargs)


class TOHINormalized(TOHIDiagnostic):
    def __init__(self, **kwargs):
        super().__init__(normalization=True, **kwargs)


class TOHIResidualNormalized(TOHIDiagnostic):
    def __init__(self, **kwargs):
        super().__init__(residual=True, normalization=True, **kwargs)


class FirstOrderResidualNormalized(TOHIResidualNormalized):
    def __init__(self, **kwargs):
        kwargs['max_order'] = 1
        super().__init__(**kwargs)


class DynamicResidualNormalized(DynamicGNN):
    def __init__(self, hidden_dim=64, prediction_steps=12, max_neighbors=8, radius=4., **kwargs):
        super().__init__(hidden_dim, prediction_steps, max_neighbors, radius)
        del self.mix
        self.fusion = ControlledSocialFusion(OrderFusion(hidden_dim, 1), hidden_dim, 1, residual=True, normalization=True)

    def forward(self, observed, valid=None, timestamps=None, return_debug=False, intervention=None):
        encoded = self.encoder(observed, valid, timestamps)
        hidden = encoded['hidden']
        b, t_steps, n, d = hidden.shape
        original = self.graph.sequence(encoded)
        frames = intervene_graphs(self, encoded, original, intervention)
        state = hidden.new_zeros(b, n, d)
        has_social = torch.zeros(b, n, device=hidden.device, dtype=torch.bool)
        states, supports = [], []
        for t, frame in enumerate(frames):
            previous = state*encoded['continuity'][:, t, :, None]
            message = aggregate(frame, self.projection(previous))
            state = self.social_gru(torch.cat([hidden[:, t], message], -1).reshape(b*n, -1), previous.reshape(b*n, -1)).reshape(b, n, d)
            state = state*encoded['valid'][:, t, :, None]
            degree = torch.zeros(b*n, dtype=torch.long, device=hidden.device).index_add(0, frame.target, torch.ones_like(frame.target)).reshape(b, n)
            has_social = (has_social & encoded['continuity'][:, t]) | (degree > 0)
            states.append(state)
            supports.append(has_social)
        context, beta, extra = self.fusion(hidden[:, -1], state[:, :, None], has_social[:, :, None], intervention)
        prediction = self.decoder(context, encoded['positions'][:, -1], encoded['valid'][:, -1])
        if return_debug:
            return prediction, dict(**encoded, graph_frames=frames, original_graph_frames=original,
                order_states=torch.stack(states, 1)[:, :, :, None], order_support=torch.stack(supports, 1)[:, :, :, None],
                order_weights=beta, **extra)
        return prediction


class PedBaseWide(PedBase):
    def __init__(self, **kwargs):
        kwargs['hidden_dim'] = 120
        super().__init__(**kwargs)


DIAGNOSTIC_MODELS = {
    'PedBase': PedBase, 'PedBase-Wide': PedBaseWide,
    'FirstOrder-GNN': FirstOrderGNN,
    'Dynamic-GNN': DynamicGNN, 'TOHI-A0': TOHIDiagnostic,
    'TOHI-A-R': TOHIResidual, 'TOHI-A-N': TOHINormalized,
    'TOHI-A-RN': TOHIResidualNormalized, 'FirstOrder-GNN-RN': FirstOrderResidualNormalized,
    'Dynamic-GNN-RN': DynamicResidualNormalized,
}
