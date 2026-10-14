"""Independent order truncation and one controlled relaxation of edge time.

Original Stage2/2.5 modules are never edited. Strict mode exactly reproduces
A-R. Relaxed mode adds only newly received, same-time lower-order messages to
the previous lower-order source; ego inputs and local carry stay unchanged.
"""
import torch
from torch import nn
from .order_state import OrderState
from .temporal_graph import aggregate
from .social_diagnostics_models import TOHIResidual


class MeasuredOrderState(OrderState):
    def __init__(self, original, max_order, mode='strict'):
        nn.Module.__init__(self)
        if mode not in ('strict', 'nondecreasing') or not 1 <= max_order <= original.max_order:
            raise ValueError('Invalid propagation specification')
        self.max_order, self.hidden_dim, self.mode = max_order, original.hidden_dim, mode
        self.projections = nn.ModuleList(list(original.projections)[:max_order])
        self.gates = nn.ModuleList(list(original.gates)[:max_order])

    def measured_step(self, old, old_support, previous_ego, current_ego, frame, continuity, return_parts=False):
        b, n, _, d = old.shape
        previous = old*continuity[:, :, None, None]
        support = old_support & continuity[:, :, None]
        gate_input = torch.cat([current_ego, frame.relation_mean], -1)
        new_states, new_support, messages, received_masks = [], [], [], []
        sources, carries = [], []
        for q in range(self.max_order):
            source = previous_ego if q == 0 else previous[:, :, q-1]
            source_valid = continuity if q == 0 else support[:, :, q-1]
            if q and self.mode == 'nondecreasing':
                # Add a same-time edge extension. Do not use the new lower
                # STATE: that would also attenuate old paths by its carry gate.
                source = source+messages[q-1]
                source_valid = source_valid | received_masks[q-1]
            message = aggregate(frame, self.projections[q](source))
            received = torch.zeros(b*n, device=old.device, dtype=torch.long).index_add(
                0, frame.target, source_valid.reshape(-1)[frame.source].long()).reshape(b, n) > 0
            valid_q = (support[:, :, q] | received) & continuity
            gamma = torch.sigmoid(self.gates[q](gate_input))
            carry = gamma*previous[:, :, q]
            state = torch.where(valid_q[..., None], carry+message, torch.zeros_like(carry))
            new_states.append(state)
            new_support.append(valid_q)
            messages.append(message)
            received_masks.append(received & continuity)
            sources.append(source)
            carries.append(carry)
        parts = (dict(before=previous, source=torch.stack(sources, 2),
                      message=torch.stack(messages, 2), carry=torch.stack(carries, 2)) if return_parts else None)
        return torch.stack(new_states, 2), torch.stack(new_support, 2), parts

    def forward(self, hidden, frames, continuity, return_parts=False):
        b, t_steps, n, d = hidden.shape
        if len(frames) != t_steps:
            raise ValueError('Graph sequence length mismatch')
        state = hidden.new_zeros(b, n, self.max_order, d)
        support = torch.zeros(b, n, self.max_order, device=hidden.device, dtype=torch.bool)
        states, supports, parts = [], [], []
        for t, frame in enumerate(frames):
            previous_ego = hidden[:, t-1] if t else torch.zeros_like(hidden[:, 0])
            state, support, detail = self.measured_step(
                state, support, previous_ego, hidden[:, t], frame, continuity[:, t], return_parts)
            states.append(state)
            supports.append(support)
            if return_parts:
                parts.append(detail)
        result = torch.stack(states, 1), torch.stack(supports, 1)
        if return_parts:
            return (*result, {k: torch.stack([p[k] for p in parts], 1) for k in parts[0]})
        return result


class OrderVerificationModel(TOHIResidual):
    def __init__(self, hidden_dim=64, order=3, time_mode='strict', delta_width=None, **kwargs):
        # Initialize full A-R once and then TRUNCATE modules, so all common
        # parameters start identically at each seed, including q1/q2 gates.
        kwargs.pop('max_order', None)
        super().__init__(hidden_dim=hidden_dim, max_order=3, **kwargs)
        self.order_state = MeasuredOrderState(self.order_state, order, time_mode)
        self.fusion.scores = nn.ModuleList(list(self.fusion.scores)[:order])
        if delta_width is not None and delta_width != hidden_dim:
            self.fusion.delta = nn.Sequential(nn.Linear(hidden_dim, delta_width, bias=False), nn.Tanh(),
                                               nn.Linear(delta_width, hidden_dim, bias=False))
        self.order, self.time_mode = order, time_mode
        self.delta_width = delta_width or hidden_dim

    def forward(self, observed, valid=None, timestamps=None, return_debug=False, intervention=None):
        encoded = self.encoder(observed, valid, timestamps)
        frames = self.graph.sequence(encoded)
        result = self.order_state(encoded['hidden'], frames, encoded['continuity'], return_parts=return_debug)
        states, support = result[:2]
        context, beta, extra = self.fusion(encoded['hidden'][:, -1], states[:, -1], support[:, -1], intervention)
        prediction = self.decoder(context, encoded['positions'][:, -1], encoded['valid'][:, -1])
        if return_debug:
            return prediction, dict(**encoded, graph_frames=frames, order_states=states,
                order_support=support, order_weights=beta, propagation_parts=result[2], **extra)
        return prediction


MODEL_SPECS = {
    'TOHI-H1': dict(order=1), 'TOHI-H2': dict(order=2), 'TOHI-H3': dict(order=3),
    'TOHI-NoOrder': dict(order=3, time_mode='nondecreasing'),
    'TOHI-H1-Cap': dict(order=1, delta_width=395),
    'TOHI-H2-Cap': dict(order=2, delta_width=230),
}


def build_order_model(name, hidden_dim=64):
    return OrderVerificationModel(hidden_dim=hidden_dim, **MODEL_SPECS[name])
