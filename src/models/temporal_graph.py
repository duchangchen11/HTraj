"""Sparse prefix graph. A[target, source] sends source -> target."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class GraphFrame:
    source: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    features: torch.Tensor
    relation_mean: torch.Tensor

    @classmethod
    def artificial(cls, batch, people, edges, device='cpu', dtype=torch.float32):
        # edges = [(batch_index, source_person, target_person, weight), ...]
        source = torch.tensor([b*people+j for b, j, i, w in edges], device=device, dtype=torch.long)
        target = torch.tensor([b*people+i for b, j, i, w in edges], device=device, dtype=torch.long)
        if source.numel() and ((source == target).any() or (source//people != target//people).any()):
            raise ValueError('Artificial graph self/cross-scene edge')
        weights = torch.tensor([w for b, j, i, w in edges], device=device, dtype=dtype)
        if not torch.isfinite(weights).all():
            raise ValueError('Nonfinite graph weights')
        return cls(source, target, weights, torch.zeros(len(edges), 7, device=device, dtype=dtype),
                   torch.zeros(batch, people, 7, device=device, dtype=dtype))


def relation_geometry(position, velocity, tau_max=4.8):
    # [B,target,source,2], no other scene can participate.
    r = position[:, :, None]-position[:, None, :]
    w = velocity[:, :, None]-velocity[:, None, :]
    distance = torch.linalg.vector_norm(r, dim=-1)
    tau = (-(r*w).sum(-1)/(w.square().sum(-1)+1e-6)).clamp(0, tau_max)
    closest = torch.linalg.vector_norm(r+tau[..., None]*w, dim=-1)
    return r, w, distance, tau, closest


def select_candidates(position, velocity, continuity, radius=4.0, max_neighbors=8, tau_max=4.8):
    if radius <= 0 or max_neighbors < 0:
        raise ValueError('Invalid graph specification')
    b, n, _ = position.shape
    r, w, distance, tau, closest = relation_geometry(position, velocity, tau_max)
    candidate = continuity[:, :, None] & continuity[:, None, :]
    candidate = candidate & ~torch.eye(n, dtype=torch.bool, device=position.device)[None] & (distance <= radius)
    # Fixed geometric ranking is allowed by A0 for the prototype; learned MLP
    # assigns alpha on the selected set. Tie groups are kept or dropped whole,
    # avoiding an index-based tie break that violates permutation equivariance.
    score = closest+0.05*distance
    if max_neighbors == 0:
        candidate = torch.zeros_like(candidate)
    elif max_neighbors < n:
        ranked = torch.where(candidate, score, torch.full_like(score, torch.inf))
        boundary = ranked.sort(-1).values[..., max_neighbors-1:max_neighbors]
        inclusive = candidate & (score <= boundary)
        strict = candidate & (score < boundary)
        candidate = torch.where((inclusive.sum(-1) > max_neighbors)[..., None], strict, inclusive)
    features = torch.cat([r, w, tau[..., None], closest[..., None], candidate[..., None].to(position.dtype)], -1)
    return candidate, features


def target_softmax(score, target, num_nodes):
    if score.numel() == 0:
        return score
    maxima = score.new_full((num_nodes,), -torch.inf)
    maxima.scatter_reduce_(0, target, score, reduce='amax', include_self=True)
    exp = torch.exp(score-maxima[target])
    denominator = exp.new_zeros(num_nodes).index_add(0, target, exp)
    return exp/denominator[target].clamp_min(torch.finfo(exp.dtype).tiny)


def aggregate(frame, source_features):
    b, n, d = source_features.shape
    messages = source_features.reshape(b*n, d)[frame.source]*frame.weight[:, None]
    return source_features.new_zeros(b*n, d).index_add(0, frame.target, messages).reshape(b, n, d)


class TemporalGraph(nn.Module):
    def __init__(self, hidden_dim=64, radius=4.0, max_neighbors=8, tau_max=4.8):
        super().__init__()
        self.radius, self.max_neighbors, self.tau_max = radius, max_neighbors, tau_max
        self.relation_score = nn.Sequential(nn.Linear(7+2*hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, position, velocity, continuity, current_ego, previous_ego):
        b, n, _ = position.shape
        mask, features = select_candidates(position, velocity, continuity, self.radius, self.max_neighbors, self.tau_max)
        batch, target, source = mask.nonzero(as_tuple=True)
        edge_features = features[batch, target, source]
        score_input = torch.cat([edge_features, current_ego[batch, target], previous_ego[batch, source]], -1)
        scores = self.relation_score(score_input).squeeze(-1)
        flat_source, flat_target = batch*n+source, batch*n+target
        weights = target_softmax(scores, flat_target, b*n)
        feature_sum = position.new_zeros(b*n, 7).index_add(0, flat_target, edge_features)
        degree = position.new_zeros(b*n).index_add(0, flat_target, torch.ones_like(weights))
        mean = (feature_sum/degree[:, None].clamp_min(1)).reshape(b, n, 7)
        return GraphFrame(flat_source, flat_target, weights, edge_features, mean)

    def sequence(self, encoded):
        hidden = encoded['hidden']
        frames = []
        for t in range(hidden.shape[1]):
            previous = hidden[:, t-1] if t else torch.zeros_like(hidden[:, 0])
            frames.append(self(encoded['positions'][:, t], encoded['velocity'][:, t],
                               encoded['continuity'][:, t], hidden[:, t], previous))
        return frames
