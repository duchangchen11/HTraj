"""Stage10A S/D/R rules. Hard edge selection has no trainable parameters."""
from dataclasses import dataclass
import torch
from .interaction_features import historical_motion, pair_features
from .risk_estimator import closest_approach


@dataclass(frozen=True)
class GraphConfig:
    radius: float = 4.
    max_neighbors: int = 8
    horizon: float = 4.8
    distance_scale: float = 1.
    dt: float = .4
    speed_epsilon: float = 1e-4

    def __post_init__(self):
        if min(self.radius, self.horizon, self.distance_scale, self.dt, self.speed_epsilon) <= 0:
            raise ValueError('Scales must be positive')
        if self.max_neighbors < 0:
            raise ValueError('Invalid neighbor budget')


def stable_select(score, candidates, ids, k):
    """Ascending score then immutable source ID; ties never use storage index."""
    n = score.shape[-1]
    if k == 0 or n == 0:
        return torch.zeros_like(candidates)
    order_id = torch.argsort(ids, dim=-1, stable=True)
    shape = [ids.shape[0]] + [1] * (score.ndim - 2) + [n]
    order_id = order_id.reshape(shape).expand_as(score)
    sorted_score = torch.gather(torch.where(candidates, score, torch.inf), -1, order_id)
    ranking = torch.argsort(sorted_score, dim=-1, stable=True)
    chosen = torch.gather(order_id, -1, ranking)[..., :min(k, n)]
    result = torch.zeros_like(candidates).scatter(-1, chosen, True)
    return result & candidates


def build_graph(position, valid, ids, mode='D', config=None):
    c = config or GraphConfig()
    if mode not in ('S', 'D', 'R', 'E'):
        raise ValueError('Unknown graph rule')
    p, v, vv = historical_motion(position, valid, c.dt)
    b, t, n, _ = p.shape
    if ids.shape != (b, n) or ids.dtype != torch.int64:
        raise ValueError('Stable integer IDs[B,N] required')
    # IDs are identifiers only: never neural inputs. Padding IDs must also be
    # unique; no fallback to storage-index tie breaks.
    if n > 1 and (ids.sort(-1).values[:, 1:] == ids.sort(-1).values[:, :-1]).any():
        raise ValueError('Duplicate node IDs')
    r, w, known = pair_features(p, v, vv)
    geometry = closest_approach(r, w, known, c.horizon, c.distance_scale, c.speed_epsilon)
    distance = torch.linalg.vector_norm(r, dim=-1)
    visible = valid[..., :, None] & valid[..., None, :]
    visible = visible & ~torch.eye(n, dtype=torch.bool, device=p.device)
    candidate = visible & (distance <= c.radius)
    first_distance = torch.zeros_like(distance)
    if mode == 'S':
        # Causal sticky static graph: each pair is considered at first joint
        # observation only. Fill remaining K slots then freeze accepted edges;
        # departures only mask them, never recycle capacity or reconsider pairs.
        seen = torch.zeros((b, n, n), dtype=torch.bool, device=p.device)
        admitted = torch.zeros_like(seen)
        first = torch.zeros((b, n, n), dtype=p.dtype, device=p.device)
        edges, pools = [], []
        for step in range(t):
            new = visible[:, step] & ~seen
            first = torch.where(new, distance[:, step], first)
            first_distance[:, step] = first
            eligible = new & (first <= c.radius)
            ranked = stable_select(first, eligible, ids, c.max_neighbors)
            # Keep only as many ranked new entries as vacant permanent slots.
            order = stable_select(first, eligible, ids, n)
            order_ids = torch.argsort(ids, stable=True)[:, None, :].expand(b, n, n)
            vals = torch.gather(torch.where(order, first, torch.inf), -1, order_ids)
            ix = torch.gather(order_ids, -1, vals.argsort(dim=-1, stable=True))
            rank = torch.zeros_like(ix).scatter(-1, ix, torch.arange(n, device=p.device).expand(b, n, n))
            vacant = (c.max_neighbors - admitted.sum(-1)).clamp_min(0)
            admitted = admitted | (ranked & (rank < vacant[..., None]))
            seen = seen | visible[:, step]
            edges.append(admitted & visible[:, step])
            pools.append(seen & (first <= c.radius) & visible[:, step])
        edge = torch.stack(edges, 1)
        candidate = torch.stack(pools, 1)
    else:
        score = -geometry['risk'] if mode == 'R' else distance
        edge = stable_select(score, candidate, ids, c.max_neighbors)
        if mode == 'E':
            edge = torch.zeros_like(edge)
    # Same nine-dimensional message features for S/D/R. Risk is diagnostic and
    # selector only; no model receives an extra feature solely because it is R.
    feature = torch.cat([r, w, vv[..., :, None, None].expand(b, t, n, n, 1).to(p.dtype),
                         vv[..., None, :, None].expand(b, t, n, n, 1).to(p.dtype),
                         distance[..., None], geometry['dca'][..., None], geometry['t_cpa'][..., None]], -1)
    feature = torch.where(visible[..., None], feature, 0.)
    return dict(candidate_mask=candidate, edge_mask=edge, relative_position=r,
                relative_velocity=w, features=feature, distance=distance,
                first_distance=first_distance, velocity=v, velocity_valid=vv, **geometry)
