"""History-only fixed groups and decoder-input diagnostics."""
import numpy as np
import torch
from .interaction_coverage import count_ordered_walks
from ..models.ego_encoder import motion_inputs
from ..models.temporal_graph import select_candidates

GROUPS = ['all', 'no_neighbors', 'low_neighbors', 'medium_neighbors', 'high_neighbors',
          'order2_exists', 'order2_absent', 'order3_exists', 'order3_absent']


def history_groups(batch, radius=4., neighbors=8):
    encoded = motion_inputs(batch['observed'], batch['observed_valid'], batch['timestamps'])
    b, t_steps, n, _ = batch['observed'].shape
    masks = torch.stack([select_candidates(encoded['positions'][:, t], encoded['velocity'][:, t],
                          encoded['continuity'][:, t], radius, neighbors)[0] for t in range(t_steps)], 1)
    degree = masks.sum((1, 3))/encoded['continuity'].sum(1).clamp_min(1)
    support = torch.zeros(b, n, 3, dtype=torch.bool)
    for scene in range(b):
        counts = count_ordered_walks(masks[scene].numpy(), encoded['continuity'][scene].numpy())
        support[scene] = torch.from_numpy(counts['final_counts'].T > 0)
    cutoff = batch['observed_valid'][:, -1]
    result = dict(all=cutoff, no_neighbors=cutoff & (degree == 0),
        low_neighbors=cutoff & (degree > 0) & (degree <= 1),
        medium_neighbors=cutoff & (degree > 1) & (degree <= 3), high_neighbors=cutoff & (degree > 3))
    for q in (2, 3):
        result[f'order{q}_exists'] = cutoff & support[:, :, q-1]
        result[f'order{q}_absent'] = cutoff & ~support[:, :, q-1]
    return result


def complete_debug(model, debug):
    hidden = debug['hidden']
    ego = hidden[:, -1]
    b, n, d = ego.shape
    if 'order_states' in debug:
        raw = debug['order_states'][:, -1]
        support = debug.get('order_support_last', debug['order_support'][:, -1])
        debug.setdefault('used_order_states', raw)
        debug.setdefault('order_support_last', support)
        if 'fused_context' not in debug:
            debug['fused_context'] = model.fusion(ego, raw, support)[0]
    elif 'social_states' in debug:
        raw = debug['social_states'][:, -1, :, None]
        has_social = torch.zeros(b, n, device=ego.device, dtype=torch.bool)
        for t, frame in enumerate(debug['graph_frames']):
            degree = torch.zeros(b*n, device=ego.device).index_add(0, frame.target, torch.ones_like(frame.weight)).reshape(b, n)
            has_social = (has_social & debug['continuity'][:, t]) | (degree > 0)
        debug['used_order_states'] = raw
        debug['order_support_last'] = has_social[..., None]
        debug['order_weights'] = has_social[..., None].to(ego.dtype)
        context = model.mix(torch.cat([ego, raw[:, :, 0]], -1))
        debug['fused_context'] = torch.where(has_social[..., None], context, ego)
        debug['order_states'] = debug['social_states'][:, :, :, None]
    else:
        debug['order_states'] = hidden.new_zeros(b, hidden.shape[1], n, 0, d)
        debug['used_order_states'] = hidden.new_zeros(b, n, 0, d)
        debug['order_support_last'] = torch.zeros(b, n, 0, dtype=torch.bool, device=ego.device)
        debug['order_weights'] = hidden.new_zeros(b, n, 0)
        debug['fused_context'] = ego
    debug.setdefault('residual_gate', ego.new_zeros(b, n))
    debug.setdefault('residual_enabled', False)
    debug.setdefault('normalization_enabled', False)
    return debug


def summarize_debug(debug_batches):
    # Values are detached before leaving evaluation. No GT filtering in state diagnostics.
    output = {}
    keys = set().union(*(d.keys() for d in debug_batches))
    for key in sorted(keys):
        pieces = [d[key] for d in debug_batches if key in d and d[key].numel()]
        if not pieces:
            continue
        array = torch.cat(pieces).double().numpy()
        output[key+'_mean'] = float(array.mean())
        if key == 'residual_gate_active':
            output.update({key+'_p05': float(np.quantile(array, .05)), key+'_p50': float(np.quantile(array, .5)),
                           key+'_p95': float(np.quantile(array, .95)), key+'_min': float(array.min()),
                           key+'_max': float(array.max()), key+'_lt_1e3_fraction': float((array < 1e-3).mean())})
    return output


def extract_debug(debug):
    valid = debug['valid'][:, -1]
    ego = debug['hidden'][:, -1]
    raw, used, beta = debug['order_states'][:, -1], debug['used_order_states'], debug['order_weights']
    support = debug['order_support_last']
    values = dict(ego_L2=ego.norm(dim=-1)[valid].detach().cpu(),
                  decoder_context_change_L2=(debug['fused_context']-ego).norm(dim=-1)[valid].detach().cpu())
    for q in range(raw.shape[2]):
        values[f'q{q+1}_raw_L2'] = raw[:, :, q].norm(dim=-1)[valid].detach().cpu()
        values[f'q{q+1}_used_L2'] = used[:, :, q].norm(dim=-1)[valid].detach().cpu()
        values[f'q{q+1}_beta'] = beta[:, :, q][valid].detach().cpu()
        values[f'q{q+1}_support'] = support[:, :, q][valid].float().detach().cpu()
    if debug['residual_enabled']:
        active = valid & support.any(-1)
        values['residual_gate_all'] = debug['residual_gate'][valid].detach().cpu()
        values['residual_gate_active'] = debug['residual_gate'][active].detach().cpu()
    return values
