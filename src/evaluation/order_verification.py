"""History-length intersections and detached state decomposition diagnostics."""
import numpy as np
import torch
from .social_diagnostics import GROUPS, history_groups, complete_debug, extract_debug

LENGTHS = ('full8', 'partial')
DENSITIES = ('no_neighbors', 'low_neighbors', 'medium_neighbors', 'high_neighbors')
CHAINS = ('order2_exists', 'order2_absent', 'order3_exists', 'order3_absent')
VERIFICATION_GROUPS = [*GROUPS, *['history_'+length for length in LENGTHS],
    *[length+'__'+group for length in LENGTHS for group in (*DENSITIES, *CHAINS)],
    *[length+'__'+density+'__'+chain for length in LENGTHS for density in DENSITIES for chain in CHAINS]]


def verification_groups(batch):
    original = history_groups(batch)
    cutoff = original['all']
    full = batch['observed_valid'].sum(1) == 8
    lengths = dict(full8=cutoff & full, partial=cutoff & ~full)
    result = dict(original)
    for length, mask in lengths.items():
        result['history_'+length] = mask
        for group in (*DENSITIES, *CHAINS):
            result[length+'__'+group] = mask & original[group]
        for density in DENSITIES:
            for chain in CHAINS:
                result[length+'__'+density+'__'+chain] = mask & original[density] & original[chain]
    assert set(result) == set(VERIFICATION_GROUPS)
    return result


def extract_measured_debug(model, debug):
    debug = complete_debug(model, debug)
    values = extract_debug(debug)
    valid = debug['valid'][:, -1]
    if 'social_delta' in debug:
        values['social_delta_L2'] = debug['social_delta'].norm(dim=-1)[valid].detach().cpu()
        values['residual_correction_L2'] = (debug['residual_gate'][..., None]*debug['social_delta']).norm(dim=-1)[valid].detach().cpu()
    if 'propagation_parts' in debug:
        for key, tensor in debug['propagation_parts'].items():
            for q in range(tensor.shape[3]):
                values[f'q{q+1}_{key}_L2'] = tensor[:, -1, :, q].norm(dim=-1)[valid].detach().cpu()
        for q in range(debug['order_states'].shape[3]):
            values[f'q{q+1}_after_L2'] = debug['order_states'][:, -1, :, q].norm(dim=-1)[valid].detach().cpu()
            norm = debug['order_states'][:, :, :, q].norm(dim=-1)
            values[f'q{q+1}_history_after_L2'] = norm[debug['valid']].detach().cpu()
    return values


def summarize_measured_debug(batches):
    output = {}
    for key in sorted(set().union(*(d.keys() for d in batches))):
        pieces = [d[key] for d in batches if key in d and d[key].numel()]
        if not pieces:
            continue
        array = torch.cat(pieces).double().numpy()
        output[key+'_mean'] = float(array.mean())
        if '_L2' in key or key.startswith('residual_gate_'):
            output[key+'_p95'] = float(np.quantile(array, .95))
        if key == 'residual_gate_active':
            for label, quantile in [('p05', .05), ('p50', .5)]:
                output[key+'_'+label] = float(np.quantile(array, quantile))
            output[key+'_min'] = float(array.min())
            output[key+'_max'] = float(array.max())
            output[key+'_lt_1e3_fraction'] = float((array < 1e-3).mean())
    return output
