"""Stage3.6 history-only strata and paired deterministic evaluation."""
import numpy as np
import torch
from .benchmark_audit import history_features, target_errors, GROUPS as OLD_GROUPS
from ..training.smoke import move_batch

GROUPS = OLD_GROUPS + ['no_legal_social_messages', 'history_CV_like', 'history_motion_moderate', 'history_velocity_changing', 'history_motion_insufficient']


def history_groups(batch):
    from ..models.ego_encoder import motion_inputs
    groups, features = history_features(batch)
    e = motion_inputs(batch['observed'], batch['observed_valid'], batch['timestamps'])
    pair = e['continuity'][:, 1:] & e['continuity'][:, :-1]
    count = pair.sum(1)
    dv = (e['velocity'][:, 1:] - e['velocity'][:, :-1]).norm(dim=-1)
    score = (dv*pair).sum(1)/count.clamp_min(1)
    valid = e['valid'][:, -1]
    groups.update(no_legal_social_messages=groups['no_neighbors'],
        history_CV_like=valid & (count > 0) & (score < .2),
        history_motion_moderate=valid & (count > 0) & (score >= .2) & (score < .5),
        history_velocity_changing=valid & (count > 0) & (score >= .5),
        history_motion_insufficient=valid & (count == 0))
    features.update(history_velocity_change_mps=score, history_velocity_change_count=count,
        cutoff_speed_mps=e['velocity'][:, -1].norm(dim=-1),
        cutoff_velocity_x_mps=e['velocity'][:, -1, :, 0], cutoff_velocity_y_mps=e['velocity'][:, -1, :, 1])
    return groups, features


@torch.no_grad()
def evaluate(model, batches, device, cached=False, rows=False):
    model.eval()
    totals = {g: [0., 0, 0., 0] for g in GROUPS}
    records, diagnostics = [], {k: [] for k in ('state_norm', 'correction_norm', 'gate', 'message_norm')}
    graph = dict(real_edges=0, used_edges=0, changed_edges=0, incoming_budget_absolute_difference=0)
    distances = {'real': [], 'used': []}
    for original in batches:
        b = move_batch(original, device)
        prediction, debug = model.forward_cached(b, True) if cached else model(b['observed'], b['observed_valid'], b['timestamps'], return_debug=True)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation')
        ade, fde, a, f, count = target_errors(prediction, b['future'], b['future_valid'])
        for group, mask in original['groups'].items():
            am, fm = a & mask.to(device), f & mask.to(device)
            total = totals[group]
            total[0] += float(ade[am].sum()); total[1] += int(am.sum())
            total[2] += float(fde[fm].sum()); total[3] += int(fm.sum())
        zeros = ade.new_zeros(ade.shape)
        state = debug['hidden'][:, -1].norm(dim=-1)
        correction = debug.get('correction', debug.get('ego_delta', torch.zeros_like(prediction))).norm(dim=-1).mean(1)
        gate = debug.get('gate', zeros)
        message = debug['pooled_message'].norm(dim=-1) if 'pooled_message' in debug else zeros
        for key, value in [('state_norm', state), ('correction_norm', correction), ('gate', gate), ('message_norm', message)]:
            diagnostics[key].extend(value[a].cpu().tolist())
        for key in graph:
            graph[key] += debug.get(key, 0)
        for kind in distances:
            if kind+'_edge_distance' in debug:
                distances[kind].extend(debug[kind+'_edge_distance'].cpu().tolist())
        base_ade, base_fde = (None, None)
        if 'base_prediction' in debug:
            base_ade, base_fde, _, _, _ = target_errors(debug['base_prediction'], b['future'], b['future_valid'])
        if rows:
            for bi, ni in a.nonzero().tolist():
                records.append(dict(recording=original['recording'][bi], sample_id=original['sample_id'][bi], target_id=int(original['agent_ids'][bi, ni]),
                    ADE_m=float(ade[bi, ni]), FDE_m=float(fde[bi, ni]) if f[bi, ni] else None, future_points=int(count[bi, ni]),
                    anchor_ADE_m=float(base_ade[bi, ni]) if base_ade is not None else None,
                    anchor_FDE_m=float(base_fde[bi, ni]) if base_fde is not None and f[bi, ni] else None,
                    state_norm=float(state[bi, ni]), correction_norm=float(correction[bi, ni]), gate=float(gate[bi, ni]), message_norm=float(message[bi, ni]),
                    **{k: float(v[bi, ni]) for k, v in original['history_features'].items()},
                    **{g: bool(mask[bi, ni]) for g, mask in original['groups'].items()}))
    groups = [dict(group=g, ADE_count=s[1], FDE_count=s[3], ADE_m=s[0]/s[1] if s[1] else None, FDE_m=s[2]/s[3] if s[3] else None) for g, s in totals.items()]
    summary = dict(**graph, changed_edge_fraction=graph['changed_edges']/max(graph['real_edges'], 1))
    for key, values in dict(diagnostics, real_edge_distance_m=distances['real'], used_edge_distance_m=distances['used']).items():
        for label, result in [('mean', np.mean(values) if values else None), ('p05', np.quantile(values, .05) if values else None), ('median', np.median(values) if values else None), ('p95', np.quantile(values, .95) if values else None)]:
            summary[key+'_'+label] = float(result) if result is not None else None
    return dict(ADE_m=groups[0]['ADE_m'], FDE_m=groups[0]['FDE_m'], groups=groups, targets=records, diagnostics=summary)
