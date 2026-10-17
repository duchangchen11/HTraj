"""Units: global metres, seconds, radians. All groups use history only."""
import numpy as np

DT = .4


def fixed_subsets(seed=7501, repetitions=20):
    """Generated without trajectories or labels; first schedule is official head order."""
    rng = np.random.default_rng(seed)
    orders = [np.arange(20)] + [rng.permutation(20) for _ in range(repetitions)]
    return [{"order_id": j, "heads": p.tolist()} for j, p in enumerate(orders)]


def angle(a, b):
    return np.arctan2(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0], (a*b).sum(-1))


def ols_velocity(x, n=4):
    t = np.arange(n)*DT
    t -= t.mean()
    return (x[..., -n:, :]*t[:, None]).sum(-2)/(t*t).sum()


def history_features(x):
    x = np.asarray(x, dtype=float)
    v = np.diff(x, axis=0)/DT
    s = np.linalg.norm(v, axis=-1)
    mean_speed = float(s.mean())
    h = ols_velocity(x)
    first = ols_velocity(x[:4])
    reliable = np.linalg.norm(h) >= .05 and np.linalg.norm(first) >= .05
    turn = float(angle(first, h)) if reliable else None
    t = np.arange(8)*DT
    fit = x.mean(0)+(t-t.mean())[:, None]*ols_velocity(x, 8)
    return {
        'history_speed_mps': mean_speed,
        'speed_group': ['stationary', 'low', 'normal', 'fast'][np.digitize(mean_speed, [.05, .4, 1.8])],
        'history_OLS4_speed_mps': float(np.linalg.norm(h)),
        'history_speed_sd_mps': float(s.std()),
        'history_speed_change_mps': float(np.linalg.norm(np.diff(v, axis=0), axis=-1).mean()),
        'history_path_m': float(s.sum()*DT),
        'history_net_m': float(np.linalg.norm(x[-1]-x[0])),
        'history_fit_RMS_m': float(np.sqrt(np.square(x-fit).sum(-1).mean())),
        'history_turn_rad': turn,
        'turn_group': 'unreliable' if turn is None else ('turning' if abs(turn) >= np.pi/6 else 'straight'),
    }


def motion_metrics(x, y, p):
    """Future used only in diagnostic outcomes, never history grouping or subset choice."""
    x, y, p = [np.asarray(a, dtype=float) for a in (x, y, p)]
    if x.shape != (8, 2) or y.shape != (12, 2) or p.shape != (12, 2):
        raise ValueError('Expected complete 8/12 positions')
    if not all(np.isfinite(a).all() for a in (x, y, p)):
        raise ValueError('Nonfinite input')
    gt_steps = np.diff(np.concatenate([x[-1:], y]), axis=0)/DT
    pr_steps = np.diff(np.concatenate([x[-1:], p]), axis=0)/DT
    gt_speed = float(np.linalg.norm(gt_steps, axis=-1).mean())
    pr_speed = float(np.linalg.norm(pr_steps, axis=-1).mean())
    gd, pd = y[-1]-x[-1], p[-1]-x[-1]
    gn, pn = np.linalg.norm(gd), np.linalg.norm(pd)
    hv = ols_velocity(x)
    hn = np.linalg.norm(hv)
    reliable = gn >= .24 and pn >= .24
    heading_error = float(angle(gd, pd)) if reliable else None
    gt_turn = float(angle(hv, gd)) if hn >= .05 and gn >= .24 else None
    pr_turn = float(angle(hv, pd)) if hn >= .05 and pn >= .24 else None
    err = np.linalg.norm(p-y, axis=-1)
    gt_path = gt_speed*4.8
    result = dict(ADE=float(err.mean()), FDE=float(err[-1]),
        GT_speed_mps=gt_speed, predicted_speed_mps=pr_speed,
        speed_bias_mps=pr_speed-gt_speed, abs_speed_bias_mps=abs(pr_speed-gt_speed),
        GT_path_m=gt_path, GT_net_m=float(gn), predicted_net_m=float(pn),
        GT_heading_rad=float(np.arctan2(gd[1],gd[0])) if gn >= .24 else None,
        predicted_heading_rad=float(np.arctan2(pd[1],pd[0])) if pn >= .24 else None,
        heading_error_rad=heading_error,
        abs_heading_error_rad=abs(heading_error) if heading_error is not None else None,
        GT_turn_rad=gt_turn, predicted_turn_rad=pr_turn,
        turn_magnitude_bias_rad=abs(pr_turn)-abs(gt_turn) if gt_turn is not None and pr_turn is not None else None,
        speed_change_bias_mps=float(np.linalg.norm(pr_steps[-3:],axis=-1).mean()-np.linalg.norm(pr_steps[:3],axis=-1).mean()
                                  -np.linalg.norm(gt_steps[-3:],axis=-1).mean()+np.linalg.norm(gt_steps[:3],axis=-1).mean()),
        endpoint_along_history_m=float((pd-gd)@hv/hn) if hn >= .05 else None,
        endpoint_cross_history_m=float((hv[0]*(pd-gd)[1]-hv[1]*(pd-gd)[0])/hn) if hn >= .05 else None)
    for floor in (.25, .5, 1.):
        result[f'ADE_over_GT_path_floor{floor}'] = float(err.mean()/max(gt_path,floor))
        result[f'FDE_over_GT_net_floor{floor}'] = float(err[-1]/max(gn,floor))
    result['ADE_over_history_travel_floor0.5'] = float(err.mean()/max(np.linalg.norm(np.diff(x,axis=0),axis=-1).mean()*12,.5))
    return result


def candidate_metrics(x, y, p):
    if np.shape(p) != (20,12,2):
        raise ValueError('Official twenty candidates required')
    p = np.asarray(p, dtype=float)
    e = np.linalg.norm(p-np.asarray(y)[None],axis=-1)
    ade, fde = e.mean(-1), e[:,-1]
    ii, jj = np.triu_indices(20,1)
    dist = np.linalg.norm(p[ii]-p[jj],axis=-1)
    end = dist[:,-1]
    cov = np.cov(p[:,-1].T,ddof=0)
    ev = np.linalg.eigvalsh(cov).clip(0)
    def credit(a):
        w = np.abs(a-a.min()) <= 1e-8
        return w/w.sum()
    summary = dict(endpoint_pair_mean_m=float(end.mean()), endpoint_pair_P10_m=float(np.quantile(end,.1)),
        path_pair_mean_m=float(dist.mean(-1).mean()),
        near_pair_fraction=float(((dist.mean(-1)<.2)&(end<.3)).mean()),
        near_pair_fraction_tight=float(((dist.mean(-1)<.1)&(end<.15)).mean()),
        near_pair_fraction_loose=float(((dist.mean(-1)<.4)&(end<.6)).mean()),
        endpoint_cov_effective_rank=float(ev.sum()**2/(np.square(ev).sum()+1e-30)),
        endpoint_RMS_m=float(np.sqrt(ev.sum())))
    return ade, fde, credit(ade), credit(fde), summary


def social_features(raw, frame, person_id):
    """All full8 historical neighbours, independently of future existence; not model mask."""
    raw = np.asarray(raw)
    raw = raw[(raw[:,0] >= frame-7) & (raw[:,0] <= frame)]
    frames = [{int(i): np.array([a,b]) for t,i,a,b in raw if int(t)==f} for f in range(frame-7,frame+1)]
    full = set.intersection(*(set(f) for f in frames))
    i = int(person_id)
    if i not in full:
        raise ValueError('Target history missing')
    others = sorted(full-{i})
    r = np.array([frames[-1][j]-frames[-1][i] for j in others]).reshape(-1,2)
    v_i = (frames[-1][i]-frames[-2][i])/DT
    rv = np.array([(frames[-1][j]-frames[-2][j])/DT-v_i for j in others]).reshape(-1,2)
    d = np.linalg.norm(r,axis=-1)
    local = d <= 2.
    closing = -(r*rv).sum(-1)/np.maximum(d,1e-8)
    return dict(history_other_full8=len(others), history_neighbors_2m=int(local.sum()),
        density_group='0' if not local.sum() else ('1-3' if local.sum()<=3 else ('4-9' if local.sum()<=9 else '10+')),
        nearest_history_neighbor_m=float(d.min()) if len(d) else None,
        local_relative_speed_mps=float(np.linalg.norm(rv[local],axis=-1).mean()) if local.any() else None,
        local_closing_mps=float(closing[local].mean()) if local.any() else None,
        local_any_approaching=bool((closing[local]>.1).any()) if local.any() else False)
