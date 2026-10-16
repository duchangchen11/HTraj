"""Only eight historical positions determine the fixed sampling distribution."""
import numpy as np

NAMES = ('stationary', 'low', 'normal', 'fast')
THRESHOLDS = (.05, .4, 1.8)


def classify_history(history):
    h = np.asarray(history, dtype=np.float64)
    if h.shape != (8, 2) or not np.isfinite(h).all():
        raise ValueError('Complete finite eight-frame history required')
    speed = float(np.linalg.norm(np.diff(h, axis=0) / .4, axis=-1).mean())
    return int(np.searchsorted(THRESHOLDS, speed, side='right')), speed


def probabilities(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, range(4)).all():
        raise ValueError('Nonempty integer classes 0..3 required')
    labels = labels.astype(np.int64)
    counts = np.bincount(labels, minlength=4)
    groups = int((counts > 0).sum())
    p = .5 / len(labels) + .5 / (groups * counts[labels])
    if not np.isclose(p.sum(), 1., atol=1e-12, rtol=0):
        raise AssertionError('Sampling probabilities do not sum to one')
    return p


def epoch_order(labels, condition, epoch, seed=17):
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError('Epoch starts at one')
    rng = np.random.default_rng(seed + epoch)
    if condition == 'Uniform':
        return rng.permutation(len(labels))
    if condition == 'Stratified':
        return rng.choice(len(labels), size=len(labels), replace=True, p=probabilities(labels))
    raise ValueError(condition)


def exposure(labels, order):
    labels = np.asarray(labels, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or len(order) != len(labels) or (order < 0).any() or (order >= len(labels)).any():
        raise ValueError('Exactly N legal visits required')
    counts = np.bincount(order, minlength=len(labels))
    rows = []
    for group, name in enumerate(NAMES):
        members = labels == group
        pool = int(members.sum())
        visits = int(counts[members].sum())
        unique = int((counts[members] > 0).sum())
        rows.append(dict(category=name, pool_targets=pool, visits=visits, unique_targets=unique,
            repeated_visits=visits-unique, repeated_target_ID_windows=int((counts[members] > 1).sum()),
            coverage=unique/pool if pool else None, sampled_fraction=visits/len(labels)))
    return counts, rows


def decision(overall, speed, stationary, integrity):
    """Preregistered gates; input rows must cover both recordings/weightings."""
    expected = {(r, w) for r in ('biwi_hotel', 'crowds_zara02')
                for w in ('target-window', 'equal-person-ID')}
    assert {(r['recording'], r['weighting']) for r in overall} == expected and len(overall) == 4
    checks = []
    for r in overall:
        ade_margin = max(.005, .01*r['Uniform_ADE_m'])
        fde_margin = max(.005, .01*r['Uniform_FDE_m'])
        checks += [dict(gate='ADE', recording=r['recording'], weighting=r['weighting'],
            observed_gain=r['Uniform_ADE_m']-r['Stratified_ADE_m'], required=ade_margin,
            passed=r['Uniform_ADE_m']-r['Stratified_ADE_m'] >= ade_margin),
            dict(gate='FDE', recording=r['recording'], weighting=r['weighting'],
            observed_regression=r['Stratified_FDE_m']-r['Uniform_FDE_m'], allowed=fde_margin,
            passed=r['Stratified_FDE_m']-r['Uniform_FDE_m'] <= fde_margin)]
    # Speed gate refers to absolute signed group-mean bias, window weighting.
    # Also report mean absolute per-target bias and ID weighting descriptively.
    a, b = abs(speed['Uniform_signed_bias_mps']), abs(speed['Stratified_signed_bias_mps'])
    checks.append(dict(gate='HOTEL_normal_speed_bias', Uniform_absolute_bias_mps=a,
        Stratified_absolute_bias_mps=b, required_ratio=.9, passed=b <= .9*a))
    assert {(r['recording'], r['weighting']) for r in stationary} == expected and len(stationary) == 4
    for r in stationary:
        margin = max(.01, .05*r['Uniform_FDE_m'])
        checks.append(dict(gate='stationary_FDE', recording=r['recording'], weighting=r['weighting'],
            observed_regression=r['Stratified_FDE_m']-r['Uniform_FDE_m'], allowed=margin,
            passed=r['Stratified_FDE_m']-r['Uniform_FDE_m'] <= margin))
    checks.append(dict(gate='integrity', passed=bool(integrity)))
    passed = all(r['passed'] for r in checks)
    return dict(all_passed=passed, decision='suggest_followup_only' if passed else 'stop_current_sampling_hypothesis',
        checks=checks, extra_training_authorized=False, confirmation_experiment=False)
