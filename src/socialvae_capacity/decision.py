"""Frozen four-cell stage5p4 development gate, not a significance test."""
import math


def classify(cells,rule):
    expected={(rec,w) for rec in ['biwi_hotel','crowds_zara02'] for w in rule['weightings']}
    if len(cells)!=4 or {(c['recording'],c['weighting']) for c in cells}!=expected:
        raise ValueError('Both recordings and both weightings are mandatory')
    outcomes=[]
    for c in cells:
        fa,ff,ea,ef=[c[k] for k in ['Full_ADE_m','Full_FDE_m','EgoCapacity_ADE_m','EgoCapacity_FDE_m']]
        if not all(math.isfinite(v) and v>=0 for v in [fa,ff,ea,ef]) or ea==0:
            raise ValueError('Finite paired errors and positive ego ADE required')
        outcomes.append(dict(**c,Full_relative_ADE_gain=(ea-fa)/ea,
            ego_sufficient=ea<=fa and ef<=ff*(1+rule['FDE_substantial_regression_relative']),
            Full_qualifies=(ea-fa)/ea>=rule['Full_meaningful_ADE_gain_relative'] and ff<=ef*(1+rule['FDE_substantial_regression_relative'])))
    sufficient=all(c['ego_sufficient'] for c in outcomes);qualified=all(c['Full_qualifies'] for c in outcomes)
    return dict(cells=outcomes,ego_sufficient_all_four=sufficient,Full_qualifies_all_four=qualified,
        phase2_allowed=qualified and not sufficient,
        verdict='ego_capacity_sufficient_stop' if sufficient else 'Full_meets_gate_allow_phase2_protocol' if qualified else 'conditional_or_weak_evidence_stop')
