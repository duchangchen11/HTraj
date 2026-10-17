"""Analytic/audit tests, not a new trajectory forecasting model."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

spec = importlib.util.spec_from_file_location('stage7p6_audit', Path(__file__).parents[1]/'scripts/stage7p6/audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_supervision_subset_and_missing_label_rejection():
    row = dict(ids_json='[1,2,3]', target_ids_json='[1]', future_lengths_json='[12,0,5]')
    assert audit.supervised_sets(row) == ([1,2,3],[1],[12,0,5])
    for targets in ('[2]','[7]'):
        with pytest.raises(ValueError):
            audit.supervised_sets(dict(row,target_ids_json=targets))


def test_context_order_does_not_change_label_qualification():
    a = audit.supervised_sets(dict(ids_json='[3,1,2]',target_ids_json='[1]',future_lengths_json='[5,12,0]'))
    b = audit.supervised_sets(dict(ids_json='[1,2,3]',target_ids_json='[1]',future_lengths_json='[12,0,5]'))
    assert dict(zip(a[0],a[2])) == dict(zip(b[0],b[2]))


@pytest.mark.parametrize('y,variance', [(0.,1.),(2.,.5),(4.,.1)])
def test_constrained_bound_and_exact_marginal_by_quadrature(y,variance):
    result = audit.constrained_gaussian_bound(y,variance)
    # Integrate p(w)N(y;w,noise), w=z_s+z_u ~ N(0,2).
    normal = lambda x,m,v: math.exp(-(x-m)**2/(2*v))/math.sqrt(2*math.pi*v)
    density = quad(lambda w: normal(w,0,2)*normal(y,w,variance),-np.inf,np.inf,epsabs=1e-12)[0]
    assert result['log_evidence'] == pytest.approx(math.log(density),abs=1e-8)
    assert result['ELBO'] < result['log_evidence']
    assert result['KL_u'] == audit.normal_kl(0.,1.) == 0.


def test_prior_completion_is_valid_but_not_the_true_posterior():
    result = audit.constrained_gaussian_bound(3.,1.)
    assert result['true_posterior_u_mu'] != 0.
    assert result['gap'] > 0.


def test_supervised_denominator_excludes_unlabeled_context():
    # Adding 10 unlabeled nodes must not dilute these two defined losses.
    assert audit.supervised_average([2.,4.]) == 3.
    assert audit.supervised_average([2.,4.]) != sum([2.,4.])/12
    with pytest.raises(ValueError):
        audit.supervised_average([])


def test_fully_supervised_limit_and_finite_variance_rules():
    row = dict(ids_json='[1,2]',target_ids_json='[1,2]',future_lengths_json='[12,12]')
    ids,targets,_ = audit.supervised_sets(row)
    assert ids == targets
    assert audit.supervised_average([8.,12.]) == sum([8.,12.])/len(ids)
    for v in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):
            audit.normal_kl(0.,v)


def test_saved_counterfactual_scope_and_resource_scope():
    root = Path(__file__).parents[1]
    tests = list((root/'results/tables/stage7p1/counterfactuals').glob('*.json'))
    assert len(tests) == 10
    for path in tests:
        row = json.loads(path.read_text())
        assert row['K1_prediction_max_difference'] == 0.
        assert row['K20_prediction_max_difference'] == 0.
    row = json.loads((root/'results/tables/stage7p1/maximum_context_resource.json').read_text())
    assert row['people'] == 73
    assert row['tested_mode'] == 'learned-prior K20 inference ONLY'
    assert row['training_memory_feasibility'] == 'NOT_ESTABLISHED'
