import ast,inspect,json
from pathlib import Path
import numpy as np
import pytest
from scripts.stage11b2c.math_audit import probability,theoretical_hit,classify,neighborhood,coverage
from scripts.stage11b2c.common import ROOT,CONF,Budget
from scripts.stage11b2c.adapter import AuditPredictor
from scripts.stage11b1r.predictor import Predictor

@pytest.mark.parametrize('mass', [0,1e-15,1e-5,.001,.1,1])
def test_theoretical_hit_stability(mass):
    h=theoretical_hit(mass)
    assert 0<=h<=1
    if mass in [0,1]:assert h==mass
    else:assert np.isclose(h,1-(1-mass)**10000,rtol=1e-3,atol=1e-12)

def test_weights_not_inplace_changed_for_offline_normalization():
    w=np.array([[.2,.3],[.4,.1]],dtype='float32');before=w.copy();p=probability(w)
    assert np.array_equal(w,before) and p.dtype==np.float64 and abs(p.sum()-1)<1e-14

def test_original_pixel_radius_and_index_mapping():
    p=np.zeros((3,5));p[1,2]=1;z=np.zeros_like(p)
    r=neighborhood(p,z,z,[12,4],.25,[4,3]);assert r[4]['P_mass']==1 and r[3]['P_mass']==0
    d,c=coverage([[8,4]],[12,4],[4,3]);assert d==4 and c[4]['cover'] and not c[3]['cover']

@pytest.mark.parametrize('A,B,C',[(a,b,c) for a in [False,True] for b in [False,True] for c in [False,True]])
def test_all_binary_events_retained(A,B,C):
    f,cat=classify(A,B,C);assert set(f)=={'T1','T2','T3','T4','T5','T6'}
    if not B and C:assert f['T5'] and cat=='proposal_miss_final_hit'
    if A and B and not C:assert f['T3'] and cat=='proposal_hit_final_miss'
    if not A and B:assert cat=='other'

def test_new_forward_preserves_original_neural_and_sampling_statement_AST():
    import textwrap
    old=ast.parse(textwrap.dedent(inspect.getsource(Predictor.predict))).body[0].body
    new=ast.parse(textwrap.dedent(inspect.getsource(AuditPredictor.predict))).body[0].body
    for a,b in zip(old[:-1],new[:len(old)-1]):assert ast.dump(a)==ast.dump(b)
    assert isinstance(old[-1],ast.Return) and isinstance(new[len(old)-1],ast.Assign)
    assert ast.dump(old[-1].value.elts[2])==ast.dump(new[len(old)-1].value)

def test_GPU_entrypoints_cannot_load_label_tables():
    for name in ['adapter.py','worker_common.py','precheck.py','extract.py']:
        source=(ROOT/'scripts/stage11b2c'/name).read_text();tree=ast.parse(source)
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert 'scripts.stage11b1c.data' not in imports
        assert 'truths(' not in source and 'motion_features.csv' not in source and '.backward(' not in source and 'torch.optim' not in source
    assert 'future' not in inspect.signature(AuditPredictor.predict).parameters

def test_frozen_sampler_config_and_budget():
    p=json.loads((CONF/'analysis_protocol.json').read_text())
    assert p['proposals']==10000 and p['kmeans']==dict(clusters=19,distance='euclidean',tol=.001,iter_limit=1000)
    assert p['radii']==[10,25,50] and p['resize']==.25 and p['budget']['hard_seconds']==7200
    with pytest.raises(TimeoutError):Budget(6000,clock=lambda:100,started=0).guard(500)
