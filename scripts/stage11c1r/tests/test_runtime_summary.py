import ast,inspect,csv
import numpy as np
from scripts.stage11c1r.analyze import distribution
from scripts.stage11c1r.observer import KMeansObserver
from scripts.stage11c1r.common import OUT

def test_runtime_distribution_and_missing_values():
    r=distribution([1.,2.,3.,4.])
    assert r['n']==4 and r['median']==2.5 and r['total']==10
    assert np.isclose(r['P90'],3.7)
    empty=distribution([]);assert empty['mean'] is None and empty['P90'] is None

def test_frozen_samples_use_history_only_selection():
    rows=list(csv.DictReader((OUT/'diagnostic_sample_manifest.csv').open()))
    assert len(rows)==15 and len({r['sample_id'] for r in rows})==15
    assert not any(k in rows[0] for k in ['ADE','FDE','GT','future','runtime'])
    for scene in {r['scene'] for r in rows}:
        g=[r for r in rows if r['scene']==scene]
        assert len(g)==3 and {float(r['quantile']) for r in g}=={.2,.5,.8}

def test_observer_callback_does_not_execute_tensor_numerics():
    source=inspect.getsource(KMeansObserver.callback)
    forbidden=['.item(','.cpu(','.cuda(','.clone(','.float(','.numpy(','.to(']
    assert all(s not in source for s in forbidden)
