import inspect
import numpy as np
import pytest,torch
from scripts.stage11b0.runtime import original_modules
from scripts.stage11b1.evaluation import sample_seed,history_record
from scripts.stage11b1r.sampler import sample_cpu,probability_weights,isolated_kmeans_rng,cluster_proposals
from scripts.stage11b1r.predictor import Predictor

def test_exact_author_CPU_semantics_and_indices():
    p=torch.arange(1.,321.).reshape(1,1,16,20);sid='test:semantic'
    before=torch.get_rng_state().clone()
    actual,w,indices=sample_cpu(p,sid)
    assert torch.equal(before,torch.get_rng_state())
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed(sid))
        expected=original_modules().image.sampling(p,10000,replacement=True,rel_threshold=.01)
    assert torch.equal(expected,actual)
    assert actual.dtype==torch.float32 and w.dtype==torch.float32
    assert torch.equal((actual[...,1]*20+actual[...,0]).long().reshape(1,-1),indices)

def test_five_repeats_order_and_threshold():
    p=torch.ones(1,1,8,8);p[0,0,0,0]=.001;p[0,0,0,1]=.01
    a=sample_cpu(p,'a')[0]
    for _ in range(5):assert torch.equal(a,sample_cpu(p,'a')[0])
    sample_cpu(p,'b');assert torch.equal(a,sample_cpu(p,'a')[0])
    w=probability_weights(p)
    assert w[0,0]==0 and w[0,1]>0  # Strict < threshold; equality retained.
    assert not ((a[...,0]==0)&(a[...,1]==0)).any()

def test_local_KMeans_rng_context_restores_on_success_and_error():
    np.random.seed(81);torch.manual_seed(22)
    n=np.random.get_state();t=torch.get_rng_state().clone()
    values=[]
    for _ in range(2):
        with isolated_kmeans_rng('identity',torch.device('cpu')):
            values.append((np.random.choice(100,19,False),torch.randint(100,(19,))))
    assert np.array_equal(values[0][0],values[1][0]) and torch.equal(values[0][1],values[1][1])
    assert np.array_equal(np.random.get_state()[1],n[1]) and torch.equal(t,torch.get_rng_state())
    with pytest.raises(RuntimeError):
        with isolated_kmeans_rng('identity',torch.device('cpu')):raise RuntimeError('probe')
    assert np.array_equal(np.random.get_state()[1],n[1]) and torch.equal(t,torch.get_rng_state())

@pytest.mark.parametrize('bad',[torch.zeros(1,1,4,4),torch.full((1,1,4,4),float('nan')),torch.full((1,1,4,4),-1.)])
def test_invalid_probabilities_fail_instead_of_silent_fallback(bad):
    with pytest.raises(ValueError):sample_cpu(bad,'a')

def test_one_target_float32_and_future_free_API():
    with pytest.raises(ValueError):sample_cpu(torch.ones(2,1,4,4),'a')
    with pytest.raises(ValueError):sample_cpu(torch.ones(1,1,4,4,dtype=torch.float64),'a')
    assert set(inspect.signature(Predictor.predict).parameters)=={'self','sample_id','history','image','semantic'}
    raw=dict(sample_id='a',xy=torch.arange(40.).reshape(20,2));changed=dict(sample_id='a',xy=raw['xy'].clone())
    changed['xy'][8:]=1e6
    assert torch.equal(history_record(raw)[1],history_record(changed)[1])

def test_original_KMeans_empty_cluster_random_fallback_is_scoped():
    # All128 proposals coincide:18 of19 initial clusters are empty, forcing
    # the author's torch.randint fallback while leaving centroids well-defined.
    points=torch.zeros(128,2)
    results=[]
    for _ in range(5):
        torch.rand(3);np.random.rand(3)  # Different caller states each time.
        before_t=torch.get_rng_state().clone();before_n=np.random.get_state()
        results.append(cluster_proposals(points,'empty-cluster-case'))
        assert torch.equal(before_t,torch.get_rng_state())
        assert np.array_equal(before_n[1],np.random.get_state()[1])
    assert all(torch.equal(r,torch.zeros(19,2)) for r in results)
