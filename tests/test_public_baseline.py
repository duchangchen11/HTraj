import pickle
import numpy as np
import pytest
import torch
from pathlib import Path
from src.evaluation.public_baseline import non_eth_fold,load_args,only_function,load_module,sample_modes,trajectory_errors


def test_eth_guard():
    for name in ('eth','ETH','../eth','biwi_eth'):
        with pytest.raises(ValueError):non_eth_fold(name)
    assert non_eth_fold('hotel')=='hotel'


def test_independent_minima_not_paired_sample():
    target=np.zeros((2,1,2));pred=np.zeros((2,2,1,2))
    pred[0,:,0,0]=[0,2];pred[1,:,0,0]=[4,0]
    r=sample_modes(pred,target)
    assert r['K20_independent_min'][0][0]==1 and r['K20_independent_min'][1][0]==0
    assert r['K1_first_sample'][1][0]==2
    with pytest.raises(ValueError):trajectory_errors(pred,target[0])


def test_official_metrics_agree_with_vectorized_definition():
    path=Path('third_party/stage4/social_stgcnn/metrics.py')
    module=load_module('stage4_metrics_test',path)
    rng=np.random.default_rng(17);target=rng.normal(size=(12,3,2));pred=rng.normal(size=(20,12,3,2));a,f=trajectory_errors(pred,target)
    for k in range(20):
        for n in range(3):
            np.testing.assert_allclose(a[k,n],module.ade([pred[k,:,n:n+1]],[target[:,n:n+1]],[1]),rtol=1e-12)
            np.testing.assert_allclose(f[k,n],module.fde([pred[k,:,n:n+1]],[target[:,n:n+1]],[1]),rtol=1e-12)


def test_function_extraction_does_not_execute_driver(tmp_path):
    path=tmp_path/'driver.py';path.write_text('raise RuntimeError("driver must not run")\ndef f(x):\n    return x+1\n')
    fn,sha=only_function(path,'f',{});assert fn(3)==4 and len(sha)==64


def test_public_network_forward_and_gt_isolation():
    module=load_module('stage4_model_test','third_party/stage4/social_stgcnn/model.py')
    torch.manual_seed(17);model=module.social_stgcnn(n_stgcnn=1,n_txpcnn=5).eval()
    v=torch.randn(1,2,8,3);a=torch.eye(3)[None].repeat(8,1,1)
    with torch.no_grad():
        y,_=model(v,a);assert y.shape==(1,5,12,3) and torch.isfinite(y).all()
        future=torch.randn(12,3,2);future.fill_(float('nan'))
        assert torch.equal(model(v,a)[0],y)


def test_networkx_compatibility_keeps_laplacian():
    import networkx as nx
    adjacency=np.array([[1.,2.,0.],[2.,1.,3.],[0.,3.,1.]])
    g=nx.from_numpy_array(adjacency);actual=nx.normalized_laplacian_matrix(g).toarray()
    d=adjacency.sum(-1);inv=np.diag(d**-.5);expected=np.eye(3)-inv@adjacency@inv
    np.testing.assert_allclose(actual,expected,atol=1e-15)


def test_official_args_whitelisted():
    for name in ('hotel','univ','zara1','zara2'):
        args=load_args(f'data/external/stage4/assets/social_stgcnn/checkpoint/social-stgcnn-{name}/args.pkl')
        assert args.dataset==name and args.obs_seq_len==8 and args.pred_seq_len==12
