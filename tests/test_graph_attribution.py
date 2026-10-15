import pytest
import torch
from src.graph_attribution.model import GraphAttributionModel,PersonTemporalNorm,window_nll,gaussian_predictions
from src.graph_attribution.data import real_relation,controlled_relation,pack
from src.public_training.model import metrics


@pytest.mark.parametrize('people',[2,3,5,8,12])
@pytest.mark.parametrize('kind',['G0','G1'])
def test_permutation_output_position_and_gradient(people,kind):
    torch.manual_seed(17);model=GraphAttributionModel(kind)
    x=torch.randn(1,2,8,people,requires_grad=True);positions=torch.randn(8,people,2)
    a=real_relation(positions)[None];valid=torch.ones(1,people,dtype=torch.bool)
    p=torch.randperm(people);inverse=torch.argsort(p)
    y=model(x,a,valid);loss=y.square().sum();gx,*gp=torch.autograd.grad(loss,[x,*model.parameters()])
    xx=x.detach()[...,p].requires_grad_();aa=a[:,:,p][:,:,:,p]
    yy=model(xx,aa,valid[:,p]);recovered=yy[:,:,inverse]
    torch.testing.assert_close(recovered,y,atol=3e-6,rtol=3e-5)
    last=positions[-1][None];torch.testing.assert_close(y[...,:2].cumsum(1)+last[:,None],recovered[...,:2].cumsum(1)+last[:,None],atol=2e-5,rtol=3e-5)
    gxx,*gpp=torch.autograd.grad(yy.square().sum(),[xx,*model.parameters()])
    torch.testing.assert_close(gxx[...,inverse],gx,atol=3e-5,rtol=3e-4)
    for first,second in zip(gp,gpp):torch.testing.assert_close(first,second,atol=1e-4,rtol=3e-4)


@pytest.mark.parametrize('kind',['G0','G1'])
def test_graph_dependency_and_edges(kind):
    model=GraphAttributionModel(kind).eval();x=torch.randn(1,2,8,3,requires_grad=True)
    valid=torch.ones(1,3,dtype=torch.bool);a=torch.zeros(1,8,3,3);a[:,:,0,1]=1
    y=model(x,a,valid);gradient=torch.autograd.grad(y[:,:,0].sum(),x)[0]
    changed=model(x,torch.zeros_like(a),valid)
    if kind=='G0':
        assert torch.equal(gradient[...,1:],torch.zeros_like(gradient[...,1:]));assert torch.equal(y,changed)
    else:
        assert gradient[...,1].abs().max()>1e-6;assert (y-changed).abs().max()>1e-5
    assert torch.equal(gradient[...,2],torch.zeros_like(gradient[...,2]))


def test_same_weights_no_neighbors_exactly_equal_and_single_person():
    torch.manual_seed(17);ego=GraphAttributionModel('G0')
    torch.manual_seed(17);social=GraphAttributionModel('G1')
    assert all(torch.equal(a,b) for a,b in zip(ego.parameters(),social.parameters()))
    for n in (1,2,5):
        x=torch.randn(1,2,8,n);a=torch.zeros(1,8,n,n);valid=torch.ones(1,n,dtype=torch.bool)
        assert torch.equal(ego(x,a,valid),social(x,a,valid))
    assert sum(p.numel() for p in ego.parameters())==sum(p.numel() for p in social.parameters())


def test_padding_batch_isolation_and_mode_consistency():
    model=GraphAttributionModel('G1');x=torch.randn(2,2,8,8,requires_grad=True);valid=torch.zeros(2,8,dtype=torch.bool);valid[:,:3]=True
    a=torch.zeros(2,8,8,8);a[:,:,0,1]=1
    train=model.train()(x,a,valid);test=model.eval()(x,a,valid);assert torch.equal(train,test)
    assert torch.equal(test[:,:,3:],torch.zeros_like(test[:,:,3:]))
    g=torch.autograd.grad(test[0].sum(),x)[0];assert torch.equal(g[1],torch.zeros_like(g[1]))
    torch.testing.assert_close(model(x[:1,:,:,:3],a[:1,:,:3,:3],valid[:1,:3]),test[:1,:,:3],atol=1e-6,rtol=1e-5)
    norm=PersonTemporalNorm(16);features=torch.randn(2,16,8,5)
    assert torch.equal(norm.train()(features),norm.eval()(features))


def test_history_graph_future_is_not_an_argument_and_control_legal_equivariant():
    x=torch.randn(8,5,2)*3;ids=[11,12,14,19,25];real=real_relation(x)
    future=torch.full((12,5,2),float('nan'));assert torch.equal(real,real_relation(x))
    control=controlled_relation(real,ids,'synthetic:10');p=torch.tensor([3,1,4,0,2]);inv=torch.argsort(p)
    changed=controlled_relation(real[:,p][:,:,p],[ids[i] for i in p],'synthetic:10')
    assert torch.equal(changed[:,inv][:,:,inv],control)
    assert not control.diagonal(dim1=-2,dim2=-1).any()
    assert torch.equal((real>0).sum(-1),(control>0).sum(-1))
    torch.testing.assert_close(real.sum(-1),control.sum(-1),atol=1e-6,rtol=1e-6)


def test_nll_matches_frozen_official_objective_and_valid_gaussian():
    model=GraphAttributionModel('G1');x=torch.randn(2,2,8,5);valid=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    a=torch.zeros(2,8,5,5);a[:,:,0,1]=1;p=model(x,a,valid);truth=torch.randn(2,12,5,2)
    actual=window_nll(p,truth,valid)
    expected=torch.stack([metrics.bivariate_loss(p[b,:,valid[b]],truth[b,:,valid[b]]) for b in range(2)])
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    actual.mean().backward();assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    mu,samples=gaussian_predictions(p,torch.zeros(2,5,2),torch.zeros(2,20,12,5,2))
    assert mu.shape==(2,12,5,2) and samples.shape==(2,20,12,5,2) and torch.isfinite(samples).all()
    assert torch.equal(samples,mu[:,None].expand_as(samples))
    torch.testing.assert_close(mu,p[...,:2].cumsum(1),atol=2e-6,rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compatibility check')
def test_deterministic_cuda_gaussian_integration():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    prior=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        parameters=torch.randn(2,12,5,5,device='cuda')*.1;last=torch.zeros(2,5,2,device='cuda');noise=torch.randn(2,20,12,5,2,device='cuda')
        mean,samples=gaussian_predictions(parameters,last,noise)
        again,draws=gaussian_predictions(parameters,last,noise)
        assert torch.equal(mean,again) and torch.equal(samples,draws)
        torch.testing.assert_close(mean.cpu(),parameters[...,:2].cpu().cumsum(1),atol=1e-6,rtol=1e-6)
    finally:torch.use_deterministic_algorithms(prior)


def test_future_label_change_does_not_change_inputs_and_padded_loss():
    from copy import deepcopy
    samples=[]
    for n in (2,5):
        obs=torch.randn(8,n,2);a=real_relation(obs)
        samples.append(dict(observed=obs,future=torch.randn(12,n,2),agent_ids=list(range(n)),relation=a,control=a))
    changed=deepcopy(samples)
    for s in changed:s['future'].fill_(999999.)
    b=pack(samples,'G1','cpu');c=pack(changed,'G1','cpu')
    for key in ('v','relation','last','valid'):assert torch.equal(b[key],c[key])
    model=GraphAttributionModel('G1');parameters=model(b['v'],b['relation'],b['valid']);loss=window_nll(parameters,b['velocity'],b['valid'])
    for i,s in enumerate(samples):
        single=pack([s],'G1','cpu');p=model(single['v'],single['relation'],single['valid'])
        torch.testing.assert_close(loss[i],window_nll(p,single['velocity'],single['valid'])[0],atol=1e-6,rtol=1e-6)
