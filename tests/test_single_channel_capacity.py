import torch
from src.models.order_verification import build_order_model


def test_single_order_scorer_parameters_are_functionally_inactive():
    model=build_order_model('TOHI-H1')
    assert sum(p.numel() for p in model.fusion.scores.parameters())==8321
    x=torch.randn(1,8,5,2).cumsum(1)*.02
    pred,debug=model(x,return_debug=True)
    active=debug['order_support'][:,-1,: ,0]
    assert debug['order_weights'][:,:,0][active].eq(1).all()
    pred.square().mean().backward()
    assert all(p.grad is not None and p.grad.eq(0).all() for p in model.fusion.scores.parameters())
    original=pred.detach().clone()
    with torch.no_grad():
        for p in model.fusion.scores.parameters():
            p.fill_(100)
    torch.testing.assert_close(original,model(x),rtol=0,atol=0)
