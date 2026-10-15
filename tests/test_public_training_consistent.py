import torch
from src.public_training.model import make_model,forward
from src.public_training.model_consistent import make_model as corrected


def test_corrected_control_matches_original_train_transform_and_has_same_capacity():
    torch.manual_seed(17);old=make_model('NoSocial').train()
    torch.manual_seed(17);new=corrected('NoSocial-ConsistentNorm').train()
    x=torch.randn(1,2,8,5);a=torch.eye(5)[None].expand(8,-1,-1)
    torch.testing.assert_close(forward(old,x,a),forward(new,x,a),atol=0,rtol=0)
    assert sum(p.numel() for p in new.parameters())==7563


def test_corrected_control_train_eval_consistency_and_no_neighbor_dependency():
    torch.manual_seed(17);m=corrected('NoSocial-ConsistentNorm');x=torch.randn(1,2,8,5,requires_grad=True);a=torch.randn(8,5,5)
    train=forward(m.train(),x,a);test=forward(m.eval(),x,a)
    assert torch.equal(train,test)
    test[:,0].sum().backward();assert torch.equal(x.grad[...,1:],torch.zeros_like(x.grad[...,1:]))
    changed=x.detach().clone();changed[...,1:]+=100
    torch.testing.assert_close(forward(m,changed,a)[:,0],test[:,0],atol=1e-6,rtol=1e-6)
