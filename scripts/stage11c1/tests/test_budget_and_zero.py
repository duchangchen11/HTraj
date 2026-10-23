"""Safety boundaries and effective semantic-input connections, using CPU only."""
import pytest
import torch
from scripts.stage11c1.common import Budget,zero_semantic,shapes,history_records

def test_safe_budget_boundary_and_reserve():
    clock=[0.]
    b=Budget(prior=16000,clock=lambda:clock[0])
    b.guard(679)
    with pytest.raises(TimeoutError):b.guard(680)
    clock[0]=1
    with pytest.raises(TimeoutError):b.guard(679)
    with pytest.raises(ValueError):b.guard(-1)

def test_full_canvas_zero_and_padding_for_all30_scenes():
    table=shapes();assert len(table)==30
    for scene,row in table.items():
        z=zero_semantic(scene,'cpu',table)
        assert list(z.shape)==row['semantic_shape'] and z.dtype==torch.float32
        assert torch.count_nonzero(z)==0
        assert z[...,row['valid_height']:, :].count_nonzero()==0
        assert z[...,:,row['valid_width']:].count_nonzero()==0

def test_semantic_connections_zero_gradient_history_active():
    torch.manual_seed(17)
    conv=torch.nn.Conv2d(14,32,3,padding=1)
    x=torch.cat([torch.zeros(2,6,8,9),torch.rand(2,8,8,9)],1)
    conv(x).square().mean().backward()
    assert torch.count_nonzero(conv.weight.grad[:,:6])==0
    assert conv.weight.grad[:,:6].numel()==1728
    assert torch.count_nonzero(conv.weight.grad[:,6:])>0

def test_history_cache_has_no_future_field():
    rows=history_records('validation')
    assert len(rows)==998 and len({r['sample_id'] for r in rows})==998
    assert all(set(r)=={'sample_id','scene','history'} and r['history'].shape==(8,2) for r in rows)
