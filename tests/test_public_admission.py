import numpy as np
import pytest
import torch
from src.public_admission.inputs import history_array,agentformer_packet,pecnet_packet,synthetic_scene


@pytest.mark.parametrize('n',[1,2,5,12])
def test_history_only_packet_and_permutation(n):
    h,y=synthetic_scene(n);p=np.arange(n)[::-1].copy()
    a=agentformer_packet(h);b=agentformer_packet(h[p])
    assert all(torch.equal(a['pre_motion_3D'][i],b['pre_motion_3D'][int(np.where(p==i)[0][0])]) for i in range(n))
    assert all(not v.any() for v in a['fut_motion_3D']+a['fut_motion_mask'])
    x,pos,mask=pecnet_packet(h);px,pp,pm=pecnet_packet(h[p])
    torch.testing.assert_close(px,x[p]);torch.testing.assert_close(pp,pos[p]);torch.testing.assert_close(pm,mask[p][:,p])
    assert mask.diag().eq(1).all() and torch.equal(mask,mask.T)


def test_units_and_coordinate_restore():
    h,y=synthetic_scene(2);x,pos,mask=pecnet_packet(h)
    restored=x.reshape(2,8,2).numpy()/1.86+h[:,:1]
    np.testing.assert_allclose(restored,h)
    np.testing.assert_allclose(pos.numpy(),h[:,-1]/1000)
    np.testing.assert_allclose(np.diff(h,axis=1)/.4,np.repeat((h[:,1]-h[:,0])[:,None]/.4,7,axis=1))


@pytest.mark.parametrize('h',[np.zeros((1,20,2)),np.zeros((0,8,2)),np.full((1,8,2),np.nan),np.full((1,8,2),np.inf)])
def test_future_packets_missing_agents_and_nonfinite_rejected(h):
    with pytest.raises(ValueError):history_array(h)


def test_relation_mask_history_only_and_no_cross_scene_grouping():
    h,_=synthetic_scene(2);h[1]+=100
    _,_,mask=pecnet_packet(h)
    assert mask.tolist()==[[1.,0.],[0.,1.]]
    # The helper accepts one explicit synchronous scene, never flattens a
    # batch of scenes or consults any future trajectory.
    with pytest.raises(ValueError):pecnet_packet(np.stack([h,h]))
