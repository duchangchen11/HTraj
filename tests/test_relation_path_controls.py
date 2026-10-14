import numpy as np
import pytest
import torch
from src.evaluation.path_identity import path_identity_counts
from src.evaluation.interaction_coverage import count_ordered_walks
from src.evaluation.relation_controls import CONTROL_MODES,altered_frames,control_counts,graph_control
from src.models.order_verification import build_order_model


def adjacency(events,n=4,steps=5):
    a=np.zeros((steps,n,n),dtype=bool)
    for t,pairs in events.items():
        for s,i in pairs:
            a[t,i,s]=True
    c=np.ones((steps,n),dtype=bool); c[0]=False
    return a,c


@pytest.mark.parametrize('mode',['strict','nondecreasing'])
def test_identity_counts_unique_chain_and_two_person_backtrack(mode):
    a,c=adjacency({1:[(0,1)],2:[(1,2)],3:[(2,3)]})
    stats=path_identity_counts(a,c,mode)
    assert stats['q3_total'][3]==1 and stats['q3_four_distinct'][3]==1
    assert stats['q2_three_distinct'][2]==1
    a,c=adjacency({1:[(0,1)],2:[(1,0)],3:[(0,1)]},n=2)
    stats=path_identity_counts(a,c,mode)
    assert stats['q3_total'][1]==1 and stats['q3_backtrack'][1]==1
    assert stats['q3_repeated'][1]==1 and stats['q3_four_distinct'][1]==0
    assert stats['q2_return'][0]==1


def test_three_cycle_and_multiple_distinct_middle_people():
    a,c=adjacency({1:[(0,1),(0,2)],2:[(1,3),(2,3)],3:[(3,0)]})
    stats=path_identity_counts(a,c)
    assert stats['q3_end_return'][0]==2 and stats['q3_backtrack'][0]==0
    assert stats['q2_nonreturn_endpoint_pairs'][3]==1
    assert stats['q2_nonreturn_pairs_multiple_middles'][3]==1
    assert stats['q2_nonreturn_distinct_middle_sum'][3]==2


def test_same_frame_and_reset_walk_counts_match_original_strict_oracle():
    a,c=adjacency({1:[(0,1),(1,2),(2,3)]})
    assert path_identity_counts(a,c)['q3_total'][3]==0
    assert path_identity_counts(a,c,'nondecreasing')['q3_total'][3]==1
    rng=np.random.default_rng(1726)
    a=rng.random((8,6,6))<.3
    a[:,np.arange(6),np.arange(6)]=False
    c=rng.random((8,6))>.1; c[0]=False
    a &= c[:,:,None] & c[:,None,:]
    old=count_ordered_walks(a,c)['final_counts']
    new=path_identity_counts(a,c)
    for q in range(3):
        np.testing.assert_array_equal(old[q],new[f'q{q+1}_total'])
    np.testing.assert_array_equal(new['q3_total'],new['q3_repeated']+new['q3_four_distinct'])
    assert np.all(new['q3_backtrack']>=0)


@pytest.mark.parametrize('mode',CONTROL_MODES)
def test_graph_controls_history_legality_degree_constraints_and_strength(mode):
    model=build_order_model('TOHI-H3',hidden_dim=16)
    # Spread enough people for nonlocal, budget-preserving controls to differ.
    x=torch.randn(2,1,15,2)*2+torch.randn(2,8,15,2).cumsum(1)*.02
    valid=torch.ones(2,8,15,dtype=torch.bool); valid[:,:4,-1]=False
    encoded=model.encoder(x,valid)
    real=model.graph.sequence(encoded)
    altered=altered_frames(model,encoded,real,mode)
    counts=control_counts(encoded,real,altered)
    assert counts['control_edges'] == sum(len(f.source) for f in altered)
    for f in altered:
        assert len(set(zip(f.source.tolist(),f.target.tolist())))==len(f.source)
        assert torch.isfinite(f.features).all() and torch.isfinite(f.weight).all()
    if mode in ['full','radius_max_change','scene_random','scene_max_change','degree_swap_global']:
        assert counts['incoming_degree_mismatch_nodes']==0 and counts['control_edges']==counts['real_edges']
    if mode in ['full','degree_swap_global']:
        assert counts['outgoing_degree_mismatch_nodes']==0
    if mode in ['full','radius_max_change','cut_indirect_paths','matched_random_delete']:
        assert counts['control_outside_radius_edges']==0
    if mode=='radius_max_change':
        assert counts['added_edges']==counts['possible_local_changed_edges']
    if mode in ['scene_max_change','degree_swap_global']:
        assert counts['added_edges']>0
    if mode=='cut_indirect_paths':
        _,mask=model.order_state(encoded['hidden'],altered,encoded['continuity'])
        assert not mask[:,:,:,1:].any()
    if mode=='matched_random_delete':
        cut=altered_frames(model,encoded,real,'cut_indirect_paths')
        for a,b in zip(altered,cut):
            torch.testing.assert_close(torch.bincount(a.target,minlength=30),torch.bincount(b.target,minlength=30))


def test_graph_override_restores_original_and_has_no_weight_or_ego_change():
    model=build_order_model('TOHI-H3',hidden_dim=16)
    x=torch.randn(1,8,12,2)*.5
    before={k:v.clone() for k,v in model.state_dict().items()}
    pred=model(x)
    with graph_control(model,'remove_all_messages') as trace:
        other=model(x)
        assert trace['real'] and all(len(f.source)==0 for f in trace['altered'])
    assert 'sequence' not in model.graph.__dict__
    torch.testing.assert_close(pred,model(x),rtol=0,atol=0)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert (pred-other).abs().max()>0
