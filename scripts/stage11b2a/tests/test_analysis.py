import numpy as np
from scripts.stage11b2a.features import motion,turn_group,speed_change_group,tercile,spearman
from scripts.stage11b2a.analysis import candidate_metrics

def test_complete_stationary_history_does_not_invent_direction():
    f=motion(np.zeros((8,2)))
    assert f['observed_positions']==8 and f['reliable_motion_steps']==0
    assert f['all_steps_stationary'] and np.isnan(f['max_turn_degrees']) and np.isnan(f['straightness'])
    assert turn_group(f)=='undefined_direction' and speed_change_group(f)=='undefined_low_speed'

def test_actual_dt_and_future_include_last_history_position():
    h=np.column_stack([np.arange(8)*2,np.zeros(8)]);f=motion(h)
    assert f['mean_speed']==5 and f['net_displacement']==14 and f['path_length']==14
    future=np.column_stack([np.arange(8,20)*2,np.zeros(12)])
    g=motion(np.vstack([h[-1],future]));assert g['path_length']==24 and g['observed_positions']==13 and g['mean_speed']==5
    assert turn_group(f)=='straight' and speed_change_group(f)=='stable_CVle0.15'

def test_turn_and_translation_rotation_invariance():
    h=np.array([[0,0],[2,0],[4,0],[4,2],[4,4],[4,6],[4,8],[4,10]],dtype=float)
    a=motion(h);assert a['max_turn_degrees']==90 and turn_group(a)=='turning_ge30deg'
    R=np.array([[0,-1],[1,0]])
    b=motion(h@R.T+1000)
    for k in ['net_displacement','path_length','mean_speed','mean_acceleration','max_turn_degrees','straightness']:
        np.testing.assert_allclose(a[k],b[k],atol=1e-10)

def test_subpixel_jitter_is_not_reliable_turning():
    h=np.array([[0,0],[.1,0],[0,0],[.1,0],[0,0],[.1,0],[0,0],[.1,0]])
    a=motion(h);assert a['valid_turn_pairs']==0 and np.isnan(a['max_turn_degrees'])
    assert a['all_steps_stationary'] and turn_group(a)=='undefined_direction'

def test_three_different_oracle_indices_and_nonnegative_gap():
    y=np.zeros((1,12,2));p=np.full((20,1,12,2),100.)
    p[0]=0;p[0,0,-1,0]=9;p[1]=2;p[1,0,-1]=0
    g=np.full((20,1,2),100.);g[2]=0
    _,_,_,_,_,m=candidate_metrics(p,y,g)
    assert m['best_ADE_head'][0]==0 and m['best_FDE_head'][0]==1 and m['closest_goal_head'][0]==2
    assert m['SelectionGap_ADE'][0]==0 and m['SelectionGap_FDE'][0]==9
    assert m['closest_goal_ADE'][0]>100 and m['minGoalFDE20'][0]==0

def test_ties_keep_original_smallest_head_and_error_translation_invariant():
    y=np.zeros((2,12,2));p=np.ones((20,2,12,2));g=np.ones((20,2,2))
    a=candidate_metrics(p,y,g)[-1];b=candidate_metrics(p+100,y+100,g+100)[-1]
    assert (a['best_ADE_head']==0).all()
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-10)

def test_linear_quantile_bin_boundary_and_rank_ties():
    assert tercile(1,[1,2])=='medium' and tercile(2,[1,2])=='high'
    rho,n=spearman([0,0,1,2,np.nan],[0,0,2,3,4]);assert n==4 and np.isclose(rho,1)
    rho,n=spearman([0,0,0],[1,2,3]);assert np.isnan(rho) and n==3
