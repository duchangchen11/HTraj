import numpy as np
from scripts.stage11c0.statistics import compare_distribution,compare_goals,compare_fixed_paths,prediction_metrics
def test_JS_actual_categorical_normalization_and_zero_probability():
    a=dict(normalized_weights=np.array([2.,0.]),logits=np.zeros(2))
    b=dict(normalized_weights=np.array([0.,5.]),logits=np.ones(2))
    x=compare_distribution(a,b);assert abs(x['JS_nats']-np.log(2))<1e-14 and x['total_variation']==1
    assert compare_distribution(a,a)['JS_nats']==0
    assert x['Original_support_pixels']==1 and x['Original_support_area_raw_px2']==16
def test_goal_set_distance_independent_of_cluster_head_permutation():
    a=np.arange(40).reshape(20,2).astype(float);b=a[::-1].copy()
    s=compare_goals(a,b);assert s['Hungarian_mean_distance']==0 and s['symmetric_Chamfer_distance']==0
    assert s['Softargmax_displacement']>0
def test_fixed_goal_paths_bitwise_counts_and_pixel_displacements():
    a=np.zeros((20,12,2));b=a.copy();b[0,:,0]=3;b[0,:,1]=4
    x=compare_fixed_paths(a,b);assert x['Path_changed_heads']==1 and x['Path_unchanged_heads']==19
    assert x['Path_point_mean_displacement']==.25 and x['Head0_endpoint_displacement']==5
def test_ADE_FDE_independent_oracles_and_Goal_different_from_Path_end():
    p=np.full((20,12,2),100.);p[0]=0;p[0,-1,0]=10;p[1,:,0]=2;p[1,:,1]=0
    g=np.full((20,2),50.);m,ade,fde,gd=prediction_metrics(p,g,np.zeros((12,2)))
    assert ade.argmin()==0 and fde.argmin()==1 and m['minADE20']==10/12 and m['minFDE20']==2
    assert m['minGoalFDE20']>m['minFDE20']
