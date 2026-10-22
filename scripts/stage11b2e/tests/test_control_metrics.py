"""Synthetic unit checks only: these are not validation-set measurements."""
import numpy as np
from scripts.stage11b2e.metrics import path_metrics,goal_metrics,gate

def test_ADE_and_FDE_oracles_are_independent_heads():
    x=np.full((20,12,2),100.,dtype=np.float32);x[:,:,1]=0
    x[0,:,0]=0;x[0,-1,0]=10;x[1,:,0]=2
    m,_,_=path_metrics(x,np.zeros((12,2)))
    assert m['ADE_best_head']==0 and m['FDE_best_head']==1
    assert m['minADE20']==10/12 and m['minFDE20']==2

def test_probability_union_does_not_double_count_duplicate_goals():
    goals=np.zeros((20,2));points=np.array([[0,0],[4,0],[0,4],[4,4]])
    m,_=goal_metrics(goals,[0,0],points,np.ones((2,2)),radii=(0,4))
    assert m['ProbabilityUnion0']==.25 and m['ProbabilityUnion4']==.75
    assert m['UniqueGoals']==1

def test_all_four_success_conditions_must_hold_simultaneously():
    a=dict(GoalCover25=.8,minADE20=100.,minFDE20=100.)
    b=dict(GoalCover25=.82,minADE20=98.,minFDE20=101.)
    other=dict(GoalCover25=.7);improved=dict(GoalCover25=.71)
    assert gate(a,b,other,improved)['status']=='CONTROL_GO'
    for field,value in [('GoalCover25',.809),('minADE20',99.5),('minFDE20',102.)]:
        bad=dict(b);bad[field]=value
        assert gate(a,bad,other,improved)['status']=='CONTROL_NO_GO'
    assert gate(a,b,other,other)['status']=='CONTROL_NO_GO'
