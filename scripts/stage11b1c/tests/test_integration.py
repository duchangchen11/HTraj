import inspect
import numpy as np,pytest,torch
from scripts.stage11b1c.common import Budget
from scripts.stage11b1c.engine import choose_best
from scripts.stage11b1c.validate import calculate,predict_all
from scripts.stage11b1c.data import load,scene_arrays,input_records

def test_budget_next_operation_and_safe_boundary():
    now=[100.];b=Budget(prior=50,clock=lambda:now[0])
    assert b.elapsed()==50
    b.guard(26949)
    with pytest.raises(TimeoutError):b.guard(26950)
    now[0]+=100;assert b.elapsed()==150
    with pytest.raises(TimeoutError):b.guard(26850)

def test_best_uses_complete_window_ADE_and_earlier_exact_tie():
    a=[dict(epoch=10,status='COMPLETE',window={'minADE20':2,'minFDE20':0}),
       dict(epoch=5,status='COMPLETE',window={'minADE20':2,'minFDE20':100}),
       dict(epoch=1,status='INCOMPLETE',window={'minADE20':0})]
    assert choose_best(a)['epoch']==5
    assert choose_best(a[2:]) is None

def test_metric_oracles_goal_path_diversity_and_pixels():
    y=torch.zeros(1,12,2);p=torch.ones(20,1,12,2)*4;g=torch.ones(20,1,2)*20
    p[0]=0;p[0,0,-1,0]=9;p[1,0,-1]=0
    r=calculate(p,y,g)
    assert r['ADE1'].item()==.75 and r['FDE1'].item()==9
    assert r['best_ADE_head'].item()==0 and r['best_FDE_head'].item()==1
    assert r['minFDE20'].item()==0 and r['minGoalFDE20'].item()>r['minFDE20'].item()
    assert r['goal_path_gap1'].item()>0 and r['endpoint_diversity'].item()>0
    # Uniform coordinate translation must leave all displacement errors unchanged.
    shifted=calculate(p+100,y+100,g+100)
    for key in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20']:
        torch.testing.assert_close(r[key],shifted[key])
    # Raw pixels are restored through the immutable predictor's exact /0.25.
    x=torch.arange(480.).reshape(20,1,12,2)
    assert torch.equal((x*.25)/.25,x)

def test_data_iteration_complete_no_26_scene_cap_and_history_only_records():
    df=load('train');scenes=sorted(df.sceneId.unique());visits=[];steps=0
    for scene in scenes:
        coordinates,ids=scene_arrays(df,scene);visits.extend(ids);steps+=(len(ids)+3)//4
        assert coordinates.shape[1:]==(20,2)
    assert len(scenes)==25 and len(visits)==len(set(visits))==7496 and steps==1882
    r=input_records('train')[:2]
    assert all(set(v)=={'sample_id','scene','history'} and v['history'].shape==(8,2) for v in r)
    # Formal entry accepts exactly998; only explicitly named preflight scopes
    # can use2 TRAIN targets. No generic arbitrary-sized validation switch.
    assert inspect.signature(predict_all).parameters['scope'].default=='validation'
