import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.stage11b2b.common import Budget,ROOT,CONF
from scripts.stage11b2b.metrics import verdict,candidate_metrics,aggregate
from scripts.stage11b1c.engine import choose_best

@pytest.mark.parametrize('a,b,c,result',[(.1,.5,True,'H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET'),(.099,.6,True,'INCONCLUSIVE'),(.2,.499,True,'INCONCLUSIVE'),(.049,.249,True,'H1_WEAKENED_WITHIN_BUDGET'),(.05,.2,True,'INCONCLUSIVE'),(.04,.25,True,'INCONCLUSIVE'),(0,0,False,'NOT_EVALUATED')])
def test_frozen_hypothesis_boundaries(a,b,c,result):assert verdict(a,b,c)==result

def test_strict_goal_miss_and_same_head():
    y=np.zeros((2,12,2));p=np.zeros((20,2,12,2));g=np.ones((20,2,2))*100
    p[:,0,:,0]=30;g[0,0]=[10,0];g[:,1]=[25,0]
    *_,m=candidate_metrics(p,y,g);f=pd.DataFrame(m);a=aggregate(f)
    assert a['A25_miss_count']==0 and a['A10_miss_count']==1 and a['GoalPathMismatch_count']==1

def test_budget_full_parent_elapsed_and_safety():
    b=Budget(100,clock=lambda:15000,started=200)
    with pytest.raises(TimeoutError):b.guard(100)

def test_checkpoint_choice_only_complete_and_tie_earlier():
    r=[dict(status=s,epoch=e,window=dict(minADE20=a)) for s,e,a in [('INCOMPLETE',50,0),('COMPLETE',40,2),('COMPLETE',30,2)]]
    assert choose_best(r)['epoch']==30

def test_protocol_no_hyperparameter_change():
    old=json.loads((ROOT/'configs/stage11b1c/train.json').read_text());new=json.loads((CONF/'train.json').read_text())
    for k in ['seed','history','future','resize','learning_rate','optimizer','target_batch_size','loss','augmentation','AMP','scheduler','TF32','deterministic_algorithms','artifacts']:assert new[k]==old[k]
    assert new['start_epoch']==31 and new['max_epochs']==50 and new['validation_epochs']==[35,40,45,50]
    assert (CONF/'evaluation.json').read_bytes()==(ROOT/'configs/stage11b1c/evaluation.json').read_bytes()

def test_future_not_in_predict_all_schema():
    from scripts.stage11b2b.validate import predict_all
    with pytest.raises(AssertionError):predict_all(None,None,None,[dict(sample_id='x',scene='x',history=[],GT=[])]*998,35,{},None)

def test_old_fixed_cohorts():
    f=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv')
    assert len(f)==998 and (f.minGoalFDE20>25).sum()==164 and f.GOAL_PATH_MISMATCH.sum()==7
    assert np.isclose(f.loc[f.minGoalFDE20>25,'minADE20'].sum()/f.minADE20.sum(),.610817,atol=1e-6)
