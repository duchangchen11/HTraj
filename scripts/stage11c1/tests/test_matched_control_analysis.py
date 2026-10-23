"""CPU synthetic structure checks only; not experimental prediction evidence."""
import numpy as np
import pandas as pd
import pytest
from scripts.stage11c1.analyze import (metrics_numpy, paired_frame, semantic_gate,
                                      summarize, choose_cases, a25_rows, SCENES, METRICS)


def synthetic_paired(sem=9., nosem=10.):
    rows=[]
    for i,scene in enumerate(SCENES):
        for j in range(i+1):
            row=dict(sample_id=f'{scene}:{j}',scene=scene,person_id=j,comparison='FIXED_EPOCH30',NoSemantic_epoch=30)
            for name,value in [('Semantic',sem),('NoSemantic',nosem)]:
                row.update({f'{name}_{k}':value for k in METRICS})
                row[f'{name}_GoalCover25']=1.
            row.update(Delta_ADE=nosem-sem,Delta_FDE=nosem-sem,Delta_GoalCover=0.,
                       Semantic_A25_failed=False,NoSemantic_A25_failed=False)
            rows.append(row)
    return pd.DataFrame(rows)


def test_independent_oracle_heads_and_full_shape():
    p=np.full((20,1,12,2),100.)
    p[0]=10.
    p[1]=1.;p[1,0,-1]=2.
    p[2]=5.;p[2,0,-1]=0.
    g=p[:,:,-1].copy();y=np.zeros((1,12,2))
    m,ade,fde,ge=metrics_numpy(p,g,y)
    assert m.best_ADE_head.iloc[0]==1
    assert m.best_FDE_head.iloc[0]==2
    assert m.minADE20.iloc[0]==pytest.approx(ade[:,0].min())
    assert m.minFDE20.iloc[0]==0
    assert ade.shape==fde.shape==ge.shape==(20,1)
    assert m.GoalCover25.iloc[0]==1


def test_missing_candidates_or_nonfinite_rejected():
    with pytest.raises(ValueError):
        metrics_numpy(np.zeros((19,1,12,2)),np.zeros((19,1,2)),np.zeros((1,12,2)))
    p=np.zeros((20,1,12,2));p[0,0,0,0]=np.nan
    with pytest.raises(ValueError):
        metrics_numpy(p,np.zeros((20,1,2)),np.zeros((1,12,2)))


def test_gate_semantic_sign_and_no_semantic_denominator():
    frame=synthetic_paired(9.,10.);result=semantic_gate(frame)
    assert result['status']=='SEMANTIC_BENEFIT_PRELIMINARY_GO'
    assert result['measurements']['macro_ADE_relative_semantic_gain']==pytest.approx(.1)
    assert semantic_gate(synthetic_paired(10.,9.))['status']=='SEMANTIC_BENEFIT_INCONCLUSIVE_OR_NO_GO'


def test_gate_requires_all_five_recordings():
    frame=synthetic_paired()
    assert semantic_gate(frame[frame.scene!=SCENES[-1]])['status']=='NOT_EVALUABLE'


def test_window_recording_and_person_averages_separated():
    f=synthetic_paired();f.loc[f.scene==SCENES[-1],'Semantic_minADE20']=100
    window=summarize(f,'ALL998','target_window')['Semantic_minADE20']
    macro=summarize(f,'FIVE_RECORDING_MACRO','recording_equal')['Semantic_minADE20']
    assert window!=macro
    # Repeated windows of the same person change window average, not ID equal average.
    copy=f.iloc[[-1]].copy();copy['sample_id']='repeated_window'
    dup=pd.concat([f,copy],ignore_index=True)
    assert summarize(f,'all','person_ID_equal')['Semantic_minADE20']==summarize(dup,'all','person_ID_equal')['Semantic_minADE20']


def test_stable_case_selection_is_lexicographic_and_order_independent():
    f=synthetic_paired();f.loc[:,'Delta_ADE']=2.
    a=choose_cases(f);b=choose_cases(f.sample(frac=1,random_state=3))
    assert a==b
    assert a[0]['sample_id']==sorted(f.sample_id)[0]
    assert any(row['status']=='NO_ELIGIBLE_CASE' for row in a)


def test_failure_cohorts_partition_without_dropping_losses():
    f=synthetic_paired();f.loc[0,['Semantic_A25_failed','NoSemantic_A25_failed']]=True
    f.loc[1,'Semantic_A25_failed']=True
    f.loc[2,'NoSemantic_A25_failed']=True
    r=pd.DataFrame(a25_rows(f));allr=r[r.scope=='ALL998'].set_index('cohort')
    cohorts=['both_failed','semantic_only_failed_recovered_by_NoSemantic','NoSemantic_only_failed_newly_lost','both_successful']
    assert int(allr.loc[cohorts,'targets'].sum())==len(f)
    assert allr.loc['semantic_only_failed_recovered_by_NoSemantic','targets']==1
    assert allr.loc['NoSemantic_only_failed_newly_lost','targets']==1


def test_paired_alignment_cannot_rely_on_row_order():
    ids=pd.DataFrame(dict(sample_id=['a','b'],scene=['s','s'],person_id=[1,2]))
    a=pd.DataFrame({k:[1.,2.] for k in METRICS});a['sample_id']=['a','b']
    b=a.iloc[::-1].copy()
    for key in ('best_ADE_head','best_FDE_head','best_Goal_head'):
        a[key]=0;b[key]=0
    assert (paired_frame(ids,a,b,'test',30).Delta_ADE==0).all()
    with pytest.raises(ValueError):
        paired_frame(ids,a,b.iloc[:1],'test',30)
