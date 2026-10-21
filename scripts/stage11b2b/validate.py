"""All-target no-GT prediction phase, then separate offline scoring phase."""
from .common import *
from scripts.stage11b1c.data import input_records,truths,manifest
from scripts.stage11b1r.predictor import Predictor
from scripts.stage11b1.evaluation import metrics as original_metrics
from scripts.stage11b0.training_adapter import preprocess_scene,fixed_semantics
import torch,numpy as np,traceback

METRICS=['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20',
         'goal_path_gap1','goal_path_gap_mean20','endpoint_diversity','trajectory_diversity','near_duplicate_endpoint_fraction']

def calculate(paths,truth,goals):
    r=original_metrics(paths,truth,goals)
    out={k:r[k] for k in ['ADE1','FDE1','GoalFDE1','best_ADE_head','best_FDE_head']}
    out.update(minADE20=r['minADE_K'],minFDE20=r['minFDE_K'],minGoalFDE20=r['minGoalFDE_K'])
    gap=(goals-paths[:,:,-1]).norm(dim=-1)
    out.update(goal_path_gap1=gap[0],goal_path_gap_mean20=gap.mean(0))
    i,j=torch.triu_indices(20,20,1)
    distance=(paths[i]-paths[j]).norm(dim=-1)
    out.update(endpoint_diversity=distance[:,:,-1].mean(0),trajectory_diversity=distance.mean(-1).mean(0),
               near_duplicate_endpoint_fraction=(distance[:,:,-1]<=1.).double().mean(0))
    return out

def predict_all(net,cfg,templates,records,epoch,checkpoint_info,budget,scope='validation'):
    assert scope in {'validation','preflight_original','preflight_counterfactual'}
    N=len(records);assert len({r['sample_id'] for r in records})==N
    if scope=='validation':assert N==998
    else:
        allowed={r['sample_id'] for r in manifest('train')}
        assert N==2 and all(r['sample_id'] in allowed for r in records)
    stem=f'epoch{epoch:03d}' if scope=='validation' else scope
    assert all(set(r)=={'sample_id','scene','history'} for r in records),'No GT field permitted'
    predictor=Predictor(net,cfg,templates);cache={};traces=[];paths=[];goals=[]
    start=time.perf_counter()
    for i,r in enumerate(records):
        budget.guard(30);scene=r['scene']
        if scene not in cache:
            image=preprocess_scene(scene,cfg,'cuda:0');cache[scene]=fixed_semantics(net,image).cpu();del image
        try:p,g,t=predictor.predict(r['sample_id'],torch.as_tensor(r['history']).unsqueeze(0),semantic=cache[scene].cuda())
        except Exception as error:
            raise RuntimeError(f"Prediction failed epoch={epoch},sample_id={r['sample_id']},scene={scene}: {error}") from error
        paths.append(p);goals.append(g)
        traces.append(dict(epoch=epoch,sample_id=r['sample_id'],scene=scene,checkpoint_SHA=checkpoint_info['sha256'],**t))
        if (i+1)%50==0:print('VALIDATE epoch',epoch,'target',i+1,'/998','seconds',round(time.perf_counter()-start),flush=True)
    # Runtime checks of trained distributions before any GT scoring. Reverse3
    # target identities; check cached and recomputed semantics for each.
    gate=[]
    for i in dict.fromkeys([N-1,N//2,0]):
        r=records[i];h=torch.as_tensor(r['history']).unsqueeze(0)
        for mode in ['cached_reverse_repeat','recompute_semantics']:
            budget.guard(30)
            if mode=='cached_reverse_repeat':p,g,t=predictor.predict(r['sample_id'],h,semantic=cache[r['scene']].cuda())
            else:p,g,t=predictor.predict(r['sample_id'],h,image=preprocess_scene(r['scene'],cfg,'cuda:0'))
            assert torch.equal(p,paths[i]) and torch.equal(g,goals[i]),'K20 integration reproducibility failed'
            assert all(t[k]==traces[i][k] for k in ['proposal_SHA','goal_SHA','path_SHA'])
            gate.append(dict(epoch=epoch,sample_id=r['sample_id'],mode=mode,bitwise=True,**t))
    predicted=torch.cat(paths,dim=1);sampled_goals=torch.cat(goals,dim=1)
    assert predicted.shape==(20,N,12,2) and sampled_goals.shape==(20,N,2)
    path=CACHE/f'predictions/{stem}.pt';assert not path.exists()
    torch.save(dict(sample_ids=[r['sample_id'] for r in records],paths=predicted,goals=sampled_goals,
        checkpoint_SHA=checkpoint_info['sha256'],epoch=epoch,K=20,TTST_proposals=10000,units='original reference pixels'),path)
    cnew(OUT/f'prediction_traces_{stem}.csv.gz',traces)
    cnew(OUT/f'reproducibility_{stem}.csv',gate)
    info=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),targets=N,K=20,TTST_proposals=10000,
        checkpoint_SHA=checkpoint_info['sha256'],all_predictions_completed_before_GT_scoring=True,
        prediction_seconds=time.perf_counter()-start,trace_SHA=sha(OUT/f'prediction_traces_{stem}.csv.gz'),scope=scope)
    jnew(OUT/f'prediction_manifest_{stem}.json',info)
    return predicted,sampled_goals,info

def score(paths,goals,records,epoch,checkpoint_info):
    # This is the first future-coordinate read in the validation computation.
    truth_index=truths('validation')
    y=torch.from_numpy(np.stack([truth_index[r['sample_id']] for r in records]))
    m=calculate(paths,y,goals);source={r['sample_id']:r for r in manifest('validation')}
    paired=[];candidate=[]
    dist=(paths-y.unsqueeze(0)).norm(dim=-1);ade=dist.mean(-1);fde=dist[:,:,-1]
    goal_error=(goals-y[None,:,-1]).norm(dim=-1)
    for i,r in enumerate(records):
        sid=r['sample_id'];identity=source[sid]
        row=dict(epoch=epoch,sample_id=sid,scene=r['scene'],person_id=identity['person_id'],frame_id=identity['frame_id'],
            history_SHA=identity['history_SHA'],GT_SHA=identity['GT_SHA'],checkpoint_SHA=checkpoint_info['sha256'],
            **{k:float(m[k][i]) for k in METRICS},best_ADE_head=int(m['best_ADE_head'][i]),best_FDE_head=int(m['best_FDE_head'][i]))
        paired.append(row)
        for k in range(20):candidate.append(dict(epoch=epoch,sample_id=sid,head=k,ADE=float(ade[k,i]),FDE=float(fde[k,i]),
            GoalFDE=float(goal_error[k,i]),goal_path_gap=float((goals[k,i]-paths[k,i,-1]).norm())))
    cnew(OUT/f'paired_epoch{epoch:03d}.csv',paired)
    cnew(OUT/f'candidate_errors_epoch{epoch:03d}.csv.gz',candidate)
    scenes=[]
    for scene in sorted({r['scene'] for r in records}):
        a=[r for r in paired if r['scene']==scene]
        scenes.append(dict(epoch=epoch,scene=scene,targets=len(a),checkpoint_SHA=checkpoint_info['sha256'],
                           **{k:float(np.mean([r[k] for r in a])) for k in METRICS}))
    window={k:float(m[k].mean()) for k in METRICS}
    macro={k:float(np.mean([r[k] for r in scenes])) for k in METRICS}
    coverage={f'within_{threshold}px':float((m['minGoalFDE20']<=threshold).double().mean()) for threshold in [10,25,50]}
    return window,macro,scenes,coverage

def validate(net,cfg,templates,epoch,checkpoint_info,budget):
    start=time.perf_counter();net.eval()
    result=dict(epoch=epoch,status='INCOMPLETE',checkpoint=checkpoint_info,targets_expected=998,targets_scored=0)
    try:
        records=input_records('validation')
        p,g,prediction_info=predict_all(net,cfg,templates,records,epoch,checkpoint_info,budget)
        window,macro,scenes,coverage=score(p,g,records,epoch,checkpoint_info)
        result.update(status='COMPLETE',targets_scored=998,K=20,TTST_proposals=10000,window=window,recording_macro=macro,
            scenes=scenes,goal_coverage=coverage,prediction_manifest=prediction_info,unit='original reference pixels')
        return result
    except Exception as e:
        traceback.print_exc();result.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        result['total_seconds']=time.perf_counter()-start
        jnew(OUT/f'validation_epoch{epoch:03d}.json',result)
