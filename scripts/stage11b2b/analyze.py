"""CPU-only paired offline audit after all GPU work; never modifies training."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from .metrics import candidate_metrics,aggregate,verdict
from scripts.stage11b1c.data import truths,manifest,input_records
import numpy as np,pandas as pd,torch

def main():
    assert not torch.cuda.is_initialized()
    analysis=json.loads((CONF/'analysis.json').read_text());frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'analysis.json')==frozen['analysis_config_SHA']
    for name,s in frozen['frozen_analysis_sources'].items():assert sha(ROOT/name)==s
    features=pd.read_csv(ROOT/analysis['frozen_motion_groups']).set_index('sample_id')
    fixed=set(features.index[features.minGoalFDE20>25]);fixed_c=set(features.index[features.GOAL_PATH_MISMATCH]);assert len(fixed)==164 and len(fixed_c)==7
    identity={r['sample_id']:r for r in manifest('validation')};truth=truths('validation');expected=set(identity)
    summary=[];coverage=[];scene=[];motion=[];cohorts=[];transitions=[];states=[];cstates=[];groups=[];allf={};sources=[];timestep=[]
    process=json.loads((OUT/'training_execution.json').read_text());oldids=None
    for ep in [1,5,10,15,20,25,30,35,40,45,50]:
        home=OLD if ep<=30 else OUT;vp=home/f'validation_epoch{ep:03d}.json'
        if not vp.exists():states.append(dict(epoch=ep,status='NOT_RUN'));continue
        v=json.loads(vp.read_text())
        if v['status']!='COMPLETE':states.append(dict(epoch=ep,status=v['status']));continue
        states.append(dict(epoch=ep,status='COMPLETE'))
        pp=ROOT/v['prediction_manifest']['path'];assert sha(pp)==v['prediction_manifest']['sha256']
        pack=torch.load(pp,map_location='cpu',weights_only=True);ids=pack['sample_ids'];assert len(ids)==998 and set(ids)==expected
        if oldids is None:oldids=ids
        assert ids==oldids and pack['checkpoint_SHA']==v['checkpoint']['sha256']
        p=pack['paths'].numpy();g=pack['goals'].numpy();y=np.stack([truth[s] for s in ids]);d,a,z,ge,gap,m=candidate_metrics(p,y,g)
        for key in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20']:assert np.isclose(m[key].mean(),v['window'][key],atol=1e-8,rtol=0),(ep,key)
        f=pd.DataFrame(m,index=pd.Index(ids,name='sample_id'));f['scene']=[identity[s]['scene'] for s in ids];f['epoch']=ep
        for col in analysis['groups']:f[col]=features.loc[ids,col]
        for col in ['person_id','frame_id','history_SHA','GT_SHA']:f[col]=[identity[s][col] for s in ids]
        for t in [10,25,50]:f[f'A{t}']=f.minGoalFDE20>t
        allf[ep]=f
        cksha=sha(ROOT/v['checkpoint']['path']);assert cksha==v['checkpoint']['sha256']
        info=dict(epoch=ep,checkpoint_SHA=cksha,prediction_cache_SHA=sha(pp),validation_seconds=v['total_seconds'],
            GPU_seconds=v['total_seconds'],GPU_seconds_scope='Validation phase only; whole process total separately in gpu_budget.json')
        summary.append(dict(**info,**aggregate(f)));sources.append(dict(epoch=ep,checkpoint=v['checkpoint'],prediction_manifest=v['prediction_manifest']))
        for t in [10,25,50]:
            sub=f[f[f'A{t}']]
            coverage.append(dict(epoch=ep,threshold_pixels=t,strict_operator='>',targets=len(sub),fraction=len(sub)/998,
                mean_ADE1=float(sub.ADE1.mean()) if len(sub) else None,mean_minADE20=float(sub.minADE20.mean()) if len(sub) else None,
                mean_minFDE20=float(sub.minFDE20.mean()) if len(sub) else None,mean_minGoalFDE20=float(sub.minGoalFDE20.mean()) if len(sub) else None,
                minGoalFDE20_P50=float(sub.minGoalFDE20.median()) if len(sub) else None,minGoalFDE20_P90=float(sub.minGoalFDE20.quantile(.9)) if len(sub) else None,
                minGoalFDE20_P95=float(sub.minGoalFDE20.quantile(.95)) if len(sub) else None,
                minADE20_error_share=float(sub.minADE20.sum()/f.minADE20.sum())))
        for name,s in f.groupby('scene'):
            scene.append(dict(epoch=ep,scene=name,**aggregate(s),A25_share_of_all_errors=float(s.loc[s.A25,'minADE20'].sum()/f.minADE20.sum())))
        if ep>=30:
            f.to_csv(OUT/f'paired_coverage_epoch{ep:03d}.csv',index=True)
            sub=f.loc[sorted(fixed)].copy();sub['fixed_cohort']='FIXED_A25_EPOCH30_COHORT';sub.reset_index().to_csv(OUT/f'fixed_a25_epoch{ep:03d}.csv',index=False)
            cohorts.extend(sub.reset_index().to_dict('records'))
            now=set(f.index[f.A25]);prev=allf[max(e for e in allf if e<ep)] if ep>30 else f
            prior=set(prev.index[prev.A25])
            for label,base in [('vs_epoch30',fixed),('vs_previous_complete',prior)]:
                transitions.append(dict(epoch=ep,comparison=label,reference_epoch=30 if label=='vs_epoch30' else int(prev.epoch.iloc[0]),
                    recovered=len(base-now),new_failures=len(now-base),persistent_failure=len(base&now),persistent_success=len(expected-base-now)))
            for sid in ids:
                cstates.append(dict(epoch=ep,sample_id=sid,original_C7=sid in fixed_c,current_GOAL_PATH_MISMATCH=bool(f.loc[sid,'GOAL_PATH_MISMATCH']),
                    original_A25=sid in fixed,current_A25=bool(f.loc[sid,'A25'])))
            for col in analysis['groups']:
                for label,s in f.groupby(col):motion.append(dict(epoch=ep,factor=col,group=label,**aggregate(s)))
            for i,sid in enumerate(ids):
                for t in range(12):timestep.append(dict(epoch=ep,sample_id=sid,scene=f.loc[sid,'scene'],future_step=t+1,default_error=d[0,i,t],
                    ADE_oracle_path_error=d[m['best_ADE_head'][i],i,t],closest_goal_path_error=d[ge.argmin(0)[i],i,t]))
    summ=pd.DataFrame(summary);old=summ[summ.epoch==30].iloc[0];assert old.A25_miss_count==164 and np.isclose(old.A25_error_share,.610817,atol=1e-6)
    last=max([e for e in allf if e>30],default=None);final=allf[last] if last else None
    decision=verdict(float((final.minGoalFDE20>25).mean()) if last else 0,float(final.loc[final.A25,'minADE20'].sum()/final.minADE20.sum()) if last else 0,last is not None)
    summ.to_csv(OUT/'validation_metrics.csv',index=False);pd.DataFrame(states).to_csv(OUT/'validation_status.csv',index=False)
    for name,rows in [('goal_coverage_epochs.csv',coverage),('scene_metrics.csv',scene),('motion_group_metrics.csv',motion),('a25_fixed_cohort.csv',cohorts),('a25_transition.csv',transitions),('target_cohort_transitions.csv',cstates)]:pd.DataFrame(rows).to_csv(OUT/name,index=False)
    pd.DataFrame(timestep).to_csv(OUT/'paired_time_step_errors.csv.gz',index=False)
    epochs=pd.read_csv(OUT/'epochs.csv');assert epochs.epoch.tolist()==list(range(31,int(process['completed_epochs'])+1))
    assert (epochs.optimizer_steps==1882).all() and (epochs.unique_targets==7496).all() and (epochs.train_target_visits==7496).all()
    assert epochs.optimizer_steps.sum()==process['optimizer_updates'] and int(epochs.global_step.iloc[-1])==process['global_step']
    steps=pd.read_csv(OUT/'training_steps.csv.gz');audit=[]
    expected_train={r['sample_id'] for r in manifest('train')}
    for ep,s in steps.groupby('epoch'):
        visits=[sid for x in s.sample_ids for sid in x.split('|')]
        if ep in epochs.epoch.values:assert len(s)==1882 and len(visits)==len(set(visits))==7496 and set(visits)==expected_train
        audit.append(dict(epoch=int(ep),steps=len(s),visits=len(visits),unique_targets=len(set(visits)),finite=bool(np.isfinite(s[['loss','goal_loss','path_loss','gradient_norm']]).all().all()),
            learning_rates=sorted(s.learning_rate.unique().tolist()),gradient_norm_mean=float(s.gradient_norm.mean()),gradient_norm_max=float(s.gradient_norm.max())))
    jnew(OUT/'training_access_audit.json',dict(passed=True,epochs=audit,train_manifest_SHA=sha(OUT/'train_target_manifest.csv'),validation_manifest_SHA=sha(OUT/'validation_target_manifest.csv'),test_loaded=False))
    budget=dict(processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],new_GPU_seconds=parent_seconds(),hard_seconds=16200,safety_seconds=15000,within_hard_limit=parent_seconds()<16200,
        CPU_analysis_excluded=True,CUDA_initialized=False,charge='Sum of whole GPU child process parent wall, including failed admission fixture')
    assert budget['within_hard_limit'];jnew(OUT/'gpu_budget.json',budget)
    best=summ.sort_values(['minADE20','epoch']).iloc[0];new=summ[summ.epoch>30].sort_values(['minADE20','epoch'])
    cf=pd.DataFrame(transitions);fr=cf[(cf.epoch==last)&(cf.comparison=='vs_epoch30')].iloc[0].to_dict() if last else None
    finalrow=summ[summ.epoch==last].iloc[0].to_dict() if last else None
    result=dict(status=process['status'],restore_GO=json.loads((OUT/'resume_admission.json').read_text())['passed'],
        final_trained_epoch=int(process['completed_epochs']),final_complete_validation_epoch=last,H1=decision,epoch30=old.to_dict(),final=finalrow,
        global_best_epoch=int(best.epoch),new_best_epoch=int(new.iloc[0].epoch) if len(new) else None,
        global_best_metrics=best.to_dict(),new_best_metrics=new.iloc[0].to_dict() if len(new) else None,
        transitions=fr,new_GPU_seconds=budget['new_GPU_seconds'],new_optimizer_updates=int(process['optimizer_updates']),
        cumulative_global_step=int(process['global_step']),semantic_direct_causal_evidence=False,
        innovation_algorithm_GO=False,automatic_more_training_GO=False,
        further_goal_coverage_audit_worthwhile=decision=='H1_PERSISTENCE_SUPPORTED_WITHIN_BUDGET',
        limits=['development only','single seed','overlapping windows not independent','semantic label mapping unverified','semantic pretrain membership unknown','not evidence of full convergence','no new model or test scoring'])
    jnew(OUT/'final_decision.json',result);jnew(OUT/'verified_validation_sources.json',sources)
    cm=json.loads((OUT/'checkpoint_manifest.json').read_text());cm.update(epoch30_reference=[s for s in sources if s['epoch']==30][0],
        new_best_epoch=result['new_best_epoch'],global_best_epoch=int(best.epoch),global_best_checkpoint=[s['checkpoint'] for s in sources if s['epoch']==int(best.epoch)][0],
        last_complete_validation_epoch=last,last_completed_train_epoch=int(process['completed_epochs']))
    replace_json(OUT/'checkpoint_manifest.json',cm)
    jnew(OUT/'offline_analysis_execution.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,source_SHA=sha(Path(__file__)),complete_validation_epochs=list(allf),
        independent_NumPy_metric_agreement_atol=1e-8,old_metrics_not_modified=True,all_targets_kept=True,same_head_goal_path=True,source_commit=commit()))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
