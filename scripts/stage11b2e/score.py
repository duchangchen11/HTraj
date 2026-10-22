"""Separate CPU GT scoring only after all998 alternative paths exist and verify."""
from .common import *
from .metrics import goal_metrics,path_metrics,gate
import torch
import numpy as np
import pandas as pd

def main():
    cpu_only();start=time.perf_counter();gen=json.loads((OUT/'generation_execution.json').read_text())
    assert gen['passed'] and gen['generated']==998 and gen['parameters_unchanged'] and not gen['GT_loaded']
    assert json.loads((OUT/'process_decode.json').read_text())['returncode']==0 and parent_seconds()<3600
    alternative=ROOT/gen['alternative_paths_path'];assert sha(alternative)==gen['alternative_paths_file_SHA']
    assert sha(PP)==PSHA and sha(CP)==CK
    new=torch.load(alternative,map_location='cpu',weights_only=True);old=torch.load(PP,map_location='cpu',weights_only=True)
    oi={sid:i for i,sid in enumerate(old['sample_ids'])};assert len(set(new['sample_ids']))==998 and set(new['sample_ids'])==set(oi)
    # Full head0 regression is a prerequisite, before looking at any GT error.
    for j,sid in enumerate(new['sample_ids']):
        assert torch.equal(new['goals'][0,j],old['goals'][0,oi[sid]])
        assert torch.equal(new['paths'][0,j],old['paths'][0,oi[sid]]),(sid,'head0 regression')
    # Future data loader is imported only after generation and regression gates.
    from scripts.stage11b1c.data import truths,manifest
    truth=truths('validation');identities={r['sample_id']:r for r in manifest('validation')};assert set(truth)==set(oi)
    m=pd.read_csv(OLD/'tensor_manifest.csv').set_index('sample_id')
    hist=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv').set_index('sample_id')
    prior=pd.read_csv(ROOT/'results/tables/stage11b2d/target_diagnostics.csv').set_index('sample_id')
    rep=pd.read_csv(OUT/'representative_selection.csv');grows=[];prows=[];heads=[];allrows=[];recover=[]
    for j,sid in enumerate(new['sample_ids']):
        i=oi[sid];r=m.loc[sid];cache=ROOT/r.path;assert sha(cache)==r.sha256
        x=torch.load(cache,map_location='cpu',weights_only=True)['tensors']
        P=x['proposals_raw'][0,0].numpy();weights=x['normalized_weights'].numpy().reshape(x['sigmoid'].shape[-2:])
        ag=new['goals'][:,j].numpy();og=old['goals'][:,i].numpy();ap=new['paths'][:,j].numpy();op=old['paths'][:,i].numpy();y=truth[sid]
        om,od=goal_metrics(og,y[-1],P,weights);am,ad=goal_metrics(ag,y[-1],P,weights)
        pm,pa,pf=path_metrics(op,y);qm,qa,qf=path_metrics(ap,y)
        assert pm['ADE1']==qm['ADE1'] and pm['FDE1']==qm['FDE1']
        population=rep[rep.sample_id==sid]
        om['FixedCellDistortion19']=float(population.original_fixed_cell_squared_loss.sum()/10000)
        am['FixedCellDistortion19']=float(population.alternative_fixed_cell_squared_loss.sum()/10000)
        for mm,g,p in [(om,og,op),(am,ag,ap)]:
            mm['GoalPathEndpointMeanDistance']=float(np.linalg.norm(g.astype(float)-p[:,-1].astype(float),axis=-1).mean())
        base=dict(sample_id=sid,scene=r.scene,history_SHA=identities[sid]['history_SHA'],GT_SHA=identities[sid]['GT_SHA'],
            checkpoint_SHA=CK,original_goal_SHA=r.goal_SHA,alternative_goal_SHA=digest(new['goals'][:,j:j+1]),
            original_path_SHA=r.path_SHA,alternative_path_SHA=digest(new['paths'][:,j:j+1]))
        gbase=dict(base);pbase=dict(base)
        for key in om:gbase['Original'+key]=om[key];gbase['Alternative'+key]=am[key];gbase['Delta'+key]=am[key]-om[key]
        for key in pm:pbase['Original'+key]=pm[key];pbase['Alternative'+key]=qm[key];pbase['Delta'+key]=qm[key]-pm[key]
        gbase.update(changed_goal_heads=int(np.any(og!=ag,axis=-1).sum()),
            changed_decoder_map_centers=int(np.any(np.round(og*.25)!=np.round(ag*.25),axis=-1).sum()),
            mean_goal_displacement=float(np.linalg.norm(ag.astype(float)-og.astype(float),axis=-1).mean()))
        pbase.update(changed_path_heads=int(np.any(op!=ap,axis=(1,2)).sum()),head0_bitwise=True,
            all20paths_bitwise=bool(np.array_equal(op,ap)),max_path_coordinate_difference=float(np.abs(op.astype(float)-ap.astype(float)).max()),
            original_head_ADE=json.dumps(pa.tolist()),alternative_head_ADE=json.dumps(qa.tolist()),
            original_head_FDE=json.dumps(pf.tolist()),alternative_head_FDE=json.dumps(qf.tolist()))
        outcome=('stayed_cover' if am['GoalCover25'] else 'lost') if om['GoalCover25'] else ('recovered' if am['GoalCover25'] else 'stayed_fail')
        cohort=prior.loc[sid,'category25'];count25=int((np.linalg.norm(P.astype(float)-y[-1],axis=-1)<=25).sum())
        rr=dict(sample_id=sid,scene=r.scene,original_category25=cohort,paired_outcome25=outcome,
          OriginalGoalCover25=om['GoalCover25'],AlternativeGoalCover25=am['GoalCover25'],LocalProposalCount25=count25,
          OriginalMinADE20=pm['minADE20'],AlternativeMinADE20=qm['minADE20'],DeltaMinADE20=qm['minADE20']-pm['minADE20'],
          OriginalMinFDE20=pm['minFDE20'],AlternativeMinFDE20=qm['minFDE20'],DeltaMinFDE20=qm['minFDE20']-pm['minFDE20'])
        assert bool(om['GoalCover25'])==bool(prior.loc[sid,'category25']=='both_hit')
        for k in ['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group']:rr[k]=hist.loc[sid,k]
        for k in range(20):
            heads.append(dict(sample_id=sid,scene=r.scene,head=k,OriginalGoal_x=float(og[k,0]),OriginalGoal_y=float(og[k,1]),
                AlternativeGoal_x=float(ag[k,0]),AlternativeGoal_y=float(ag[k,1]),OriginalPathEnd_x=float(op[k,-1,0]),OriginalPathEnd_y=float(op[k,-1,1]),
                AlternativePathEnd_x=float(ap[k,-1,0]),AlternativePathEnd_y=float(ap[k,-1,1]),OriginalGoalFDE=float(od[k]),AlternativeGoalFDE=float(ad[k]),
                OriginalADE=float(pa[k]),AlternativeADE=float(qa[k]),OriginalFDE=float(pf[k]),AlternativeFDE=float(qf[k]),
                GoalBitwiseUnchanged=bool(np.array_equal(og[k],ag[k])),PathBitwiseUnchanged=bool(np.array_equal(op[k],ap[k]))))
        grows.append(gbase);prows.append(pbase);recover.append(rr)
        allrows.append(dict(**base,**{'Original'+k:v for k,v in {**om,**pm}.items()},**{'Alternative'+k:v for k,v in {**am,**qm}.items()}))
        if (j+1)%200==0:print('CPU_SCORE',j+1,'/998',flush=True)
    gf=pd.DataFrame(grows);pf=pd.DataFrame(prows);rf=pd.DataFrame(recover);f=pd.DataFrame(allrows)
    for name,df in [('paired_goal_metrics.csv',gf),('paired_path_metrics.csv',pf),('paired_head_metrics.csv',pd.DataFrame(heads)),('failure_recovery.csv',rf)]:df.to_csv(OUT/name,index=False)
    metrics=[k for k in {**om,**pm} if not k.endswith('best_head')];numeric=['Original'+k for k in metrics]+['Alternative'+k for k in metrics]
    def agg(q,label):
        row=dict(scope=label,targets=len(q),aggregation='target_window_weighted')
        row.update({k:float(q[k].mean()) for k in numeric})
        for k in metrics:row['Delta'+k]=row['Alternative'+k]-row['Original'+k]
        return row
    scene=[agg(f,'ALL998')]+[agg(q,name) for name,q in f.groupby('scene')]+[agg(f[f.scene!='deathCircle_3'],'OTHER4')]
    cf=pd.DataFrame(scene);individual=cf[cf.scope.isin(['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'])]
    macro=dict(scope='FIVE_RECORDING_MACRO',targets=998,aggregation='equal_recording_weight')
    for k in numeric+['Delta'+k for k in metrics]:macro[k]=float(individual[k].mean())
    cf=pd.concat([cf,pd.DataFrame([macro])],ignore_index=True);cf.to_csv(OUT/'scene_comparison.csv',index=False)
    motion=[]
    for factor in ['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group']:
        groups=sorted(rf[factor].unique())
        for scene in ['ALL998','bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4']:
            for group in groups:
                ids=rf[(rf[factor]==group)&((rf.scene==scene) if scene!='ALL998' else True)].sample_id
                q=f[f.sample_id.isin(ids)];rr=agg(q,scene);rr.update(factor=factor,group=group);motion.append(rr)
    pd.DataFrame(motion).to_csv(OUT/'motion_group_comparison.csv',index=False)
    original_macro={k:macro['Original'+k] for k in metrics};alternative_macro={k:macro['Alternative'+k] for k in metrics}
    other=cf[cf.scope=='OTHER4'].iloc[0];d=gate(original_macro,alternative_macro,{k:other['Original'+k] for k in metrics},{k:other['Alternative'+k] for k in metrics})
    q=rf[rf.original_category25=='proposal_hit_final_miss'];z=rf[rf.original_category25=='zero_support_final_miss'];assert len(q)==86 and len(z)==77
    oldmean=f.mean(numeric_only=True)
    for key,rounded in [('ADE1',32.0931),('FDE1',60.5390),('minADE20',16.4031),('minFDE20',25.6340)]:assert abs(float(oldmean['Original'+key])-rounded)<.0001,(key,oldmean['Original'+key])
    cases=[]
    for outcome,sub in rf.groupby('paired_outcome25'):
        sub=sub.sort_values('sample_id',kind='stable');r=sub.iloc[(len(sub)-1)//2]
        cases.append(dict(sample_id=r.sample_id,scene=r.scene,outcome=outcome,eligible_targets=len(sub),rank=(len(sub)-1)//2))
    pd.DataFrame(cases).to_csv(OUT/'case_manifest.csv',index=False)
    d.update(scope='CONTROL_EXPERIMENT_ONLY',targets=998,all_head0_goals_paths_bitwise=True,
      baseline_A25_covered=int(rf.OriginalGoalCover25.sum()),alternative_A25_covered=int(rf.AlternativeGoalCover25.sum()),
      original86_recovered=int(q.AlternativeGoalCover25.sum()),original86_still_fail=int((~q.AlternativeGoalCover25).sum()),
      original86_minADE20_original=float(q.OriginalMinADE20.mean()),original86_minADE20_alternative=float(q.AlternativeMinADE20.mean()),
      original86_minFDE20_original=float(q.OriginalMinFDE20.mean()),original86_minFDE20_alternative=float(q.AlternativeMinFDE20.mean()),
      lost_originally_covered=int((rf.paired_outcome25=='lost').sum()),recovered_all_original_fail=int((rf.paired_outcome25=='recovered').sum()),
      zero77_final_covered=int(z.AlternativeGoalCover25.sum()),zero77_support_unchanged=True,
      goal_heads_changed=int(gf.changed_goal_heads.sum()),unchanged20goal_targets=int((gf.changed_goal_heads==0).sum()),
      decoder_map_centers_changed=int(gf.changed_decoder_map_centers.sum()),path_heads_changed=int(pf.changed_path_heads.sum()),
      unchanged20path_targets=int(pf.all20paths_bitwise.sum()),mean_goal_displacement_all20=float(gf.mean_goal_displacement.mean()),
      novelty_GO=False,training_steps=0,new_sampling=0,new_kmeans=0,test_set_used=False,source_commit=commit())
    qnum=q[['LocalProposalCount25','DeltaMinADE20']].rank().corr().iloc[0,1]
    d['within_original86_local_count_ADEdelta_Spearman_descriptive']=None if pd.isna(qnum) else float(qnum)
    jnew(OUT/'final_decision.json',d)
    sel=json.loads((OUT/'selection_audit.json').read_text());reg=json.loads((OUT/'decoder_regression.json').read_text())
    jnew(OUT/'computational_cost.json',dict(CPU_selection=sel,CPU_score_seconds=time.perf_counter()-start,
        GPU_process_seconds=parent_seconds(),GPU_regression=reg,GPU_decode=gen,new_training_seconds=0,
        note='algorithm operation times vs full subprocess wall are separately labeled; frozen semantic cache reused, no new segmentation inference'))
    jnew(OUT/'gpu_budget.json',dict(actual_GPU_process_seconds=parent_seconds(),hard_seconds=3600,safety_seconds=3300,
        processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],within_budget=parent_seconds()<3300,training_seconds=0))
    jnew(OUT/'score_integrity.json',dict(passed=True,all998_ID_GT_history_manifest_matched=True,
      complete_generation_before_actual_GT_loaded=True,all998_head0_bitwise=True,original_goal_path_cache_SHA=PSHA,
      predictions_from_actual_new_decoder=True,no_test_data=True,no_GT_in_representatives=True,actual_score_CPU_seconds=time.perf_counter()-start))
    print(json.dumps(d,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
