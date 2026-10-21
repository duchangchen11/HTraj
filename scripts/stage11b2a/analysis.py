"""Offline paired descriptive analysis of an immutable prediction cache."""
from .common import *
from .features import motion,turn_group,speed_change_group,tercile,stats,spearman
import numpy as np,pandas as pd

METRICS=['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','SelectionGap_ADE','SelectionGap_FDE',
    'ADE1_diag','FDE1_diag','minADE20_diag','minFDE20_diag','minADE20_future_path_ratio','closest_goal_ADE','closest_goal_FDE']

def candidate_metrics(paths,y,goals):
    d=np.linalg.norm(paths-y[None],axis=-1);ade=d.mean(-1);fde=d[:,:,-1];ge=np.linalg.norm(goals-y[None,:,-1],axis=-1)
    gap=np.linalg.norm(goals-paths[:,:,-1],axis=-1);a=ade.argmin(0);f=fde.argmin(0);g=ge.argmin(0);ii=np.arange(len(y))
    m=dict(ADE1=ade[0],FDE1=fde[0],GoalFDE1=ge[0],minADE20=ade[a,ii],minFDE20=fde[f,ii],minGoalFDE20=ge[g,ii],
        best_ADE_head=a,best_FDE_head=f,closest_goal_head=g,closest_goal_ADE=ade[g,ii],closest_goal_FDE=fde[g,ii],
        closest_goal_path_gap=gap[g,ii],ADE_oracle_GoalFDE=ge[a,ii],FDE_oracle_GoalFDE=ge[f,ii])
    m['SelectionGap_ADE']=ade[0]-ade[a,ii];m['SelectionGap_FDE']=fde[0]-fde[f,ii]
    assert (m['SelectionGap_ADE']>=-1e-12).all() and (m['SelectionGap_FDE']>=-1e-12).all()
    return d,ade,fde,ge,gap,m

def grouped(frame,family,column,levels,scope):
    rows=[]
    for level in levels:
        a=frame[frame[column]==level];r=dict(family=family,group=level,feature_scope=scope,targets=len(a),
            scene_composition=json.dumps(a.scene.value_counts().sort_index().to_dict()),
            sample_ids_SHA=hashlib.sha256('\n'.join(sorted(a.sample_id)).encode()).hexdigest())
        for k in METRICS:
            r[k]=float(a[k].mean()) if len(a) else np.nan
            r[k+'_median']=float(a[k].median()) if len(a) else np.nan
        r.update(hist_mean_speed=float(a.hist_mean_speed.mean()) if len(a) else np.nan,
            future_path_length=float(a.future_path_length.mean()) if len(a) else np.nan,
            minGoalFDE20_over25_fraction=float((a.minGoalFDE20>25).mean()) if len(a) else np.nan)
        rows.append(r)
    return rows

def main():
    cpu_guard();start=time.perf_counter();config=protocol()
    assert (OUT/'protocol_freeze.json').exists(),'Commit protocol before grouping/statistics'
    freeze=readj(OUT/'protocol_freeze.json');assert sha(CONF/'analysis_protocol.json')==freeze['protocol_SHA']
    pack,h,y,identities=frozen();ids=pack['sample_ids'];p=pack['paths'].numpy().astype(float);goals=pack['goals'].numpy().astype(float)
    d,ade,fde,ge,gap,m=candidate_metrics(p,y,goals);audit=readj(OUT/'data_cache_audit.json')
    from scripts.stage11b1c.data import input_records
    refs=readj(OLD/'data_audit.json')['reference'];train=[]
    for r in input_records('train'):
        f=motion(r['history']);info=refs[r['scene']];diag=float(np.hypot(info['height'],info['width']))
        train.append(dict(sample_id=r['sample_id'],scene=r['scene'],hist_speed_diag=f['mean_speed']/diag,hist_path_diag=f['path_length']/diag))
    train=pd.DataFrame(train);train.to_csv(OUT/'source_history_threshold_features.csv',index=False)
    cuts={key:np.quantile(train[key],[1/3,2/3],method='linear').tolist() for key in ['hist_speed_diag','hist_path_diag']}
    old=pd.read_csv(OLD/'paired_epoch030.csv').set_index('sample_id').loc[ids]
    baseline=pd.read_csv(OLD/'best_paired_baseline_comparison.csv').set_index('sample_id').loc[ids]
    features=[]
    for i,r in enumerate(identities):
        hf=motion(h[i]);ff=motion(np.vstack([h[i,-1],y[i]]));info=audit['references'][r['scene']];diag=info['diagonal']
        row={k:r[k] for k in ['sample_id','scene','person_id','frame_id','history_SHA','GT_SHA','reference_SHA']}
        row.update(checkpoint_SHA=CK,reference_diagonal=diag,reference_width=info['width'],reference_height=info['height'],history_observations=8)
        row.update({'hist_'+k:v for k,v in hf.items()});row.update({'future_'+k:v for k,v in ff.items()})
        row.update({k:float(a[i]) if k not in ['best_ADE_head','best_FDE_head','closest_goal_head'] else int(a[i]) for k,a in m.items()})
        row.update({k:float(old.iloc[i][k]) for k in ['endpoint_diversity','trajectory_diversity','near_duplicate_endpoint_fraction']})
        row.update(hist_speed_diag=hf['mean_speed']/diag,hist_path_diag=hf['path_length']/diag,future_path_diag=ff['path_length']/diag,
            future_net_diag=ff['net_displacement']/diag,hist_turn_group=turn_group(hf),future_turn_group=turn_group(ff),
            hist_speed_change_group=speed_change_group(hf),future_speed_change_group=speed_change_group(ff),
            hist_stationary_group='all_steps_le1px' if hf['all_steps_stationary'] else 'other',
            hist_direction_evidence='0_or1_reliable_steps' if hf['reliable_motion_steps']<=1 else '2_to7_reliable_steps',
            SelectionGap_ADE_relative=m['SelectionGap_ADE'][i]/max(m['ADE1'][i],1),SelectionGap_FDE_relative=m['SelectionGap_FDE'][i]/max(m['FDE1'][i],1),
            minADE20_future_path_ratio=m['minADE20'][i]/max(ff['path_length'],1))
        for key in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20']:row[key+'_diag']=m[key][i]/diag
        for name in ['CV-Last2','OLS4','OLS8']:
            row[name+'_ADE1']=float(baseline.iloc[i][name+'_ADE1']);row[name+'_FDE1']=float(baseline.iloc[i][name+'_FDE1'])
        row['YNet_ADE1_minus_OLS8_ADE1']=row['ADE1']-row['OLS8_ADE1'];row['OLS8_ADE1_minus_YNet_ADE1']=-row['YNet_ADE1_minus_OLS8_ADE1']
        row['YNet_FDE1_minus_OLS8_FDE1']=row['FDE1']-row['OLS8_FDE1']
        for t in [10,25,50]:row[f'goal_miss{t}']=bool(m['minGoalFDE20'][i]>t)
        features.append(row)
    f=pd.DataFrame(features)
    future_cuts={key:np.quantile(f[key],[1/3,2/3],method='linear').tolist() for key in ['future_path_diag','future_net_diag']}
    for column,key in [('hist_speed_group','hist_speed_diag'),('hist_amplitude_group','hist_path_diag')]:f[column]=[tercile(x,cuts[key]) for x in f[key]]
    f['future_amplitude_group']=[tercile(x,future_cuts['future_path_diag']) for x in f.future_path_diag]
    f['future_net_group']=[tercile(x,future_cuts['future_net_diag']) for x in f.future_net_diag]
    writej(OUT/'derived_thresholds.json',dict(source_training_targets=7496,source_history_terciles=cuts,future_posthoc_terciles=future_cuts,
        method='Frozen linear quantiles; ties enter higher bin; all thresholds descriptive and not independent confirmation'))
    # Every goal is paired with its own decoded trajectory throughout.
    close=ge<=10;mismatch=close&(ade>=25);middle=mismatch&(fde<=10)&(d[:,:,:11].mean(-1)>=25);terminal=mismatch&(fde>=25)
    i0,i1=np.triu_indices(20,1);candidate_distance=np.linalg.norm(p[i0]-p[i1],axis=-1)
    f['default_is_ADE_best']=(m['best_ADE_head']==0);f['ADE_FDE_same_head']=m['best_ADE_head']==m['best_FDE_head']
    f['closest_goal_is_ADE_best']=m['closest_goal_head']==m['best_ADE_head'];f['closest_goal_is_FDE_best']=m['closest_goal_head']==m['best_FDE_head']
    f['high_gap_ADE25']=f.SelectionGap_ADE>=25
    f['DEFAULT_ORACLE_GAP']=(f.ADE1>=25)&(f.minADE20<=10)&(f.SelectionGap_ADE>=15)&(f.SelectionGap_ADE_relative>=.5)
    f['GOAL_COVERAGE_MISS']=f.goal_miss25;f['GOAL_PATH_MISMATCH']=mismatch.any(0)
    f['closest_goal_mismatch']=(m['minGoalFDE20']<=10)&(m['closest_goal_ADE']>=25)
    f['mismatch_middle_target']=middle.any(0);f['mismatch_terminal_target']=terminal.any(0)
    f['MOTION_EXTRAPOLATION_DIFFICULTY']=(f.minADE20>=25)&((f.hist_turn_group=='turning_ge30deg')|(f.hist_speed_change_group=='variable_CVge0.4')|(f.hist_direction_evidence=='0_or1_reliable_steps'))
    f['future_motion_difficulty_POSTHOC']=(f.minADE20>=25)&((f.future_turn_group=='turning_ge30deg')|(f.future_speed_change_group=='variable_CVge0.4'))
    f['history_straight_future_turn']=(f.hist_turn_group=='straight')&(f.future_turn_group=='turning_ge30deg')
    candidates=[];outpoints=[];geometry=[];step_rows=[]
    for i,sid in enumerate(ids):
        step=np.linalg.norm(np.diff(np.concatenate([np.broadcast_to(h[i,-1],(20,1,2)),p[:,i]],axis=1),axis=1),axis=-1)
        histmax=float(np.linalg.norm(np.diff(h[i],axis=0),axis=-1).max());limit=max(50.,4*histmax)
        out=(p[:,i,:,0]<0)|(p[:,i,:,0]>=f.iloc[i].reference_width)|(p[:,i,:,1]<0)|(p[:,i,:,1]>=f.iloc[i].reference_height)
        goalout=(goals[:,i,0]<0)|(goals[:,i,0]>=f.iloc[i].reference_width)|(goals[:,i,1]<0)|(goals[:,i,1]>=f.iloc[i].reference_height)
        geometry.append(dict(sample_id=sid,outside_point_count=int(out.sum()),outside_head_count=int(out.any(1).sum()),
            max_predicted_step_pixels=float(step.max()),max_history_step_pixels=histmax,large_step_threshold_pixels=limit,
            large_step_head_count=int((step.max(1)>limit).sum()),GEOMETRIC_ANOMALY=bool(out.any() or (step>limit).any())))
        for k in range(20):
            candidates.append(dict(sample_id=sid,scene=f.iloc[i].scene,head=k,ADE=ade[k,i],FDE=fde[k,i],GoalFDE=ge[k,i],goal_path_gap=gap[k,i],
                error_t1_4=d[k,i,:4].mean(),error_t5_8=d[k,i,4:8].mean(),error_t9_12=d[k,i,8:].mean(),
                is_closest_goal=k==m['closest_goal_head'][i],is_ADE_best=k==m['best_ADE_head'][i],is_FDE_best=k==m['best_FDE_head'][i],
                close_goal10=close[k,i],goal_path_mismatch=mismatch[k,i],mismatch_middle=middle[k,i],mismatch_terminal=terminal[k,i],
                outside_points=int(out[k].sum()),goal_outside_reference=bool(goalout[k]),large_step=bool(step[k].max()>limit)))
            for t in np.where(out[k])[0]:outpoints.append(dict(sample_id=sid,scene=f.iloc[i].scene,head=k,future_step=int(t+1),x=p[k,i,t,0],y=p[k,i,t,1],
                width=f.iloc[i].reference_width,height=f.iloc[i].reference_height,ADE=ade[k,i],FDE=fde[k,i],GoalFDE=ge[k,i]))
        for t in range(12):step_rows.append(dict(sample_id=sid,scene=f.iloc[i].scene,future_step=t+1,default_error=d[0,i,t],
            ADE_oracle_path_error=d[m['best_ADE_head'][i],i,t],closest_goal_path_error=d[m['closest_goal_head'][i],i,t]))
    geo=pd.DataFrame(geometry);f=f.merge(geo,on='sample_id',validate='one_to_one');assert len(f)==998
    assert f.outside_point_count.sum()==51,'Frozen previous geometry count must match'
    f.to_csv(OUT/'motion_features.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'goal_path_decomposition.csv',index=False)
    pd.DataFrame(outpoints).to_csv(OUT/'geometric_out_of_image_points.csv',index=False)
    pd.DataFrame(step_rows).to_csv(OUT/'paired_time_step_errors.csv',index=False)
    cols=['sample_id','scene','checkpoint_SHA','history_SHA','GT_SHA','ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20',
        'closest_goal_head','best_ADE_head','best_FDE_head','closest_goal_ADE','closest_goal_FDE','closest_goal_path_gap',
        'ADE_oracle_GoalFDE','FDE_oracle_GoalFDE','closest_goal_is_ADE_best','closest_goal_is_FDE_best','GOAL_PATH_MISMATCH','closest_goal_mismatch']
    f[cols].to_csv(OUT/'goal_path_summary.csv',index=False)
    cols=['sample_id','scene','ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','SelectionGap_ADE','SelectionGap_FDE',
        'SelectionGap_ADE_relative','SelectionGap_FDE_relative','best_ADE_head','best_FDE_head','endpoint_diversity','trajectory_diversity',
        'near_duplicate_endpoint_fraction','default_is_ADE_best','ADE_FDE_same_head','high_gap_ADE25','DEFAULT_ORACLE_GAP']
    f[cols].to_csv(OUT/'candidate_selection_gap.csv',index=False)
    groups=[]
    definitions=[('history_speed','hist_speed_group',['low','medium','high'],'HISTORY_INPUT'),
        ('history_amplitude','hist_amplitude_group',['low','medium','high'],'HISTORY_INPUT'),
        ('history_turn','hist_turn_group',['undefined_direction','straight','intermediate','turning_ge30deg'],'HISTORY_INPUT'),
        ('history_speed_change','hist_speed_change_group',['undefined_low_speed','stable_CVle0.15','intermediate','variable_CVge0.4'],'HISTORY_INPUT'),
        ('history_stationary','hist_stationary_group',['all_steps_le1px','other'],'HISTORY_INPUT'),
        ('history_direction_evidence','hist_direction_evidence',['0_or1_reliable_steps','2_to7_reliable_steps'],'HISTORY_INPUT'),
        ('future_amplitude','future_amplitude_group',['low','medium','high'],'FUTURE_GT_POSTHOC'),
        ('future_net','future_net_group',['low','medium','high'],'FUTURE_GT_POSTHOC'),
        ('future_turn','future_turn_group',['undefined_direction','straight','intermediate','turning_ge30deg'],'FUTURE_GT_POSTHOC'),
        ('future_speed_change','future_speed_change_group',['undefined_low_speed','stable_CVle0.15','intermediate','variable_CVge0.4'],'FUTURE_GT_POSTHOC')]
    for fam,col,levels,scope in definitions:groups.extend(grouped(f,fam,col,levels,scope))
    for t in [10,25,50]:
        g=f.copy();col=f'goal_miss{t}';g[col]=g[col].map({False:'covered',True:'missed'});groups.extend(grouped(g,f'goal_coverage_{t}px',col,['covered','missed'],'FUTURE_GT_POSTHOC'))
    pd.DataFrame(groups).to_csv(OUT/'motion_error_groups.csv',index=False)
    # Candidate correspondence summaries; candidates are not independent samples.
    candidate=pd.DataFrame(candidates);correlations=[]
    for scene in ['ALL_WINDOWS']+sorted(f.scene.unique()):
        s=f if scene=='ALL_WINDOWS' else f[f.scene==scene]
        for x in ['hist_path_length','hist_speed_diag','hist_speed_CV','hist_max_turn_degrees','hist_straightness','hist_reliable_motion_steps',
            'future_path_diag','future_max_turn_degrees','minGoalFDE20','closest_goal_FDE']:
            for target in ['minADE20','minADE20_diag','SelectionGap_ADE']:
                rho,n=spearman(s[x],s[target]);correlations.append(dict(scene=scene,x=x,y=target,finite_pairs=n,Spearman=rho,
                    feature_scope='FUTURE_GT_POSTHOC' if x.startswith(('future','minGoal','closest_goal')) else 'HISTORY_INPUT'))
        c=candidate if scene=='ALL_WINDOWS' else candidate[candidate.scene==scene]
        for xx,yy in [('GoalFDE','FDE'),('GoalFDE','ADE'),('goal_path_gap','ADE')]:
            rho,n=spearman(c[xx],c[yy]);correlations.append(dict(scene=scene,x='same_candidate_'+xx,y='same_candidate_'+yy,finite_pairs=n,Spearman=rho,feature_scope='FUTURE_GT_POSTHOC; clustered candidates, no significance claim'))
    pd.DataFrame(correlations).to_csv(OUT/'descriptive_correlations.csv',index=False)
    scene_rows=[];strata=[];standard=[];loo=[]
    for scene,s in f.groupby('scene',sort=True):
        r=dict(scene=scene,targets=len(s),diagonal=float(s.reference_diagonal.iloc[0]),
            hist_path_mean=s.hist_path_length.mean(),future_path_mean=s.future_path_length.mean(),hist_speed_mean=s.hist_mean_speed.mean(),
            hist_turn_mean=s.hist_mean_turn_degrees.mean(),future_turn_mean=s.future_mean_turn_degrees.mean(),
            hist_speed_diag_mean=s.hist_speed_diag.mean(),future_path_diag_mean=s.future_path_diag.mean(),
            goal_miss10_fraction=s.goal_miss10.mean(),goal_miss25_fraction=s.goal_miss25.mean(),goal_miss50_fraction=s.goal_miss50.mean(),
            YNet_ADE1_minus_OLS8_ADE1=s.YNet_ADE1_minus_OLS8_ADE1.mean(),YNet_FDE1_minus_OLS8_FDE1=s.YNet_FDE1_minus_OLS8_FDE1.mean())
        r.update({k:s[k].mean() for k in METRICS})
        for b in ['CV-Last2','OLS4','OLS8']:
            for k in ['ADE1','FDE1']:r[b+'_'+k]=s[b+'_'+k].mean();r[b+'_'+k+'_diag']=(s[b+'_'+k]/s.reference_diagonal).mean()
        scene_rows.append(r)
        for col in ['hist_speed_group','future_amplitude_group']:
            sizes=f[col].value_counts(normalize=True);valid=True;weighted=0
            for level in ['low','medium','high']:
                b=s[s[col]==level];valid&=len(b)>0
                strata.append(dict(scene=scene,factor=col,group=level,targets=len(b),mean_minADE20=b.minADE20.mean(),mean_minADE20_diag=b.minADE20_diag.mean(),
                    mean_ADE1=b.ADE1.mean(),mean_future_path_diag=b.future_path_diag.mean(),mean_hist_speed_diag=b.hist_speed_diag.mean()))
                if len(b):weighted+=sizes[level]*b.minADE20_diag.mean()
            standard.append(dict(scene=scene,factor=col,all_three_bins_present=bool(valid),standardized_minADE20_diag=weighted if valid else np.nan,
                rule='Common pooled target weights; no missing-bin imputation',pooled_low_weight=float(sizes.get('low',0)),pooled_medium_weight=float(sizes.get('medium',0)),pooled_high_weight=float(sizes.get('high',0))))
    for excluded in ['NONE']+sorted(f.scene.unique()):
        s=f if excluded=='NONE' else f[f.scene!=excluded]
        for b in ['CV-Last2','OLS4','OLS8']:
            a=s.ADE1-s[b+'_ADE1'];z=s.FDE1-s[b+'_FDE1']
            loo.append(dict(excluded_recording=excluded,remaining_recordings=s.scene.nunique(),targets=len(s),baseline=b,
                paired_ADE1_delta=a.mean(),paired_FDE1_delta=z.mean(),YNet_ADE1=s.ADE1.mean(),baseline_ADE1=s[b+'_ADE1'].mean(),
                recording_macro_ADE1_delta=s.assign(delta=a).groupby('scene').delta.mean().mean(),
                recording_macro_FDE1_delta=s.assign(delta=z).groupby('scene').delta.mean().mean(),
                targets_ADE_improved=int((a<0).sum()),targets_ADE_worsened=int((a>0).sum())))
    pd.DataFrame(scene_rows).to_csv(OUT/'scene_normalized_analysis.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'scene_motion_strata.csv',index=False)
    pd.DataFrame(standard).to_csv(OUT/'scene_motion_standardization.csv',index=False)
    pd.DataFrame(loo).to_csv(OUT/'leave_one_recording_out.csv',index=False)
    names=['GOAL_COVERAGE_MISS','DEFAULT_ORACLE_GAP','GOAL_PATH_MISMATCH','MOTION_EXTRAPOLATION_DIFFICULTY','GEOMETRIC_ANOMALY']
    f[['sample_id','scene','history_SHA','GT_SHA','checkpoint_SHA']+names+['goal_miss10','goal_miss25','goal_miss50','closest_goal_mismatch',
        'mismatch_middle_target','mismatch_terminal_target','future_motion_difficulty_POSTHOC','outside_point_count','outside_head_count','max_predicted_step_pixels',
        'large_step_threshold_pixels','large_step_head_count']+['ADE1','FDE1','minADE20','minFDE20']].to_csv(OUT/'failure_taxonomy.csv',index=False)
    overlap=[];taxrows=[]
    for scene in ['ALL_WINDOWS']+sorted(f.scene.unique()):
        s=f if scene=='ALL_WINDOWS' else f[f.scene==scene]
        for name in names:
            a=s[s[name]];taxrows.append(dict(scene=scene,type=name,targets=len(a),total_targets=len(s),fraction=len(a)/len(s),
                ADE1=a.ADE1.mean(),minADE20=a.minADE20.mean(),minGoalFDE20=a.minGoalFDE20.mean(),
                cohort_minADE20_error_share=float(a.minADE20.sum()/s.minADE20.sum())))
        for a in names:
            for b in names:overlap.append(dict(scene=scene,type_A=a,type_B=b,intersection_targets=int((s[a]&s[b]).sum())))
    pd.DataFrame(taxrows).to_csv(OUT/'failure_type_summary.csv',index=False);pd.DataFrame(overlap).to_csv(OUT/'failure_overlap.csv',index=False)
    combination=f[names].apply(lambda r:'+'.join(name for name in names if r[name]) or 'NONE',axis=1)
    f.assign(combination=combination).groupby(['scene','combination']).agg(targets=('sample_id','size'),minADE20=('minADE20','mean')).reset_index().to_csv(OUT/'failure_exact_combinations.csv',index=False)
    distributions=[]
    for scene in ['ALL_WINDOWS']+sorted(f.scene.unique()):
        s=f if scene=='ALL_WINDOWS' else f[f.scene==scene]
        for key in METRICS+['SelectionGap_ADE_relative','SelectionGap_FDE_relative','endpoint_diversity','near_duplicate_endpoint_fraction']:
            distributions.append(dict(scene=scene,metric=key,**stats(s[key])))
    pd.DataFrame(distributions).to_csv(OUT/'metric_distributions.csv',index=False)
    # Use only existing seven validation files; no new model calls.
    curves=[]
    for ep in [1,5,10,15,20,25,30]:
        v=readj(OLD/f'validation_epoch{ep:03d}.json');assert v['status']=='COMPLETE'
        for s in v['scenes']:
            curves.append(dict(epoch=ep,scene=s['scene'],targets=s['targets'],**{k:s[k] for k in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','endpoint_diversity','trajectory_diversity','goal_path_gap_mean20']}))
        curves.append(dict(epoch=ep,scene='ALL_WINDOWS',targets=998,**v['window'],**v['goal_coverage']))
    pd.DataFrame(curves).to_csv(OUT/'frozen_validation_trends.csv',index=False)
    checks=dict(passed=True,targets=998,candidate_pairs=19960,CPU_only=True,CUDA_initialized=False,new_model_calls=0,new_training_steps=0,test_loaded=False,
        protocol_SHA=sha(CONF/'analysis_protocol.json'),protocol_freeze_commit=freeze['commit'],analysis_source_SHA=sha(Path(__file__)),source_checkpoint_SHA=CK,source_prediction_SHA=PSHA,
        selection_gap_nonnegative=True,all_targets_retained=True,all20_heads_retained=True,
        mean_metrics={k:float(f[k].mean()) for k in METRICS},default_ADE_best_targets=int(f.default_is_ADE_best.sum()),ADE_FDE_same_head_targets=int(f.ADE_FDE_same_head.sum()),
        high_ADE_gap25_targets=int(f.high_gap_ADE25.sum()),goal_misses={str(t):int(f[f'goal_miss{t}'].sum()) for t in [10,25,50]},
        all_types={name:int(f[name].sum()) for name in names},closest_goal_mismatch_targets=int(f.closest_goal_mismatch.sum()),
        mismatch_middle_targets=int(f.mismatch_middle_target.sum()),mismatch_terminal_targets=int(f.mismatch_terminal_target.sum()),
        outside_points=int(f.outside_point_count.sum()),outside_targets=int((f.outside_point_count>0).sum()),large_step_targets=int((f.large_step_head_count>0).sum()),
        seconds=time.perf_counter()-start)
    cpu_guard();writej(OUT/'analysis_execution.json',checks)
    print(json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
