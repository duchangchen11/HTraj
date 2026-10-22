"""All generation complete first; only this CPU phase reads real future labels."""
from .common import *
from .statistics import compare_distribution,distribution_stats,compare_goals,prediction_metrics,compare_fixed_paths
import torch,numpy as np,pandas as pd

def main():
    cpu_only();started=time.perf_counter()
    fz=json.loads((OUT/'protocol_frozen.json').read_text())
    for rec in fz['executable_files']:assert sha(ROOT/rec['path'])==rec['sha256']
    data=json.loads((CONF/'generation_config.json').read_text());assert sha(PP)==PSHA and sha(CP)==CK
    full=json.loads((OUT/'full_generation.json').read_text());fixed=json.loads((OUT/'fixed_generation.json').read_text())
    assert full['passed'] and fixed['passed'] and full['generated']==fixed['generated']==2994
    assert full['parameters_unchanged'] and fixed['parameters_unchanged'] and not full['GT_loaded'] and not fixed['GT_loaded']
    assert parent_seconds()<6600
    hp=checked(data['history']);old=checked(data['original_predictions']);ids=hp['sample_ids'];assert ids==old['sample_ids']
    packs={'S0':old};B={}
    for rec in full['groups']:
        x=checked(rec);assert x['sample_ids']==ids and x['paths'].shape==(20,998,12,2) and torch.isfinite(x['paths']).all();packs[rec['condition']]=x
    for rec in fixed['groups']:
        x=checked(rec);assert x['sample_ids']==ids and torch.equal(x['goals'],old['goals']) and x['paths'].shape==(20,998,12,2)
        assert torch.isfinite(x['paths']).all();B[rec['condition'][:2]]=x
    assert set(packs)=={'S0','S1','S2','S3'} and set(B)=={'S1','S2','S3'}
    # Actual GT loader is delayed until every group, target, fixed Goal and SHA is checked.
    from scripts.stage11b1c.data import truths,manifest
    truth=truths('validation');identity={r['sample_id']:r for r in manifest('validation')};assert set(ids)==set(identity)==set(truth)
    hist=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv').set_index('sample_id')
    oldcohort=pd.read_csv(ROOT/'results/tables/stage11b2d/target_diagnostics.csv').set_index('sample_id')
    manifests={c:pd.read_csv(OUT/f'{c}_tensor_manifest.csv').set_index('sample_id') for c in ['S1','S2','S3']}
    distributions=[];goalsets=[];metrics=[];fixedrows=[];transitions=[];headrows=[]
    for i,sid in enumerate(ids):
        scene=hp['scenes'][i];y=truth[sid];rec=data['original_tensor_records'][i];orig=checked(rec)['tensors']
        om,_,_,_=prediction_metrics(old['paths'][:,i].numpy(),old['goals'][:,i].numpy(),y)
        base=dict(sample_id=sid,scene=scene,history_SHA=identity[sid]['history_SHA'],GT_SHA=identity[sid]['GT_SHA'],
            checkpoint_SHA=CK,original_A25_failure=not bool(om['GoalCover25']))
        for k in ['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group']:base[k]=hist.loc[sid,k]
        assert bool(om['GoalCover25'])==bool(oldcohort.loc[sid,'category25']=='both_hit')
        _,dist0=distribution_stats(orig['normalized_weights'])
        for condition in ['S0','S1','S2','S3']:
            pack=packs[condition];path=pack['paths'][:,i].numpy();goals=pack['goals'][:,i].numpy()
            q,ades,fdes,gfdes=prediction_metrics(path,goals,y)
            if condition=='S0':ds=dict(JS_nats=0.,total_variation=0.,logits_mean_abs_difference=0.,logits_RMS_difference=0.,logits_max_abs_difference=0.,
                **{'Original_'+k:v for k,v in dist0.items()},**{'Alternative_'+k:v for k,v in dist0.items()})
            else:
                rec=manifests[condition].loc[sid];assert sha(ROOT/rec.path)==rec.sha256
                x=torch.load(ROOT/rec.path,map_location='cpu',weights_only=True)['tensors'];ds=compare_distribution(orig,x)
                assert torch.equal(x['goals'][:,0],pack['goals'][:,i]) and torch.equal(x['paths'][:,0],pack['paths'][:,i])
                for head in range(20):headrows.append(dict(sample_id=sid,scene=scene,condition=condition,head=head,ADE=float(ades[head]),FDE=float(fdes[head]),GoalFDE=float(gfdes[head])))
            cs=compare_goals(old['goals'][:,i].numpy(),goals)
            distributions.append(dict(**base,condition=condition,**ds));goalsets.append(dict(**base,condition=condition,**cs))
            metrics.append(dict(**base,condition=condition,experiment='SEMANTIC_TO_GOAL_AND_PATH',**q,**ds,**cs,
                **{'Delta'+k:float(q[k])-float(om[k]) for k in om},Goal_SHA=digest(pack['goals'][:,i:i+1]),Path_SHA=digest(pack['paths'][:,i:i+1])))
            if condition!='S0':
                p=B[condition]['paths'][:,i].numpy();bm,_,_,_=prediction_metrics(p,old['goals'][:,i].numpy(),y);change=compare_fixed_paths(old['paths'][:,i].numpy(),p)
                row=dict(**base,condition=condition,experiment='SEMANTIC_TO_PATH_GIVEN_GOAL',**bm,**change,
                    **{'Original'+k:om[k] for k in om},**{'Delta'+k:float(bm[k])-float(om[k]) for k in om},
                    all20Goals_S0_bitwise=True,all_paths_really_decoded=True,Goal_SHA=digest(B[condition]['goals'][:,i:i+1]),Path_SHA=digest(B[condition]['paths'][:,i:i+1]))
                fixedrows.append(row)
                outcome=('stayed_cover' if q['GoalCover25'] else 'lost') if om['GoalCover25'] else ('recovered' if q['GoalCover25'] else 'stayed_fail')
                transitions.append(dict(**base,condition=condition,OriginalGoalCover25=om['GoalCover25'],AlternativeGoalCover25=q['GoalCover25'],
                    outcome=outcome,OriginalMinADE20=om['minADE20'],AlternativeMinADE20=q['minADE20'],DeltaMinADE20=q['minADE20']-om['minADE20']))
        if (i+1)%200==0:print('CPU_SCORE',i+1,'/998',flush=True)
    mf=pd.DataFrame(metrics);bf=pd.DataFrame(fixedrows);tr=pd.DataFrame(transitions)
    for name,rows in [('goal_distribution_sensitivity.csv',distributions),('goal_set_comparison.csv',goalsets),('full_prediction_metrics.csv',metrics),
        ('fixed_goal_path_metrics.csv',fixedrows),('a25_transition.csv',transitions),('full_head_errors.csv',headrows)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False)
    for name,value in [('ADE1',32.0931),('FDE1',60.5390),('minADE20',16.4031),('minFDE20',25.6340)]:assert abs(mf[mf.condition=='S0'][name].mean()-value)<.0001
    factors=['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group'];scene_rows=[];group_rows=[]
    skipkeys={'original_A25_failure','all20Goals_S0_bitwise','all_paths_really_decoded'}
    def aggregate(q,scope):
        row=dict(scope=scope,targets=len(q),aggregation='target_window_weighted')
        for key in q.select_dtypes(include=[np.number,'bool']).columns:
            if key not in skipkeys:row[key]=float(q[key].mean()) if len(q) else None
        return row
    names=sorted(set(hp['scenes']))
    for frame in [mf,bf]:
        for (experiment,condition),q in frame.groupby(['experiment','condition']):
            individual=[]
            for scope in ['ALL998']+names+['OTHER4']:
                subset=q if scope=='ALL998' else q[q.scene!='deathCircle_3'] if scope=='OTHER4' else q[q.scene==scope]
                row=aggregate(subset,scope);row.update(experiment=experiment,condition=condition);scene_rows.append(row)
                if scope in names:individual.append(row)
            macro=dict(scope='FIVE_RECORDING_MACRO',targets=998,aggregation='equal_recording_weight',experiment=experiment,condition=condition)
            for key in individual[0]:
                if isinstance(individual[0][key],float):macro[key]=float(np.mean([z[key] for z in individual]))
            scene_rows.append(macro)
            for factor in factors:
                for scope in ['ALL998']+names:
                    for group in sorted(hist[factor].unique()):
                        sub=q[(q[factor]==group)&((q.scene==scope) if scope!='ALL998' else True)]
                        row=aggregate(sub,scope);row.update(experiment=experiment,condition=condition,factor=factor,group=group);group_rows.append(row)
    pd.DataFrame(scene_rows).to_csv(OUT/'scene_comparison.csv',index=False);pd.DataFrame(group_rows).to_csv(OUT/'motion_group_comparison.csv',index=False)
    summary=[]
    for condition,q in tr.groupby('condition'):
        for scope in ['ALL998']+names+['OTHER4']:
            sub=q if scope=='ALL998' else q[q.scene!='deathCircle_3'] if scope=='OTHER4' else q[q.scene==scope]
            fail=sub[sub.original_A25_failure];success=sub[~sub.original_A25_failure]
            summary.append(dict(condition=condition,scope=scope,targets=len(sub),original_failures=len(fail),original_successes=len(success),
                recovered=int((sub.outcome=='recovered').sum()),newly_lost=int((sub.outcome=='lost').sum()),
                stayed_fail=int((sub.outcome=='stayed_fail').sum()),stayed_cover=int((sub.outcome=='stayed_cover').sum()),
                total_new_failures=int((~sub.AlternativeGoalCover25).sum()),
                fail164_original_minADE20=float(fail.OriginalMinADE20.mean()),fail164_new_minADE20=float(fail.AlternativeMinADE20.mean()),
                success834_original_minADE20=float(success.OriginalMinADE20.mean()),success834_new_minADE20=float(success.AlternativeMinADE20.mean())))
    pd.DataFrame(summary).to_csv(OUT/'a25_transition_summary.csv',index=False)
    assert all(int(tr[tr.condition==c].original_A25_failure.sum())==164 for c in ['S1','S2','S3'])
    # Fixed case rule: median sample_id per source and per predeclared paired outcome.
    cases=[]
    for scene in names:
        elig=sorted(s for s,source in zip(ids,hp['scenes']) if source==scene);sid=elig[(len(elig)-1)//2]
        cases.append(dict(sample_id=sid,scene=scene,condition='ALL',case_type='scene_median_ID',eligible=len(elig),rank=(len(elig)-1)//2))
    for (condition,outcome),q in tr.groupby(['condition','outcome']):
        q=q.sort_values('sample_id',kind='stable');row=q.iloc[(len(q)-1)//2]
        cases.append(dict(sample_id=row.sample_id,scene=row.scene,condition=condition,case_type=outcome,eligible=len(q),rank=(len(q)-1)//2))
    pd.DataFrame(cases).to_csv(OUT/'qualitative_case_manifest.csv',index=False)
    # Passive feature/logit probes at3 fixed IDs; no extra post-score GPU calls.
    original={r['sample_id']:r for r in json.loads((OUT/'baseline_internal_probes.json').read_text())};probe_rows=[]
    for row in json.loads((OUT/'fixed_internal_probes.json').read_text()):
        base=original[row['sample_id']]
        probe_rows.append(dict(sample_id=row['sample_id'],condition=row['condition'],
            encoder_changed_levels=sum(a['SHA']!=b['SHA'] for a,b in zip(base['encoder_features'],row['encoder_features'])),
            path_logit_changed_heads=sum(a['SHA']!=b['SHA'] for a,b in zip(base['path_logits'],row['path_logits'])),
            encoder_norm_abs_changes=json.dumps([abs(a['norm']-b['norm']) for a,b in zip(base['encoder_features'],row['encoder_features'])]),
            path_logit_norm_abs_changes=json.dumps([abs(a['norm']-b['norm']) for a,b in zip(base['path_logits'],row['path_logits'])])))
    pd.DataFrame(probe_rows).to_csv(OUT/'internal_probe_comparison.csv',index=False)
    # Research labels, not deployment or novelty gates. Predeclared1px scale avoids bitwise-only claims.
    js=[float(mf[mf.condition==c].JS_nats.median()) for c in ['S2','S3']]
    gd=[float(mf[mf.condition==c].Hungarian_mean_distance.median()) for c in ['S2','S3']]
    pdist=[float(bf[bf.condition==c].Path_point_mean_displacement.median()) for c in ['S1','S2','S3']]
    err=[float(mf[mf.condition==c].DeltaminADE20.abs().mean()) for c in ['S1','S2','S3']]
    h2=any(j>=.001 and g>=1 for j,g in zip(js,gd));h3=max(pdist)>=1;h4=max(err)>=1
    verdict=dict(H1='SUPPORTED' if max(gd+pdist)>=1 else 'PARTIALLY_SUPPORTED',H2='SUPPORTED' if h2 else 'PARTIALLY_SUPPORTED',
        H3='SUPPORTED' if h3 else 'PARTIALLY_SUPPORTED',H4='SUPPORTED' if h4 else 'PARTIALLY_SUPPORTED',H5='UNVERIFIED',H6='UNVERIFIED')
    jnew(OUT/'final_decision.json',dict(scope='FROZEN_MODEL_SEMANTIC_INTERVENTION_ONLY',execution='COMPLETE',hypotheses=verdict,
        spatial_condition_median_JS_nats=js,spatial_condition_median_goal_match_px=gd,fixed_condition_median_path_move_px=pdist,
        condition_mean_absolute_paired_ADE_delta_px=err,
        research_recommendation='MATCHED_NO_SEMANTIC_TRAINING_CONTROL_WORTH_PROPOSING' if h3 or h2 else 'NO_GO_FOR_SEMANTIC_AUGMENTATION_WITH_CURRENT_EVIDENCE',
        semantic_quality_correctness='UNVERIFIED; no class map or ground truth segmentation',
        semantic_benefit_independently_proven=False,semantic_error_causes_failure_proven=False,novel_algorithm_GO=False,
        reliability_gate_GO=False,automatic_next_experiment=False,training_steps=0,test_loaded=False,GT_in_generation=False,
        generation_before_GT_scoring=True,targets=998,full_groups=3,fixed_groups=3,source_commit=commit()))
    jnew(OUT/'score_integrity.json',dict(passed=True,all998_targets_and_GT_SHA_matched=True,original164_failure_IDs_unchanged=True,
        all_six_generated_groups_before_GT_read=True,fixed_all_goals_bitwise=True,CPU_seconds=time.perf_counter()-started,
        baseline_exact_recomputed=True,original_checkpoint_SHA=CK,original_prediction_SHA=PSHA))
    jnew(OUT/'gpu_budget.json',dict(actual_GPU_process_seconds=parent_seconds(),hard_seconds=7200,safety_seconds=6600,within_budget=parent_seconds()<6600,
        processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],training_seconds=0))
    print('OFFLINE_ALL998_SCORE_COMPLETE',time.perf_counter()-started,flush=True)
if __name__=='__main__':main()
