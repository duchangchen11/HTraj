"""CPU scoring starts only after complete generation + reverse regression exit."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from .math_audit import probability,neighborhood,coverage,classify
from scripts.stage11b1c.data import truths
import numpy as np,pandas as pd,torch,itertools

METRICS=['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20','SelectionGap_ADE']
CATEGORIES=['zero_support_final_miss','positive_support_sampling_miss_final_miss','proposal_hit_final_miss','both_hit','proposal_miss_final_hit','other']

def summarize(s,scope,r):
    fail=~s.C;den=max(1,len(s));out=dict(scope=scope,radius=r,targets=len(s),P_mass_mean=float(s.P_mass.mean()),P_mass_median=float(s.P_mass.median()),
        support_zero_count=int((~s.A).sum()),positive_mass_sampling_miss_count=int((s.A&~s.B).sum()),proposal_cover_count=int(s.B.sum()),final_cover_count=int(s.C.sum()),
        proposal_cover_fraction=float(s.B.mean()),final_cover_fraction=float(s.C.mean()),compression_loss_count=int((s.B&~s.C).sum()),compression_loss_fraction=float((s.B&~s.C).mean()),
        final_recovery_count=int((~s.B&s.C).sum()),final_fail_count=int(fail.sum()),final_fail_support_zero=int((fail&~s.A).sum()),
        final_fail_positive_sampling_miss=int((fail&s.A&~s.B).sum()),final_fail_after_proposal_hit=int((fail&s.B).sum()),
        theoretical_P_hit_mean=float(s.theoretical_P_hit.mean()),low_mass_reference_count=int((s.theoretical_P_hit<.5).sum()),
        ProposalMinDist_diag_mean=float(s.ProposalMinDist_diag.mean()),FinalMinDist20_diag_mean=float(s.FinalMinDist20_diag.mean()))
    for k in METRICS:out[k]=float(s[k].mean())
    return out

def main():
    assert not torch.cuda.is_initialized();start=time.perf_counter()
    generation=json.loads((OUT/'generation_execution.json').read_text());assert generation['passed'] and generation['generated']==generation['reverse_regressed']==998
    assert json.loads((OUT/'process_extract.json').read_text())['returncode']==0
    frozen=json.loads((OUT/'protocol_frozen.json').read_text());assert sha(CONF/'analysis_protocol.json')==frozen['analysis_protocol_SHA']
    cfg=json.loads((CONF/'analysis_protocol.json').read_text());radii=cfg['radii']
    manifest=pd.read_csv(OUT/'tensor_manifest.csv');assert len(manifest)==998 and manifest.sample_id.nunique()==998
    old=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv').set_index('sample_id');truth=truths('validation')
    assert set(manifest.sample_id)==set(old.index)==set(truth)
    probrows=[];proprows=[];finalrows=[];softrows=[];events=[];samples=[];oldfixed=set(old.index[old.minGoalFDE20>25]);assert len(oldfixed)==164
    for i,row in enumerate(manifest.to_dict('records')):
        sid=row['sample_id'];p=ROOT/row['path'];assert sha(p)==row['sha256'];pack=torch.load(p,map_location='cpu',weights_only=True);x=pack['tensors'];H,W=x['sigmoid'].shape[-2:]
        raw=x['sigmoid'][0,0].numpy();th=x['threshold_weights'][0,0].numpy();weights=x['normalized_weights'].reshape(H,W).numpy();pn=probability(weights)
        gt=np.asarray(truth[sid][-1],dtype=float);ref=pack['reference'];diag=float(np.hypot(ref['width'],ref['height']))
        xy=x['proposals_raw'][0,0].numpy();idx=x['indices'].numpy().ravel()
        assert len(idx)==10000 and np.array_equal(xy,np.stack([idx%W,idx//W],axis=-1)/.25)
        assert (weights.ravel()[idx]>0).all()
        goals=x['goals'][:,0].numpy();centers=x['centers_raw'].numpy();soft=x['softargmax_raw'].numpy()
        assert np.array_equal(goals[0],soft) and np.array_equal(goals[1:],centers)
        pdist,pc=coverage(xy,gt,radii);fdist,fc=coverage(goals,gt,radii);kdist,kc=coverage(centers,gt,radii);sdist,sc=coverage(soft[None],gt,radii)
        assert np.isclose(fdist,float(old.loc[sid,'minGoalFDE20']),rtol=0,atol=1e-8)
        neigh=neighborhood(pn,raw,th,gt,.25,radii);positive=pn[pn>0];entropy=float(-np.sum(positive*np.log(positive)));unique=len(np.unique(idx))
        yy,xx=np.indices((H,W));inside=(xx/.25<ref['width'])&(yy/.25<ref['height']);masspad=float(pn[~inside].sum());n=pn.size;topk=max(1,int(np.ceil(n*.01)))
        base=dict(sample_id=sid,scene=pack['scene'],checkpoint_SHA=CK,probability_SHA=row['probability_SHA'],weights_SHA=row['weights_SHA'],proposal_SHA=row['proposal_SHA'],
            indices_SHA=row['indices_SHA'],centers_SHA=row['centers_SHA'],goal_SHA=row['goal_SHA'],path_SHA=row['path_SHA'],GT_x=gt[0],GT_y=gt[1],history_SHA=old.loc[sid,'history_SHA'],GT_SHA=old.loc[sid,'GT_SHA'],
            reference_diagonal=diag,reference_width=ref['width'],reference_height=ref['height'],canvas_width=W,canvas_height=H,resized_width=ref['resized_width'],resized_height=ref['resized_height'],pad_right=W-ref['resized_width'],pad_bottom=H-ref['resized_height'],resize=.25,
            GT_inside_canvas=bool(0<=gt[0]*.25<W and 0<=gt[1]*.25<H),GT_inside_reference=bool(0<=gt[0]<ref['width'] and 0<=gt[1]<ref['height']),
            nonzero_support_pixels=int((pn>0).sum()),canvas_pixels=n,sigmoid_sum=float(raw.astype(float).sum()),sigmoid_max=float(raw.max()),threshold_weight_sum=float(th.astype(float).sum()),
            original_normalized_weights_float64_sum=float(weights.astype(float).sum()),entropy_nats=entropy,normalized_support_entropy=entropy/np.log(len(positive)) if len(positive)>1 else 0.,effective_support_pixels=float(np.exp(entropy)),max_sampling_probability=float(pn.max()),top_1pct_canvas_mass=float(np.partition(pn.ravel(),n-topk)[-topk:].sum()),padding_probability_mass=masspad,
            unique_proposal_locations=unique,repeated_proposal_draws=10000-unique,ProposalMinDist=pdist,FinalMinDist20=fdist,KMeans19MinDist=kdist,SoftargmaxDist=sdist,
            ProposalMinDist_diag=pdist/diag,FinalMinDist20_diag=fdist/diag,fixed_A25_epoch30=sid in oldfixed)
        for k in METRICS:base[k]=float(old.loc[sid,k])
        for k in cfg['groups']:base[k]=old.loc[sid,k]
        massrow={k:base[k] for k in ['sample_id','scene','checkpoint_SHA','probability_SHA','weights_SHA','GT_x','GT_y','GT_inside_canvas','GT_inside_reference','nonzero_support_pixels','entropy_nats','max_sampling_probability','sigmoid_sum','sigmoid_max','threshold_weight_sum','normalized_support_entropy','effective_support_pixels','top_1pct_canvas_mass','padding_probability_mass','canvas_width','canvas_height','reference_width','reference_height','resize','pad_right','pad_bottom','original_normalized_weights_float64_sum']}
        pr=dict(sample_id=sid,scene=pack['scene'],ProposalMinDist=pdist,unique_proposal_locations=unique,repeated_proposal_draws=10000-unique,proposal_SHA=row['proposal_SHA'],indices_SHA=row['indices_SHA'],sampling_seconds=row['sampling_seconds'])
        fr=dict(sample_id=sid,scene=pack['scene'],FinalMinDist20=fdist,old_minGoalFDE20=float(old.loc[sid,'minGoalFDE20']),goal_SHA=row['goal_SHA'],head_order_unchanged=True)
        sr=dict(sample_id=sid,scene=pack['scene'],SoftargmaxDist=sdist,KMeans19MinDist=kdist,FinalMinDist20=fdist,softargmax_x=float(soft[0]),softargmax_y=float(soft[1]),nearest_center_to_softargmax=float(np.linalg.norm(centers-soft,axis=-1).min()))
        for r in radii:
            nr=neigh[r];A=nr['P_mass']>0;B=pc[r]['cover'];C=fc[r]['cover'];flags,category=classify(A,B,C)
            assert not (not A and B),'Inconsistent probability-grid/proposal geometry'
            massrow.update({f'{k}{r}':v for k,v in nr.items()});pr[f'ProposalCover{r}']=B;pr[f'proposal_draws_within{r}']=pc[r]['draws_in_neighborhood'];fr[f'FinalCover{r}']=C
            sr.update({f'SoftCover{r}':sc[r]['cover'],f'KMeansCover{r}':kc[r]['cover'],f'CombinedCover{r}':C,f'SoftRescuesKMeans{r}':sc[r]['cover'] and not kc[r]['cover']})
            e=dict(**base,radius=r,P_mass=nr['P_mass'],theoretical_P_hit=nr['theoretical_P_hit'],proposal_draws_within=pc[r]['draws_in_neighborhood'],A=A,B=B,C=C,event_code=f'A{int(A)}B{int(B)}C{int(C)}',category=category,**flags);events.append(e)
            if r==25:base.update(P_mass25=nr['P_mass'],theoretical_P_hit25=nr['theoretical_P_hit'],A25=A,B25=B,C25=C,category25=category)
        samples.append(base);probrows.append(massrow);proprows.append(pr);finalrows.append(fr);softrows.append(sr)
        if (i+1)%100==0:print('CPU_SCORE',i+1,'/998',flush=True)
    f=pd.DataFrame(samples);e=pd.DataFrame(events);soft=pd.DataFrame(softrows)
    for r,n in [(10,489),(25,164),(50,99)]:assert int((~e[e.radius==r].C).sum())==n
    for name,rows in [('probability_mass.csv',probrows),('proposal_coverage.csv',proprows),('final_goal_coverage.csv',finalrows),('softargmax_kmeans_comparison.csv',softrows),('coverage_transition.csv',events),('sample_level_mechanism.csv',samples)]:pd.DataFrame(rows).to_csv(OUT/name,index=False)
    scenes=[];motion=[];type_rows=[];combos=[]
    scopes={'ALL_998':e,'deathCircle_3_443':e[e.scene=='deathCircle_3'],'OTHER4_555':e[e.scene!='deathCircle_3']}
    scopes.update({s:g for s,g in e.groupby('scene')})
    for scope,sub in scopes.items():
        for r,s in sub.groupby('radius'):
            scenes.append(summarize(s,scope,int(r)))
            for category in CATEGORIES:
                q=s[s.category==category];n=len(q);within=q.fixed_A25_epoch30.sum();row=dict(scope=scope,radius=r,kind='exclusive_category',type=category,targets=n,fraction=n/len(s),fixed_A25_count=int(within),fraction_of_fixed164=float(within/164))
                row.update({k:float(q[k].mean()) if n else None for k in METRICS});type_rows.append(row)
            for flag in ['T1','T2','T3','T4','T5','T6']:
                q=s[s[flag]];row=dict(scope=scope,radius=r,kind='named_overlapping_indicator',type=flag,targets=len(q),fraction=len(q)/len(s),fixed_A25_count=int(q.fixed_A25_epoch30.sum()),fraction_of_fixed164=float(q.fixed_A25_epoch30.sum()/164))
                row.update({k:float(q[k].mean()) if len(q) else None for k in METRICS});type_rows.append(row)
            for A,B,C in itertools.product([False,True],repeat=3):
                q=s[(s.A==A)&(s.B==B)&(s.C==C)];combos.append(dict(scope=scope,radius=r,A=A,B=B,C=C,event_code=f'A{int(A)}B{int(B)}C{int(C)}',targets=len(q),fraction=len(q)/len(s),fixed_A25_count=int(q.fixed_A25_epoch30.sum())))
    for factor in cfg['groups']:
        for (r,g),s in e.groupby(['radius',factor]):motion.append(dict(factor=factor,group=g,**summarize(s,'historical_group',int(r))))
    pd.DataFrame(scenes).to_csv(OUT/'scene_mechanism_summary.csv',index=False);pd.DataFrame(motion).to_csv(OUT/'motion_mechanism_summary.csv',index=False)
    pd.DataFrame(type_rows).to_csv(OUT/'mechanism_type_summary.csv',index=False);pd.DataFrame(combos).to_csv(OUT/'all_event_combinations.csv',index=False)
    ss=pd.DataFrame(scenes);only=ss[ss.scope.isin(cfg['recordings'])];numeric=['P_mass_mean','proposal_cover_fraction','final_cover_fraction','compression_loss_fraction','theoretical_P_hit_mean']+METRICS
    macro=only.groupby('radius')[numeric].mean().reset_index();macro['recordings']=5;macro['scope']='FIVE_RECORDING_EQUAL_WEIGHT';macro.to_csv(OUT/'recording_macro_summary.csv',index=False)
    s25=e[e.radius==25];failed=s25[~s25.C];counts=failed.category.value_counts().to_dict();largest=max(counts,key=counts.get) if counts else None
    agrees=[]
    for name,s in [('deathCircle',s25[s25.scene=='deathCircle_3']),('other4',s25[s25.scene!='deathCircle_3'])]:
        cc=s[~s.C].category.value_counts();agrees.append(dict(scope=name,counts=cc.to_dict(),largest=str(cc.index[0]) if len(cc) else None))
    unique=len([k for k,v in counts.items() if v==max(counts.values())])==1 if counts else False
    consistent=largest is not None and unique and all(x['largest']==largest and len([v for v in x['counts'].values() if v==max(x['counts'].values())])==1 for x in agrees)
    consistent=consistent and failed[failed.category==largest].scene.nunique()>=3
    recommendation='MECHANISM_INCONCLUSIVE'
    if consistent and largest=='proposal_hit_final_miss':recommendation='PRIORITIZE_FINITE_CANDIDATE_COMPRESSION_AUDIT'
    elif consistent and largest in {'zero_support_final_miss','positive_support_sampling_miss_final_miss'}:recommendation='PRIORITIZE_GOAL_DISTRIBUTION_AUDIT'
    cases=[]
    for category in CATEGORIES:
        eligible=f[f.category25==category].sort_values('sample_id',kind='stable');N=len(eligible)
        for rank in sorted({int(np.floor(q*(N-1))) for q in [.25,.5,.75]}) if N else []:
            row=eligible.iloc[rank].to_dict();row.update(category=category,rank=rank,eligible_targets=N,case=f'{category}_{rank}');cases.append(row)
    pd.DataFrame(cases).to_csv(OUT/'mechanism_case_manifest.csv',index=False)
    result=dict(status='COMPLETE',all998_regression_bitwise=True,analysis_targets=998,checkpoint_SHA=CK,radius=25,
        proposal_covered=int(s25.B.sum()),proposal_coverage=float(s25.B.mean()),final_covered=int(s25.C.sum()),final_coverage=float(s25.C.mean()),
        final_A25_count=len(failed),final_fail_category_counts=counts,final_fail_proposal_previously_covered=int((failed.B).sum()),
        all_targets_zero_mass=int((~s25.A).sum()),final_fail_zero_mass=int((~failed.A).sum()),positive_mass_actual_sampling_miss=int((s25.A&~s25.B).sum()),
        final_fail_positive_mass_sampling_miss=int((failed.A&~failed.B).sum()),compression_loss_count=int((s25.B&~s25.C).sum()),
        final_recovery_without_proposal_hit=int((~s25.B&s25.C).sum()),softargmax_rescues_KMeans25=int(soft.SoftRescuesKMeans25.sum()),
        P_mass25_mean=float(s25.P_mass.mean()),P_mass25_median=float(s25.P_mass.median()),low_mass_analytic_reference_count=int((s25.theoretical_P_hit<.5).sum()),
        recording_mechanisms=agrees,same_dominant_in_deathCircle_and_other4=consistent,recommendation=recommendation,
        mechanism_evidence='SUPPORTED' if recommendation!='MECHANISM_INCONCLUSIVE' else 'PARTIALLY_SUPPORTED',
        algorithm_novelty='UNVERIFIED',direct_semantic_causal_evidence='UNVERIFIED',new_algorithm_GO=False,automatic_next_training_GO=False,
        GPU_seconds=parent_seconds(),CPU_seconds=time.perf_counter()-start,scope='Frozen seed17 Epoch30 reused development998; observational; no test or new performance result')
    jnew(OUT/'final_decision.json',result);jnew(OUT/'gpu_budget.json',dict(new_GPU_seconds=parent_seconds(),hard_seconds=7200,safety_seconds=6600,processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],CPU_analysis_excluded=True,CUDA_initialized=False))
    jnew(OUT/'cache_integrity.json',dict(passed=True,checkpoint_SHA=sha(CP),old_prediction_SHA=sha(PP),tensor_manifest_SHA=sha(OUT/'tensor_manifest.csv'),targets=998,
        all_cached_files_SHA_verified=True,all_original_probability_weights_proposal_goal_path_SHA=True,all998_reverse_bitwise=True,
        original_miss_counts={'10':489,'25':164,'50':99},history_GT_identity_verified=True,original_goals_match_per_sample_atol=1e-8,
        support_and_index_mapping_consistent=True,model_unchanged=generation['parameters_unchanged'],generation_completed_before_GT_scoring=True,
        raw_dtype='float32 logits/sigmoid/threshold/weights/xy;int64 indices; no quantization',offline_probability_float64_analysis_only=True,CPU_only=True,CUDA_initialized=False,source_SHA=sha(Path(__file__))))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
