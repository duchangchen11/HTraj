"""Existing tensors -> descriptive geometry. No model or sampler is imported."""
from .common import *
from .geometry import assign_final, stats, rms, theory
import numpy as np
import pandas as pd
import torch
import time

def main():
    cpu_guard();start=time.perf_counter()
    cfg=json.loads((CONF/'analysis_protocol.json').read_text())
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'analysis_protocol.json')==frozen['protocol_SHA']
    assert json.loads((OUT/'cache_audit.json').read_text())['passed']
    for row in frozen['numerical_source_SHA']:
        assert sha(ROOT/row['path'])==row['sha256']
    src=pd.read_csv(OLD/'sample_level_mechanism.csv').set_index('sample_id')
    hist=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv').set_index('sample_id')
    prop=pd.read_csv(OLD/'proposal_coverage.csv').set_index('sample_id')
    manifest=pd.read_csv(OLD/'tensor_manifest.csv')
    targets=[];clusters=[];local_rows=[];zero=[]
    for ix,row in enumerate(manifest.to_dict('records')):
        path=ROOT/row['path'];assert sha(path)==row['sha256']
        pack=torch.load(path,map_location='cpu',weights_only=True);t=pack['tensors'];sid=row['sample_id'];base=src.loc[sid]
        P=t['proposals_raw'][0,0].numpy().astype(np.float64)
        C=t['centers_raw'].numpy().astype(np.float64)
        y=np.array([base.GT_x,base.GT_y]);diag=float(base.reference_diagonal)
        assignment,rad=assign_final(P,C,cfg['chunk_size']);dGT=np.linalg.norm(P-y,axis=-1);cGT=np.linalg.norm(C-y,axis=-1)
        local=dGT<=25;touch=np.unique(assignment[local]);pop=np.bincount(assignment,minlength=19)
        localpop=np.bincount(assignment[local],minlength=19)
        nonempty=pop[pop>0];medianpop=float(np.median(nonempty))
        cdist=np.linalg.norm(C[:,None]-C[None,:],axis=-1)[np.triu_indices(19,1)]
        nearest_c=int(np.argmin(cGT));nearest_p=int(np.argmin(dGT));local_n=int(local.sum())
        rr={k:base[k] for k in ['scene','category25','history_SHA','GT_SHA','checkpoint_SHA','probability_SHA',
            'proposal_SHA','indices_SHA','centers_SHA','goal_SHA','GT_x','GT_y','reference_diagonal','P_mass25',
            'ADE1','FDE1','minADE20','minFDE20','FinalMinDist20','KMeans19MinDist','SoftargmaxDist']+cfg['motion_groups']}
        rr.update(sample_id=sid,assignment=cfg['assignment'],ProposalMinDist=float(dGT.min()),
            LocalUniqueLocations25=len(np.unique(P[local],axis=0)),LocalProposalFraction25=local_n/10000,
            LocalAssignedClusterCount25=len(touch),NearestCenterToGT25=float(cGT.min()),
            NearestCenterToNearestProposalDistance=float(np.linalg.norm(C[nearest_c]-P[nearest_p])),
            NearestProposalAssignedCenter=int(assignment[nearest_p]),NearestGTCenterIndex=nearest_c,
            NearestProposalAssignedCenterToGT=float(cGT[assignment[nearest_p]]),
            ProposalRMS=rms(P),CenterRMS=rms(C),CenterPairwiseMean=float(cdist.mean()),
            CenterPairwiseMedian=float(np.median(cdist)),CenterPairwiseMin=float(cdist.min()),CenterPairwiseMax=float(cdist.max()),
            EmptyClusters=int((pop==0).sum()),DuplicateCenterPairs=int((cdist==0).sum()),
            LargestClusterPopulation=int(pop.max()),WithinTargetMedianPopulation=medianpop,
            AssignedClusterPopulation=int(pop[touch].sum()),
            LocalWeightedClusterPopulation=float(pop[assignment[local]].mean()) if local_n else None,
            MaxLocalClusterShare=float(localpop.max()/local_n) if local_n else None,
            LocalDrawsInLargeClustersFraction=float((pop[assignment[local]]>=medianpop).mean()) if local_n else None,
            LocalDrawsInMinorityClustersFraction=float((localpop[assignment[local]]/pop[assignment[local]]<=.1).mean()) if local_n else None,
            LocalCountLE10=local_n<=10,LocalCountLE100=local_n<=100,
            DominantLocalCluster80=bool(localpop.max()/local_n>=.8) if local_n else False)
        for radius in cfg['radii']:
            mask=dGT<=radius
            rr[f'LocalProposalCount{radius}']=int(mask.sum())
            assert int(mask.sum())==int(prop.loc[sid,f'proposal_draws_within{radius}'])
        for k,v in stats(rad[local]).items():rr['LocalToAssignedCenter_'+k]=v
        radii90=[];centroid_res=[]
        for k in range(19):
            sel=assignment==k;nonlocal_sel=sel&~local;r=rad[sel];ss=stats(r)
            center_empirical=float(np.linalg.norm(P[sel].mean(axis=0)-C[k])) if sel.any() else None
            radii90.append(ss['p90']);centroid_res.append(center_empirical)
            cr=dict(sample_id=sid,scene=base.scene,category25=base.category25,assignment=cfg['assignment'],
                center_index=k,center_x=float(C[k,0]),center_y=float(C[k,1]),
                Population=int(sel.sum()),UniqueLocations=len(np.unique(P[sel],axis=0)),PopulationFraction=float(sel.mean()),
                Local25Count=int(localpop[k]),Local25FractionOfCluster=float(localpop[k]/pop[k]) if pop[k] else None,
                CenterDistanceToGT=float(cGT[k]),NonlocalCount=int(nonlocal_sel.sum()),
                NonlocalGTDistanceMean=float(dGT[nonlocal_sel].mean()) if nonlocal_sel.any() else None,
                CenterToAssignedEmpiricalMean=center_empirical,Empty=not bool(sel.any()),
                radius_mean=ss['mean'],radius_median=ss['median'],radius_p90=ss['p90'],radius_max=ss['max'],
                reference_diagonal=diag)
            for k2 in ['CenterDistanceToGT','NonlocalGTDistanceMean','CenterToAssignedEmpiricalMean','radius_mean','radius_median','radius_p90','radius_max']:
                cr[k2+'_diag']=None if cr[k2] is None else cr[k2]/diag
            clusters.append(cr)
        rr['AssignedClusterRadiusP90']=float(np.mean(np.array(radii90,dtype=float)[assignment[local]])) if local_n else None
        rr['MaxCenterToEmpiricalMean']=max(v for v in centroid_res if v is not None)
        if base.category25=='proposal_hit_final_miss':
            for j in np.flatnonzero(local):
                k=int(assignment[j]);local_rows.append(dict(sample_id=sid,scene=base.scene,proposal_index=int(j),
                    x=float(P[j,0]),y=float(P[j,1]),distance_GT=float(dGT[j]),assigned_final_center=k,
                    center_x=float(C[k,0]),center_y=float(C[k,1]),assigned_distance=float(rad[j]),
                    assigned_distance_diag=float(rad[j]/diag),cluster_population=int(pop[k]),cluster_local25=int(localpop[k]),
                    center_distance_GT=float(cGT[k]),assignment=cfg['assignment'],proposal_SHA=base.proposal_SHA,centers_SHA=base.centers_SHA))
        for k in ['ProposalMinDist','NearestCenterToGT25','NearestCenterToNearestProposalDistance','NearestProposalAssignedCenterToGT',
            'ProposalRMS','CenterRMS','CenterPairwiseMean','CenterPairwiseMedian','CenterPairwiseMin','CenterPairwiseMax',
            'LocalToAssignedCenter_mean','LocalToAssignedCenter_median','LocalToAssignedCenter_p90','LocalToAssignedCenter_max','AssignedClusterRadiusP90']:
            rr[k+'_diag']=None if rr[k] is None else rr[k]/diag
        for k in ['hist_speed_diag','hist_mean_turn_degrees','hist_speed_CV','hist_straightness']:
            rr[k]=hist.loc[sid,k]
        targets.append(rr)
        if base.category25=='zero_support_final_miss':
            raw=t['sigmoid'][0,0].numpy().astype(float);th=t['threshold_weights'][0,0].numpy()
            H,W=raw.shape;yy,xx=np.indices((H,W));neigh=(xx/.25-y[0])**2+(yy/.25-y[1])**2<=25**2
            m=float(raw[neigh].sum()/raw.sum());mx=float(raw[neigh].max()) if neigh.any() else 0.
            removed=(raw[neigh]>0)&(th[neigh]==0)
            assert th[neigh].sum()==0 and raw[neigh].sum()>0
            zero.append(dict(sample_id=sid,scene=base.scene,GT_x=y[0],GT_y=y[1],probability_SHA=base.probability_SHA,
                raw_neighborhood_weight_sum=float(raw[neigh].sum()),raw_total_weight_sum=float(raw.sum()),
                unthresholded_normalized_mass=m,neighborhood_max_over_global_max=mx/float(raw.max()),
                neighborhood_pixels=int(neigh.sum()),positive_neighborhood_pixels=int((raw[neigh]>0).sum()),
                removed_positive_neighborhood_pixels=int(removed.sum()),actual_sampling_mass25=0.,relative_threshold=.01,
                unthresholded_theoretical_hit10000=theory(m),counterfactual_status='ANALYTIC_COUNTERFACTUAL_ONLY'))
        if (ix+1)%200==0:print('CPU geometry',ix+1,'/998',flush=True)
    f=pd.DataFrame(targets);cl=pd.DataFrame(clusters)
    assert len(f)==998 and len(cl)==998*19 and len(zero)==77
    for category,ids in cfg['identities'].items():assert sorted(f.loc[f.category25==category,'sample_id'])==ids
    f.to_csv(OUT/'target_diagnostics.csv',index=False)
    cl.to_csv(OUT/'cluster_diagnostics.csv',index=False)
    csvnew(OUT/'local_proposal_assignments.csv.gz',local_rows)
    pd.DataFrame(zero).to_csv(OUT/'threshold_support_review.csv',index=False)
    numeric=[k for k in f.select_dtypes(include=['number','bool']).columns if k not in ['GT_x','GT_y']]
    summaries=[];recordings=[]
    scopes={'ALL998':f,'deathCircle_3':f[f.scene=='deathCircle_3'],'OTHER4':f[f.scene!='deathCircle_3']}
    scopes.update({s:q for s,q in f.groupby('scene')})
    for scope,q in scopes.items():
        nr=dict(scope=scope,targets=len(q),compression_fail=int((q.category25=='proposal_hit_final_miss').sum()),
            both_hit=int((q.category25=='both_hit').sum()),zero_support=int((q.category25=='zero_support_final_miss').sum()),
            positive_sampling_miss=int((q.category25=='positive_support_sampling_miss_final_miss').sum()))
        nr['compression_loss_rate_all']=nr['compression_fail']/len(q)
        nr['compression_loss_rate_proposal_hit']=nr['compression_fail']/(nr['compression_fail']+nr['both_hit'])
        recordings.append(nr)
        for category in cfg['identities']:
            sub=q[q.category25==category]
            for key in numeric:
                a=sub[key].dropna().astype(float)
                rec=dict(scope=scope,category=category,metric=key,targets=len(sub),defined_targets=len(a),mean=float(a.mean()) if len(a) else None)
                rec.update({f'q{int(v*100):02}':float(a.quantile(v)) if len(a) else None for v in cfg['quantiles']})
                summaries.append(rec)
    comp=pd.DataFrame(summaries);comp.to_csv(OUT/'compression_failure_comparison.csv',index=False)
    pd.DataFrame(recordings).to_csv(OUT/'recording_comparison.csv',index=False)
    scene_stats=comp[comp.scope.isin(cfg['recordings'])]
    # Equal recording weight over defined recording-category cells; report nonempty count, never replace an empty category with zero.
    macro=scene_stats.groupby(['category','metric'],sort=True).agg(recordings_defined=('mean','count'),macro_mean=('mean','mean')).reset_index()
    macro.to_csv(OUT/'five_recording_macro.csv',index=False)
    motion=[]
    for factor in cfg['motion_groups']:
        for (scene,group),q in f.groupby(['scene',factor],dropna=False):
            for category in cfg['identities']:
                sub=q[q.category25==category]
                motion.append(dict(scene=scene,factor=factor,group=group,category=category,targets=len(sub),
                    group_total=len(q),rate=len(sub)/len(q),LocalCount25_mean=float(sub.LocalProposalCount25.mean()) if len(sub) else None,
                    LocalCount25_median=float(sub.LocalProposalCount25.median()) if len(sub) else None,
                    CenterGT_diag_mean=float(sub.NearestCenterToGT25_diag.mean()) if len(sub) else None,
                    cluster_population_mean=float(sub.LocalWeightedClusterPopulation.mean()) if len(sub) else None))
    pd.DataFrame(motion).to_csv(OUT/'motion_comparison.csv',index=False)
    corr=[]
    pairs=[('LocalProposalCount25','NearestCenterToGT25'),('LocalProposalCount25','LocalToAssignedCenter_mean'),
           ('ProposalRMS','NearestCenterToGT25'),('LocalWeightedClusterPopulation','LocalToAssignedCenter_mean'),
           ('LocalProposalCount25','P_mass25'),('CenterRMS','NearestCenterToGT25')]
    for category,q in [('ALL998',f)]+list(f.groupby('category25')):
        for a,b in pairs:
            pair=q[[a,b]].dropna();rho=pair.rank().corr().iloc[0,1] if len(pair)>1 and pair[a].nunique()>1 and pair[b].nunique()>1 else np.nan
            corr.append(dict(scope=category,x=a,y=b,targets=len(pair),spearman_rho=None if pd.isna(rho) else float(rho),interpretation='DESCRIPTIVE_NO_P_VALUE'))
    csvnew(OUT/'descriptive_correlations.csv',corr)
    fail=f[f.category25=='proposal_hit_final_miss'];success=f[f.category25=='both_hit'];z=pd.DataFrame(zero)
    d=dict(scope=cfg['scope'],CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),new_model_inference=0,new_training_steps=0,
      assignment=cfg['assignment'],targets=998,clusters=len(cl),local_assignment_rows=len(local_rows),failure_targets=86,
      failure_count25=stats(fail.LocalProposalCount25),success_count25=stats(success.LocalProposalCount25),
      failure_le10=int(fail.LocalCountLE10.sum()),failure_le100=int(fail.LocalCountLE100.sum()),
      failure_single_touched_cluster=int((fail.LocalAssignedClusterCount25==1).sum()),
      failure_dominant80=int(fail.DominantLocalCluster80.sum()),
      failure_large_cluster_local_fraction=stats(fail.LocalDrawsInLargeClustersFraction),
      failure_minority_cluster_local_fraction=stats(fail.LocalDrawsInMinorityClustersFraction),
      failure_local_assignment_distance=stats(fail.LocalToAssignedCenter_mean),
      failure_assigned_cluster_population=stats(fail.LocalWeightedClusterPopulation),
      failure_assigned_radiusP90=stats(fail.AssignedClusterRadiusP90),
      empty_cluster_count=int(cl.Empty.sum()),max_final_center_to_empirical_mean=float(cl.CenterToAssignedEmpiricalMean.max()),
      zero_support_rawmass_quantiles={str(v):float(z.unthresholded_normalized_mass.quantile(v)) for v in cfg['quantiles']},
      zero_support_max_ratio_max=float(z.neighborhood_max_over_global_max.max()),
      figures_case_ids=cfg['case_ids'],analysis_seconds=time.perf_counter()-start,code_commit=commit())
    jnew(OUT/'analysis_summary.json',d)
    print(json.dumps(d,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
