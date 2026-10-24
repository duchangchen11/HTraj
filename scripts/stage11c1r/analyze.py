"""CPU-only runtime summaries and explicitly conditional full-run extrapolations."""
from .common import *
import numpy as np,pandas as pd,math

def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(n=0,mean=None,median=None,P90=None,minimum=None,maximum=None,total=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)),
        minimum=float(x.min()),maximum=float(x.max()),total=float(x.sum()))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    used=parent_seconds();combined=ORIGINAL_USED+used;remaining=18000-combined
    ledger=dict(OriginalStage11C1Used=ORIGINAL_USED,Stage11C1RUsed=used,CombinedUsed=combined,CombinedRemaining=remaining,
        new_hard=2400,new_safe=2250,combined_hard=18000,combined_original_safe_reference=16800,
        original_safe_remaining_reference=16800-combined,
        passed=used<=2400 and combined<=18000,
        processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],
        GPU_worker_wall_includes_CPU_diagnostics=True,CPU_only_reporting_and_pytest_no_GPU=True)
    jnew(OUT/'gpu_budget.json',ledger)
    diag=json.loads((OUT/'diagnostic_epoch1_execution.json').read_text())
    if not (OUT/'all_prediction_runtime.csv').exists():
        jnew(OUT/'final_decision.json',dict(ENGINEERING_AUDIT='NO_GO',KMEANS_ROOT_CAUSE='UNVERIFIED',
            BUDGET_ADMISSION='UNCERTAIN',PROTOCOL_COMPATIBILITY='PRESERVED',FORMAL_TRAINING='NOT_AUTHORIZED',
            reason='Profiling not completed',new_metrics_computed=False));return
    allr=pd.read_csv(OUT/'all_prediction_runtime.csv')
    iterations=pd.read_csv(OUT/'kmeans_iteration_counts.csv')
    raw=pd.read_csv(OUT/'kmeans_runtime.csv')
    cpu=pd.read_csv(OUT/'cpu_resident_diagnostic.csv')
    observed=iterations[iterations['mode']=='observed'].copy()
    selected=allr[allr['mode'].isin(['direct','direct_repeat'])]
    keys=['state','role','sample_id','scene']
    quantities=['feature_seconds','sampling_seconds','kmeans_seconds','decode20_seconds','total_seconds',
        'outer_prediction_seconds','cpu_process_seconds']
    paired=selected.groupby(keys,sort=False)[quantities].mean().reset_index()
    paired=paired.merge(observed[['state','sample_id','iterations','stop_reason']],on=['state','sample_id'],validate='one_to_one')
    paired=paired.merge(raw[['state','sample_id','wall_seconds','seconds_per_iteration']],on=['state','sample_id'],validate='one_to_one')
    paired.to_csv(OUT/'paired_runtime_by_target.csv',index=False,lineterminator='\n')
    summaries=[];scene_rows=[]
    for state,group in paired.groupby('state',sort=False):
        summaries.append(dict(state=state,targets=len(group),**{f'{metric}_{k}':v for metric in quantities+['iterations','wall_seconds','seconds_per_iteration']
            for k,v in distribution(group[metric]).items()}))
        for scene,g in group.groupby('scene',sort=False):
            scene_rows.append(dict(state=state,scene=scene,targets=len(g),**{f'{metric}_{k}':v for metric in ['outer_prediction_seconds','wall_seconds','iterations']
                for k,v in distribution(g[metric]).items()}))
    cnew(OUT/'state_summary.csv',summaries);cnew(OUT/'scene_runtime_summary.csv',scene_rows)
    devices=observed[['proposal_device','initial_center_device','input1_device','input2_device','requested_device','output_device',
        'choice_cluster_device','center_update_device','dtype']].drop_duplicates().to_dict('records')
    merged=raw.merge(cpu,on=['state','sample_id','scene'],suffixes=('_original','_CPU'),validate='one_to_one')
    variable=raw['iterations'].nunique()>1 and raw['wall_seconds'].nunique()>1
    correlation=float(np.corrcoef(raw['iterations'],raw['wall_seconds'])[0,1]) if variable else None
    device_audit=dict(observed_paths=devices,original_dtype='float32',
        original_unmonitored_isolated_calls=len(raw),CPU_unmonitored_diagnostic_calls=len(cpu),
        CPU_observed_repeat_calls=len(cpu),CPU_total_function_calls=2*len(cpu),
        GPU_full_predictor_calls=len(allr)+6,GPU_observed_predictor_calls=len(iterations),
        raw_KMeans_seconds=distribution(raw.wall_seconds),raw_iterations=distribution(raw.iterations),
        original_seconds_per_iteration=distribution(raw.seconds_per_iteration),
        CPU_seconds_per_iteration=distribution(cpu.seconds_per_iteration),
        CPU_centers_bitwise_count=int(cpu.centers_bitwise.sum()),CPU_centers_total=len(cpu),
        CPU_center_max_abs_difference=float(cpu.center_max_abs_difference.max()),
        explicit_D2H_bytes_per_iteration=80152,
        explicit_D2H_estimated_total_bytes=int(raw.explicit_D2H_payload_bytes.sum()),
        explicit_transfer_bytes_are_payload_estimates_not_profiler_measurements=True,
        H2D_selection_mask_or_index_payload='UNVERIFIED; backend-dependent CUDA indexing of CPU choices',
        transfer_time_isolated='UNVERIFIED',fixed_fraction_transfer_claim=False,
        hypothetical1000_iterations_seconds=float(raw.seconds_per_iteration.median()*1000),
        historical5second_iteration_count='UNVERIFIED; old per-call iteration trace not recorded',
        CPU_path_comparison_caveats=['Both timings unmonitored; CPU iteration counts observed in separate identical repeat.',
            'Device transfer, kernel launch/synchronization, selection, reduction implementation all change jointly.',
            'Even bitwise agreement on15samples cannot establish equivalence on998 or other checkpoints.'],
        iteration_wall_Pearson_descriptive=correlation,p_value_computed=False,
        time_per_iteration_all_states=distribution(raw.seconds_per_iteration),
        no_formal_algorithm_replacement=True)
    jnew(OUT/'device_execution_audit.json',device_audit)
    # Historical full-run measurements are frozen records, not new validation scoring.
    old=ROOT/'results/tables/stage11b1c';ep=pd.read_csv(old/'epochs.csv')
    vals=[json.loads(p.read_text()) for p in sorted(old.glob('validation_epoch*.json'))]
    old_train=float(ep.train_seconds.sum());old_val=sum(v['total_seconds'] for v in vals)
    old_whole=json.loads((old/'training_execution.json').read_text())['GPU_process_seconds']
    ck=diag.get('checkpoint',{});initial=json.loads((OUT/'diagnostic_initialization.json').read_text())
    save_seconds=max(ck.get('save_and_hash_seconds',0),initial['initial_checkpoint']['save_and_hash_seconds'])
    train_seconds=diag.get('train_seconds')
    counts=pd.read_csv(ROOT/'results/tables/stage11b1c/validation_target_manifest.csv').groupby('scene').size()
    overhead=max(v['total_seconds']-v['prediction_manifest']['prediction_seconds'] for v in vals)
    startup=max(30.,old_whole-old_train-old_val)
    recovery=120.;exit_reserve=60.
    scenarios=[]
    scenarios.append(dict(name='A_HISTORICAL',kind='Historical measured components plus explicit reserves',
        training_seconds=old_train,validation_seconds=old_val,checkpoint_seconds=0.,checkpoint_note='Already inside historical epoch times',
        startup_seconds=startup,recovery_reserve_seconds=recovery,exit_reserve_seconds=exit_reserve,
        timing_margin_seconds=0.,total_seconds=old_train+old_val+startup+recovery+exit_reserve,
        samples='30historicalepochs+7full998validations',interpolation_used=False,
        assumptions='Semantic timings only; no guarantee for new NoSemantic distributions'))
    required={'NoSemantic-initial','NoSemantic-epoch001'}
    if train_seconds is not None and required.issubset(set(paired.state)):
        def project(state,quantile):
            g=paired[paired.state==state];scene={}
            for name,n in counts.items():
                x=g[g.scene==name].outer_prediction_seconds
                assert len(x)==3
                scene[name]=float(x.mean() if quantile=='mean' else x.quantile(.9))*int(n)
            return scene
        after=project('NoSemantic-epoch001','mean');init90=project('NoSemantic-initial','P90');after90=project('NoSemantic-epoch001','P90')
        lastmax=float(paired[paired.state=='NoSemantic-epoch001'].outer_prediction_seconds.max())
        slowmax=float(paired[paired.state.isin(required)].outer_prediction_seconds.max())
        Bval=7*(sum(after.values())+6*lastmax+overhead)
        Cval=7*(sum(max(init90[s],after90[s]) for s in counts.index)+6*slowmax+overhead)
        for label,val,margin,assumption,details in [
            ('B_POST_EPOCH1',Bval,.10,'All7later validations use postdiagnosticEpoch1 mean; later checkpoint behavior unmeasured',after),
            ('C_SLOW_STRESS',Cval,.20,'All7validations retain slower initial/epoch1 per-sceneP90; conditional stress, not empirical runtime bound',
             {s:max(init90[s],after90[s]) for s in counts.index})]:
            tr=30*train_seconds;ch=31*save_seconds;extra=(tr+val)*margin
            total=tr+val+ch+startup+recovery+exit_reserve+extra
            scenarios.append(dict(name=label,kind='EXTRAPOLATION_NOT_MEASURED_FULL_RUN',training_seconds=tr,validation_seconds=val,
                checkpoint_seconds=ch,startup_seconds=startup,recovery_reserve_seconds=recovery,exit_reserve_seconds=exit_reserve,
                timing_margin_seconds=extra,total_seconds=total,one_validation_scene_projected_seconds=details,
                samples='1newfulltrainingepoch;3fixedhistorytargets/scene,15unique perstate;2unmonitored fullpath calls/target',
                interpolation_used=False,assumptions=assumption))
    for s in scenarios:
        s.update(CombinedRemaining=remaining,shortfall_seconds=max(0.,s['total_seconds']-remaining),
            proposed_combined_budget_seconds=combined+s['total_seconds'],
            proposed_additional_budget_over18000=max(0.,combined+s['total_seconds']-18000),
            fits_current_hard_budget=s['total_seconds']<=remaining,
            fits_unchanged_original_safe_reference=s['total_seconds']<=16800-combined)
    totals=[s['total_seconds'] for s in scenarios]
    budget_status='SUFFICIENT' if len(scenarios)==3 and max(totals)<=16800-combined else (
        'INSUFFICIENT' if len(scenarios)==3 and min(totals)>remaining else 'UNCERTAIN')
    safe=math.ceil((combined+max(totals))/1800)*1800
    jnew(OUT/'training_budget_scenarios.json',dict(scenarios=scenarios,ledger=ledger,status=budget_status,
        scenarios_are_not_probabilistic_confidence_intervals=True,interpolation_not_used=True,
        conservative_stress_combined_budget_seconds=safe,stress_assumptions_required=True,
        primary_sampling_limitation='3deterministiclexicographicstrata per recording, not a random population sample or full998validation',
        full_training_completed=False,formal_training_authorized=False))
    complete=json.loads((OUT/'instrumentation_equivalence.json').read_text())
    engineering=diag.get('status')=='COMPLETED_DIAGNOSTIC_EPOCH1' and complete['passed'] and ledger['passed']
    jnew(OUT/'final_decision.json',dict(ENGINEERING_AUDIT='GO' if engineering else 'NO_GO',
        KMEANS_ROOT_CAUSE='PARTIALLY_SUPPORTED' if engineering else 'UNVERIFIED',BUDGET_ADMISSION=budget_status,
        PROTOCOL_COMPATIBILITY='PRESERVED',FORMAL_TRAINING='NOT_AUTHORIZED',
        formal_Stage11C1_epochs=0,diagnostic_epochs=1 if diag.get('epoch')==1 else None,
        measured_new_metrics=False,test_loaded=False,semantic_benefit='NOT_EVALUATED',
        iterations=distribution(raw.iterations),wall_seconds=distribution(raw.wall_seconds),
        seconds_per_iteration=distribution(raw.seconds_per_iteration),iteration_wall_correlation=correlation,
        transfer_time_fraction='UNVERIFIED',CPU_centers_bitwise_count=int(cpu.centers_bitwise.sum()),CPU_comparisons=len(cpu),
        full998_equivalence_not_tested=True,
        recommend_budget_review=budget_status!='SUFFICIENT',
        next_unique_recommendation='User review of unchanged full-run budget/authorization using measuredEpoch1 and conditional scenarios; no automatic training or sampler replacement',
        diagnostic_checkpoint_reuse='Requires separate approval and strict continuation equivalence; not approved here'))
    print('ANALYSIS_COMPLETE',budget_status,'newGPU',used,'remaining',remaining,flush=True)
if __name__=='__main__':main()
