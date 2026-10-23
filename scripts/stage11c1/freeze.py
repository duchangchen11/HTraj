"""Freeze the requested numerical experiment and its pretraining budget rejection."""
from .common import *
import numpy as np

def main():
    pre=json.loads((OUT/'preflight_results.json').read_text());assert pre['passed']
    old=ROOT/'results/tables/stage11b1c'
    oldepochs=list(csv.DictReader((old/'epochs.csv').open()))
    oldvals=[json.loads(p.read_text())['total_seconds'] for p in old.glob('validation_epoch*.json')]
    used=parent_seconds();epoch=max(pre['projected_epoch_seconds'],max(float(r['train_seconds']) for r in oldepochs))
    guard_epoch=epoch*1.05;guard_val=pre['maximum_K20_target_seconds']*998*1.1
    reserve=180
    budget=dict(already_used_GPU_seconds=used,reference_full_run_seconds=16198.880032846995,
        measured_epoch=False,measured_validation998=False,measured_K20_unique_targets=2,measured_K20_calls=6,
        projected_epoch_seconds=pre['projected_epoch_seconds'],guard_epoch_seconds=guard_epoch,
        guard_validation_seconds=guard_val,save_exit_startup_reserve_seconds=reserve,
        conservative_all_validations_seconds=used+30*guard_epoch+7*guard_val+reserve,
        first_validation_slow_rest_historical_high_seconds=used+30*guard_epoch+guard_val+6*max(oldvals)*1.1+reserve,
        historical_only_optimistic_seconds=used+sum(float(r['train_seconds']) for r in oldepochs)+sum(oldvals)+reserve,
        fit_30_epochs_and7_validations=False,hard_seconds=18000,safety_seconds=16800,
        decision='PRETRAINING_BUDGET_NO_GO',
        limitations='Two training probes from a temporary early model are not a validation timing estimate or a lower bound. '
            'KMeans depends on the trained distribution; later convergence may reduce cost. Conservative admission is unsupported.',
        method='Fixed 5% epoch margin; 10% target evaluation margin;180s startup/save/exit reserve. '
            'Even only first validation at smoke speed with remaining six at historical maximum exceeds hard budget.')
    assert budget['first_validation_slow_rest_historical_high_seconds']>18000
    jnew(OUT/'budget_preestimate.json',budget)
    train=json.loads((ROOT/'configs/stage11b1c/train.json').read_text())
    train.update(stage='Stage11C-1',base=BASE,experiment='MATCHED_TRAINING_CONTROL_ONLY',model='NoSemantic-YNet',
        semantic_input='All six channels exactly float32 zero including padding, throughout train and inference',
        semantic_shape_source='configs/stage11c1/scene_shapes.json',segmentation_forward_forbidden=True,
        initialize_sha=INITIAL,initialize_checkpoint_loaded=False,new_Adam_empty=True,
        GPU_hard_seconds=18000,GPU_safety_seconds=16800,per_operation_save_exit_reserve_seconds=120,
        scene_order_SHA='SHA256(newline.join(scene names)); no trailing newline',
        visit_order_SHA='Original SHA256(newline.join(sample_ids)); no trailing newline',
        budget_admission='Rejected before formal initialization; no epoch or validation can start')
    ev=json.loads((ROOT/'configs/stage11b1c/evaluation.json').read_text())
    ev.update(runtime_repeat='Indices997/499/0 repeated in reverse and recreated full-canvas zero; bitwise mandatory',
        prediction_history='Physically GT-free Stage11B2C history-only cache',
        primary_research_metric='Five-recording equal macro minADE20, fixed Epoch30 vs Epoch30',
        secondary_comparison='Semantic Epoch30 (old best) versus NoSemantic best COMPLETE validation',
        difference_signs={'ADE':'NoSemantic-Semantic','FDE':'NoSemantic-Semantic','GoalCover':'Semantic-NoSemantic'},
        screening={'ADE_macro_relative_improvement_min':.02,'FDE_macro_semantic_relative_degradation_max':.01,
            'recordings_favor_semantic_min':3,'OTHER4_pooled_semantic_ADE_relative_degradation_max':.01,
            'denominator':'NoSemantic metric for all relative differences',
            'missing':'UNKNOWN; never passed, failed, or zero-imputed from an unrun model'},
        case_selection={'improved':'lexicographically first sample_id with Delta_ADE>=1px',
            'worse':'first sample_id with Delta_ADE<=-1px','close':'first abs(Delta_ADE)<1px',
            'both_fail':'first both minGoalFDE20>25','cover_change':'first disagreeing Cover25',
            'scene':'first matching sample_id in each of five recordings; absent cohort not invented'},
        inference_GT='Never used; oracle heads for offline metrics only',no_test=True)
    jnew(CONF/'train_control.json',train);jnew(CONF/'evaluation.json',ev)
    paths=list((ROOT/'scripts/stage11c1').rglob('*.py'))
    # Freeze the original numerical operators explicitly in addition to new adapters.
    paths += [ROOT/n for n in ['scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py',
        'scripts/stage11b1c/engine.py','scripts/stage11b1c/data.py','scripts/stage11b1r/predictor.py','scripts/stage11b1r/sampler.py']]
    frozen=dict(stage='Stage11C-1',base=BASE,train_config_SHA=sha(CONF/'train_control.json'),
        evaluation_config_SHA=sha(CONF/'evaluation.json'),initial_predictor_SHA=INITIAL,
        data_manifest_SHA={s:sha(ROOT/f'results/tables/stage11b1c/{s}_target_manifest.csv') for s in ['train','validation']},
        zero_protocol_SHA=sha(CONF/'scene_shapes.json'),budget=budget,
        semantic_baseline={'checkpoint':'experiments/stage11b1c/cache/checkpoints/epoch_030.pt',
            'checkpoint_SHA':'0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312',
            'predictions':'experiments/stage11b1c/cache/predictions/epoch030.pt',
            'prediction_SHA':'2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab'},
        executable_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(set(paths))],
        numerical_execution_authorized=False,reason='Conservative budget admission failed; freeze does not authorize training')
    for p,k in [('checkpoint','checkpoint_SHA'),('predictions','prediction_SHA')]:assert sha(ROOT/frozen['semantic_baseline'][p])==frozen['semantic_baseline'][k]
    jnew(OUT/'protocol_frozen.json',frozen)
    jnew(OUT/'training_execution.json',dict(status='STOPPED_BUDGET',phase='PRETRAINING_ADMISSION',formal_training_started=False,
        completed_epochs=0,global_step=0,optimizer_updates=0,completed_validation_epochs=[],best_epoch=None,best_metrics=None,
        test_loaded=False,validation_GT_used_in_prediction=False,GPU_process_seconds=used,reason='Conservative budget exceeds18000s',
        temporary_preflight_model_is_not_formal_initialization=True,initialization_audit='preflight only'))
    jnew(OUT/'checkpoint_manifest.json',dict(checkpoints=[],best=None,best_epoch=None,last=None,
        formal_initial_checkpoint=None,temporary_preflight_checkpoint=pre['checkpoint'],reason='No formal model created'))
    for name,fields in {'epochs.csv':['epoch','global_step','train_loss','goal_loss','path_loss'],
        'training_steps.csv.gz':['epoch','global_step','loss','targets'],
        'validation_metrics.csv':['epoch','status','ADE1','FDE1','minADE20','minFDE20','GoalCover25']}.items():
        f,w=open_csv(OUT/name,fields);f.close()
    jnew(OUT/'gpu_budget.json',dict(hard_seconds=18000,safety_seconds=16800,actual_GPU_related_process_wall_seconds=used,
        processes=[json.loads(p.read_text()) for p in sorted(OUT.glob('process_*.json'))],formal_run_started=False,
        remaining_hard_seconds=18000-used,scope='Whole GPU subprocess wall, including startup,build/load,train probes,K20,save,exit'))
    print('BUDGET_NO_GO',budget,flush=True)
if __name__=='__main__':main()
