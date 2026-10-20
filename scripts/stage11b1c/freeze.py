from .common import *

def main():
    a=json.loads((OUT/'data_audit.json').read_text());assert a['passed']
    train=dict(stage='Stage11B-1C',protocol='HTraj-YNet-SDD-Development-v1',status='PROTOCOL_DEVELOPMENT_ONLY',
        base=BASE,official_commit=OFFICIAL,seed=17,history=8,future=12,resize=.25,semantic_classes=6,waypoints=[11],
        optimizer='Adam',learning_rate=.0001,target_batch_size=4,scene_batch_size=1,max_epochs=30,
        augmentation=False,AMP=False,gradient_clipping=None,scheduler=None,loss_scale=1000,
        loss='Original mean BCE(goal_logits,GT heatmap)*1000 + mean BCE(path_logits,GT heatmap)*1000',
        teacher_forcing='Original training GT waypoint only; no validation GT input',
        prediction_initialization='Random seed17, fresh empty optimizer; no author/smoke/one-pass predictor weights',
        segmentation_SHA='75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194',
        segmentation_requires_grad=False,segmentation_mode='eval',segmentation_BN_frozen=True,
        expected_parameters=53155339,expected_predictor_parameters=1641381,
        scene_order='Sorted25 recordings; torch.randperm with independent generator seed17+epoch-1',
        within_scene_order='Unchanged source target row order',targets_per_epoch=7496,steps_per_epoch=1882,
        validation_epochs=[1,5,10,15,20,25,30],
        best='Minimum COMPLETE validation window-weighted minADE20, exact tie earlier epoch',
        save='Initial + all completed epoch boundaries; best/last immutable-file references in manifest',
        TF32=False,deterministic_algorithms=True,artifacts=a['artifacts'],limits=a['limits'])
    inherited=json.loads((ROOT/'configs/stage11b1/evaluation.json').read_text())
    evaluation=dict(protocol=train['protocol'],status=train['status'],K=20,TTST=True,TTST_proposals=10000,
        temperature=1.,rel_threshold=.01,CWS=False,Ke=20,Ka=1,units='original reference image pixels',
        sampler='scripts/stage11b1r/sampler.py',predictor='scripts/stage11b1r/predictor.py',
        sampler_protocol_SHA=sha(ROOT/'configs/stage11b1r/sampling_protocol.json'),
        sampler_SHA=sha(ROOT/'scripts/stage11b1r/sampler.py'),predictor_SHA=sha(ROOT/'scripts/stage11b1r/predictor.py'),
        operation='Per-sample_id CPU Generator and isolated original KMeans RNG; single-process serial',
        GT_scoring='Only after all998 history-only predictions and completeness gates saved',
        runtime_repeat='Indices997/499/0 repeated in reverse with cached and recomputed semantics, bitwise mandatory',
        metrics=['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20'],
        oracle='Independent ADE and FDE best complete path; first candidate is softargmax goal path, not a distribution expectation',
        diversity='Mean pairwise endpoint distance and mean pairwise trajectory ADE; near-duplicate endpoint fraction<=1 pixel, descriptive only',
        goal_coverage_threshold_pixels=[10,25,50],qualitative_selection=inherited['qualitative_selection'],
        baseline_comparison='YNet ADE1/FDE1 vs frozen same998-target CV-Last2/OLS4/OLS8; K20 separate oracle budget',
        plotting=dict(backend='python',formats=['PNG','SVG'],source_data_required=True,no_smoothing=True,
            conclusion='Describe actually recorded learning,validation and selected error cases; no innovation claim',
            risk='Single seed,overlapping windows,shared backgrounds and semantic pretraining exposure; no independent test claim'))
    budget=dict(hard_seconds=28800,safety_seconds=27000,process_charge='Whole GPU worker wall including Python startup,loads,checks,train,val,checkpoints',
        final_cleanup_reserve_seconds=120,planning_training_seconds_per_epoch=453.825087123,
        planning_validation_seconds_per_target=.5178994235004211,planning_validation_overhead_factor=1.1,
        planning_fixed_overhead_seconds=600,
        before_epoch='Reserve max(1.25x recent epoch seconds,previous reference) + due validation max(1.25x last full validation,reference)',
        per_batch_or_target_guard_seconds=30,on_preestimate_over8hours='Stop before formal training',
        on_safety_threshold='Save boundary or failure checkpoint; stop without automatic restart',
        on_numeric_or_identity_or_repro_failure='Stop related experiment; no numerical/hyperparameter repair mid-run')
    jnew(CONF/'train.json',train);jnew(CONF/'evaluation.json',evaluation);jnew(CONF/'budget.json',budget)
    print('Train,evaluation,budget and qualitative rules frozen; no validation feedback',flush=True)

if __name__=='__main__':main()
