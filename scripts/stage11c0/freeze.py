"""Freeze all numerical generation, metrics and descriptive decision rules before validation inference."""
from .common import *
import shutil,pandas as pd
def main():
    cpu_only();assert json.loads((OUT/'precheck_results.json').read_text())['passed']
    assert json.loads((OUT/'history_before.json').read_text())['passed']
    d=json.loads((CONF/'generation_config.json').read_text());h=checked(d['history'])
    m=pd.read_csv(ROOT/'results/tables/stage11b2c/tensor_manifest.csv');estimate=int(m.bytes.max())*998*3+200*1024**2
    assert shutil.disk_usage(CACHE).free>estimate+10*1024**3
    p=dict(stage='Stage11C-0',scope='FROZEN_MODEL_SEMANTIC_INTERVENTION_ONLY',base_commit=BASE,official_commit=OFFICIAL,
        checkpoint_SHA=CK,original_prediction_SHA=PSHA,protocol='HTraj-YNet-SDD-Development-v1',
        data=dict(targets=998,recordings=sorted(set(h['scenes'])),history=8,future=12,interval_seconds=.4,units='original SDD reference pixels',
            independent_confirmation=False,test_access=False,original_manifest='results/tables/stage11b1c/validation_target_manifest.csv',
            original_manifest_SHA=sha(ROOT/'results/tables/stage11b1c/validation_target_manifest.csv')),
        conditions=dict(S0='Original six-channel semantic tensor, bitwise cache reuse',S1='all six channels and padding zero',
            S2='per-channel float32 CPU spatial mean of valid RGB-resized region; padding original',
            S3='six channels same horizontal flip inside valid RGB-resized region; padding original; no interpolation'),
        class_map='UNVERIFIED: channel0..channel5 only',valid_region='cv2 INTER_AREA fx=fy=.25 actual shape; top-left original image, right/bottom padding to32',
        full_prediction=dict(conditions=['S1','S2','S3'],pipeline='unchanged stage11b2c.AuditPredictor using stage11b1r CPU sampler and original KMeans',
            K=20,proposals=10000,replacement=True,relative_threshold=.01,kmeans_centers=19,temperature=1.,
            categorical_seed='first4 SHA256 bytes big endian of HTraj-YNet-SDD-Development-v1:17:+sample_id',
            kmeans_seed='first4 SHA256 bytes big endian of HTraj-YNet-K20-v2:kmeans:+sample_id',
            same_seed_across_conditions=True,head_order='original1 Softargmax +19 KMeans in returned order, no GT reorder'),
        fixed_goal_prediction=dict(conditions=['S1','S2','S3'],goals='S0 original20 goals, all998 bitwise',
            pipeline='unchanged stage11b2e goal-conditioned decoder math',sampler_calls=0,new_clustering=0,actual_path_decodes=2994*20),
        generation_order=['training-only precheck','this freeze commit','S0 regression3x2','all S1/S2/S3 full groups','all S1/S2/S3 fixed-goal groups','all-six integrity gates','separate CPU GT score'],
        regression_indices=[0,499,997],regression_required='all cached logits/sigmoid/threshold_weights/normalized_weights/indices/proposals/centers/Softargmax/goals/paths bitwise, repeat reverse',
        metrics=dict(goal_radii=[10,25,50],paths=['ADE1','FDE1','minADE20','minFDE20'],ADE_FDE_independent_min_heads=True,
            JS='natural log; original and intervention ACTUAL postthreshold float32 categorical weights normalized mathematically in float64 for analysis; zero terms0',
            entropy='natural log on same full padded-grid categorical distribution, not calibrated uncertainty',
            support='positive categorical pixels; raw-coordinate grid-area=count*16px², not true segmentation area',
            set_distance='Euclidean Hungarian minimum-sum all20 goals; also Chamfer/Hausdorff; head0 separate; no head index mode correspondence',
            fixed_path_difference='same frozen goal/head: all20x12 Euclidean raw-pixel point differences and exact bitwise counts'),
        sensitivity_reporting=dict(numerical_only_scale_px=1e-4,descriptive_practical_scale_px=1.,additional_scales_px=[5.,25.],
            H1='SUPPORTED if max median spatial Goal Hungarian or fixed Path mean displacement >=1px, else PARTIALLY_SUPPORTED',
            H2='SUPPORTED if S2 or S3 median JS>=.001nats AND median Goal Hungarian>=1px, else PARTIALLY_SUPPORTED',
            H3='SUPPORTED if any fixed-goal condition median point displacement>=1px, else PARTIALLY_SUPPORTED',
            H4='SUPPORTED if any condition mean absolute paired minADE20 delta>=1px, else PARTIALLY_SUPPORTED',
            H5='UNVERIFIED: frozen OOD ablations cannot prove trained semantic benefit',H6='UNVERIFIED: no semantic GT or reliable six-channel names',
            recommendation='propose matched-budget no-semantic training control if H2 or H3 supported; no automatic training or new module'),
        group_factors=['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group'],
        group_source='frozen Stage11B-2A motion_features.csv; historical inputs only; all groups and empty cells retained',
        aggregations=['same998 windows','each5 recordings','five-recording equal macro','deathCircle separately','other4 pooled'],
        statistics='descriptive paired; only5 recordings; no998 independent-scene p-values; group/scene confounding explicit',
        fixed_original_failure_cohort='original164 A25 failures,834 successes; never redefine by interventions',
        case_rule='per recording lexicographic median sample_id; additionally for each condition and A25 outcome recovered/lost/stayed_fail/stayed_cover lexicographic median; absent groups explicit absence',
        figure_contract=dict(backend='python',formats=['PNG','SVG'],archetypes=['schematic-led composite','quantitative grid','image plate plus quant'],
            claim='report real sensitivity, beneficial or harmful changes separately; no semantic correctness inference',
            data='same actual caches/paired tables; GT only offline; no fictional heatmaps or paths',
            integrity='no inferred channel labels; global range per channel/scene across inputs, same probability color scale across four conditions'),
        restrictions=['no training','no network/loss/decoder/threshold/K/KMeans change','no GT in generation','no sealed test','no fifth intervention','no extra sampling strategy'],
        GPU=dict(hard_seconds=7200,safety_seconds=6600,full_process_meter=True,precheck_seconds=parent_seconds(),
            precheck_high_estimate_seconds=json.loads((OUT/'precheck_results.json').read_text())['estimated_all_groups_seconds_high']),
        storage=dict(estimated_high_bytes=estimate,free_bytes=shutil.disk_usage(CACHE).free))
    jnew(CONF/'experiment_protocol.json',p)
    sources=sorted((ROOT/'scripts/stage11c0').glob('*.py'))
    sources += [ROOT/x for x in ['scripts/stage11b2e/common.py','scripts/stage11b2e/decoder.py','scripts/stage11b2e/metrics.py',
        'scripts/stage11b2c/adapter.py','scripts/stage11b1r/sampler.py','scripts/stage11b1r/predictor.py','scripts/stage11b1/evaluation.py',
        'scripts/stage11b0/runtime.py','scripts/stage11b0/training_adapter.py','scripts/stage11b1/common.py']]
    jnew(OUT/'protocol_frozen.json',dict(protocol_SHA=sha(CONF/'experiment_protocol.json'),generation_config_SHA=sha(CONF/'generation_config.json'),
        executable_files=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in sources],pre_freeze_commit=commit(),required_successful_commit_before_S0_regression=True))
    print('FREEZE_READY: commit protocol and all executable files before regression',flush=True)
if __name__=='__main__':main()
